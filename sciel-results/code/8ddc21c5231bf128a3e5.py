"""Combined program: smooth-surrogate + repulsion annealing + exact-polish for 50-sphere packing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before 120s/150s limit

def _compute_radius(pts):
    """Compute the maximum valid radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 3):
        return -1e6
    if not np.all(np.isfinite(pts)):
        return -1e6
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _neg_radius(x):
    """Objective: negative radius (to be minimized)."""
    pts = x.reshape(N, 3)
    return -_compute_radius(pts)

def _soft_min_radius_and_grad(x, beta=50.0):
    """
    Smooth surrogate for negative radius and its gradient.
    Uses log-sum-exp soft-min.
    """
    pts = x.reshape(N, 3)
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e6, np.zeros_like(x)
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-12)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(dists_sq, np.inf)
    
    pair_dists = dists / 2.0
    iu = np.triu_indices(N, k=1)
    pair_flat = pair_dists[iu]
    
    # Wall distances
    wall_dists = (0.5 - np.abs(pts)).flatten()
    
    all_dists = np.concatenate([pair_flat, wall_dists])
    
    # Stable soft-min: min(d) ≈ d_min - (1/beta)*log(sum(exp(-beta*(d-d_min))))
    d_min = np.min(all_dists)
    shifted = -beta * (all_dists - d_min)
    # Clip to prevent overflow
    shifted = np.clip(shifted, -700, 0)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum = np.log(sum_exp)
    soft_min_val = d_min - log_sum / beta
    
    # Gradient calculation
    # d(soft_min)/d(all_dists) = -exp(-beta*(d-d_min)) / sum_exp = -exp_shifted / sum_exp
    # Note: The term d_min cancels out in the derivative of the log-sum-exp part?
    # Let S = sum(exp(-beta(d_i - d_min))).
    # f = d_min - (1/beta) ln(S)
    # df/d(d_j) = 1 (if j is min) - (1/beta) * (1/S) * dS/d(d_j)
    # This is tricky because d_min depends on d_j.
    # However, the standard soft-min gradient is:
    # grad = -w_j where w_j = exp(-beta(d_j - d_min)) / sum(exp(...))
    # And the sum of w_j is 1.
    # So grad(soft_min) w.r.t d_j is -w_j.
    
    weights = exp_shifted / sum_exp
    # weights corresponds to all_dists
    # We need to map back to x
    
    grad_d = -weights
    
    # Now chain rule to x
    grad_x = np.zeros_like(pts)
    
    # Pairwise part: indices 0 to len(pair_flat)-1
    n_pairs = len(pair_flat)
    pair_grads = grad_d[:n_pairs]
    
    # For each pair (i, j), d_ij = 0.5 * ||p_i - p_j||
    # d(d_ij)/d(p_i) = 0.5 * (p_i - p_j) / ||p_i - p_j||
    # d(d_ij)/d(p_j) = 0.5 * (p_j - p_i) / ||p_i - p_j||
    
    # Vectorized update for pairs
    # iu[0], iu[1] are indices i, j
    i_idx, j_idx = iu
    # Create arrays of gradients for i and j
    # grad_x[i] += pair_grads * 0.5 * diff[i,j] / dists[i,j]
    # grad_x[j] += pair_grads * 0.5 * diff[j,i] / dists[j,i]
    
    # diff is (N, N, 3)
    # dists is (N, N)
    # We only need upper triangle
    # Let's use scatter_add or simple loops if N is small enough (N=50, pairs=1225, loops might be ok but vectorized is better)
    
    # Vectorized:
    # Create a tensor of gradients for all pairs
    # G[i,j] = pair_grads[k] * 0.5 * diff[i,j] / dists[i,j]
    # This is hard to vectorize directly with the flat pair_grads.
    
    # Alternative: Loop over pairs is 1225 iterations, which is fast in Python if simple?
    # Or use np.add.at
    
    # Let's try np.add.at
    # We need to add to grad_x[i] and grad_x[j]
    
    # grad_contrib_i = pair_grads * 0.5 * diff[i_idx, j_idx] / dists[i_idx, j_idx][:, None]
    # grad_contrib_j = -grad_contrib_i
    
    valid_dists = dists[i_idx, j_idx]
    # Avoid div by 0
    valid_dists = np.maximum(valid_dists, 1e-8)
    
    dir_vecs = diff[i_idx, j_idx] / valid_dists[:, None]
    weights_pair = pair_grads * 0.5
    
    grad_i = weights_pair[:, None] * dir_vecs
    grad_j = -grad_i
    
    np.add.at(grad_x, i_idx, grad_i)
    np.add.at(grad_x, j_idx, grad_j)
    
    # Wall part: indices n_pairs to end
    wall_grads = grad_d[n_pairs:]
    # wall_dists = (0.5 - |p_ik|).flatten()
    # d(0.5 - |x|)/dx = -sign(x)
    # So grad_x_ik += wall_grads[k] * (-sign(p_ik))
    
    # wall_grads is shape (N*3,)
    wall_grads_reshaped = wall_grads.reshape(N, 3)
    signs = np.sign(pts)
    # If pts is 0, sign is 0, gradient is undefined, but soft-min handles it.
    # We can use -sign(pts)
    grad_x += wall_grads_reshaped * (-signs)
    
    return -soft_min_val, grad_x.flatten()

def _repulsion_anneal(pts, n_steps=50, lr_start=0.01, lr_end=0.001):
    """
    Repulsion annealing: iteratively push spheres apart using gradient of repulsive potential.
    Vectorized for speed.
    """
    pts = pts.copy().astype(np.float64)
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 30.0:
            break
        lr = lr_start * (lr_end / lr_start) ** (step / max(1, n_steps - 1))
        
        # Compute forces (vectorized)
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion: compute all pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
        dists = np.sqrt(np.sum(diff * diff, axis=-1))  # (N, N)
        np.fill_diagonal(dists, 1e6)  # Avoid division by zero on diagonal
        
        # Force magnitude: 1/dist^2, applied along direction
        # Force on i from j: (1/dist^2) * (pts[i]-pts[j])/dist = diff[i,j] / dist^3
        inv_dist3 = 1.0 / (dists ** 3 + 1e-8)
        # Zero out diagonal
        np.fill_diagonal(inv_dist3, 0)
        
        # forces[i] = sum_j inv_dist3[i,j] * diff[i,j]
        forces += np.sum(inv_dist3[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion: push away from walls
        # For each axis, compute distance to nearest wall
        for k in range(3):
            pos = pts[:, k]
            # Distance to positive wall: 0.5 - pos (for pos > 0)
            # Distance to negative wall: 0.5 + pos (for pos < 0)
            wall_dist_pos = 0.5 - pos
            wall_dist_neg = 0.5 + pos
            
            # Force from positive wall (push left if close to right wall)
            force_pos = np.where(wall_dist_pos < 0.15, -1.0 / (wall_dist_pos + 0.01), 0.0)
            # Force from negative wall (push right if close to left wall)
            force_neg = np.where(wall_dist_neg < 0.15, 1.0 / (wall_dist_neg + 0.01), 0.0)
            
            forces[:, k] += force_pos + force_neg
        
        # Update positions
        pts += lr * forces
        pts = np.clip(pts, -0.495, 0.495)
    
    return pts

def _metropolis_mc(pts, n_steps=2000, temp_start=0.05, temp_end=0.005):
    """
    Metropolis Monte Carlo to escape local minima.
    Tries random perturbations and accepts if radius improves or with probability exp(-delta/temp).
    """
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 30.0:
            break
        
        # Temperature annealing
        t = step / max(1, n_steps - 1)
        temp = temp_start * (temp_end / temp_start) ** t
        
        # Generate perturbation: move a random subset of points
        n_move = np.random.randint(1, 6)  # Move 1-5 points
        idx = np.random.choice(N, size=n_move, replace=False)
        delta = np.random.normal(0, 0.02, size=(n_move, 3))
        
        new_pts = pts.copy()
        new_pts[idx] += delta
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_r = _compute_radius(new_pts)
        
        # Accept if better, or with Metropolis probability
        if new_r >= best_r:
            pts = new_pts
            best_pts = new_pts.copy()
            best_r = new_r
        else:
            delta_r = best_r - new_r
            if np.random.random() < np.exp(-delta_r / max(temp, 1e-10)):
                pts = new_pts
    
    return best_pts

def _local_search(pts, n_steps=2000, step_size_start=0.01, step_size_end=0.001):
    """
    Fast local search: try small random perturbations to individual points.
    Accept if it improves the radius. Much faster than Metropolis since no
    acceptance probability computation needed for non-improving moves.
    """
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 20.0:
            break
        
        # Anneal step size
        t = step / max(1, n_steps - 1)
        step_size = step_size_start * (step_size_end / step_size_start) ** t
        
        # Move one random point
        idx = np.random.randint(N)
        delta = np.random.normal(0, step_size, size=(1, 3))
        
        new_pts = pts.copy()
        new_pts[idx] += delta
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_r = _compute_radius(new_pts)
        
        if new_r > best_r:
            pts = new_pts
            best_pts = new_pts.copy()
            best_r = new_r
    
    return best_pts

def _init_grid():
    """5x5x2 grid with small jitter."""
    xs = (np.arange(5) + 0.5) / 5.0 - 0.5
    ys = (np.arange(5) + 0.5) / 5.0 - 0.5
    zs = (np.arange(2) + 0.5) / 2.0 - 0.5
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    centers += np.random.normal(0, 0.001, size=(N, 3))
    return np.clip(centers, -0.49, 0.49)

def _init_fcc():
    """FCC lattice initialization."""
    a = 1.0 / 3.0
    pts = []
    basis = [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
    for i in range(-1, 4):
        for j in range(-1, 4):
            for k in range(-1, 4):
                for b in basis:
                    p = a * np.array([i, j, k]) + a * np.array(b)
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < N:
        return _init_grid()
    # Greedy selection for spread
    selected = []
    remaining = pts.copy()
    for _ in range(N):
        if len(remaining) == 0:
            break
        # Pick point farthest from already selected
        if not selected:
            idx = 0
        else:
            sel = np.array(selected)
            d = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            idx = np.argmax(d)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    return np.array(selected)

def _init_bcc():
    """BCC lattice initialization."""
    a = 1.0 / 3.0
    pts = []
    basis = [[0,0,0], [0.5,0.5,0.5]]
    for i in range(-1, 4):
        for j in range(-1, 4):
            for k in range(-1, 4):
                for b in basis:
                    p = a * np.array([i, j, k]) + a * np.array(b)
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < N:
        return _init_grid()
    # Greedy selection for spread
    selected = []
    remaining = pts.copy()
    for _ in range(N):
        if len(remaining) == 0:
            break
        if not selected:
            idx = 0
        else:
            sel = np.array(selected)
            d = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            idx = np.argmax(d)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    return np.array(selected)

def _init_random():
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _init_shell_core():
    """
    Initialize with shell-core structure: spheres at corners, edges, face centers,
    then fill interior with remaining spheres. This captures the known optimal
    packing structure for 50 spheres.
    """
    pts = []
    # 8 corners
    for sx in [-0.45, 0.45]:
        for sy in [-0.45, 0.45]:
            for sz in [-0.45, 0.45]:
                pts.append([sx, sy, sz])
    # 12 edge midpoints
    for s in [-0.45, 0.45]:
        for t in [-0.45, 0.45]:
            pts.append([s, t, 0.0])
            pts.append([s, 0.0, t])
            pts.append([0.0, s, t])
    # 6 face centers
    for s in [-0.45, 0.45]:
        pts.append([s, 0.0, 0.0])
        pts.append([0.0, s, 0.0])
        pts.append([0.0, 0.0, s])
    # That's 8 + 12 + 6 = 26 points. Need 24 more in interior.
    # Use a grid in the interior region
    remaining = N - len(pts)
    if remaining > 0:
        # Place remaining in a 3x3x3 grid centered at origin, scaled to fit
        n_per_axis = max(1, int(round(remaining ** (1.0 / 3.0))))
        interior_pts = []
        for i in range(n_per_axis):
            for j in range(n_per_axis):
                for k in range(n_per_axis):
                    if len(interior_pts) >= remaining:
                        break
                    x = (i + 0.5) / n_per_axis - 0.5
                    y = (j + 0.5) / n_per_axis - 0.5
                    z = (k + 0.5) / n_per_axis - 0.5
                    # Scale to fit within [-0.3, 0.3]
                    interior_pts.append([x * 0.6, y * 0.6, z * 0.6])
        # If we still need more, add random points
        while len(pts) + len(interior_pts) < N:
            interior_pts.append(np.random.uniform(-0.3, 0.3, size=3).tolist())
        pts.extend(interior_pts[:remaining])
    
    pts = np.array(pts[:N], dtype=np.float64)
    # Add small jitter to break symmetry
    pts += np.random.normal(0, 0.002, size=(N, 3))
    return np.clip(pts, -0.49, 0.49)

def _refine_smooth(pts, use_repulsion=True, use_metropolis=True):
    """Refine using optional repulsion annealing, Metropolis MC, smooth surrogate with gradients."""
    pts = np.clip(pts, -0.49, 0.49)
    
    # Phase 0: Optional repulsion annealing
    if use_repulsion and time.time() - START_TIME < TIME_LIMIT - 50.0:
        pts = _repulsion_anneal(pts, n_steps=40, lr_start=0.008, lr_end=0.001)
        pts = np.clip(pts, -0.495, 0.495)
    
    # Phase 0.5: Metropolis MC to escape local minima
    if use_metropolis and time.time() - START_TIME < TIME_LIMIT - 40.0:
        pts = _metropolis_mc(pts, n_steps=1500, temp_start=0.04, temp_end=0.003)
        pts = np.clip(pts, -0.499, 0.499)
    
    x = pts.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    # Phase 1: Smooth surrogate with increasing beta and analytic gradient
    betas = [10.0, 25.0, 60.0, 150.0, 300.0, 600.0]
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT - 10.0:
            break
        try:
            def obj(x):
                val, _ = _soft_min_radius_and_grad(x, beta)
                return val
            def jac(x):
                _, g = _soft_min_radius_and_grad(x, beta)
                return g
            res = minimize(obj, x, method='L-BFGS-B', bounds=bounds, jac=jac,
                           options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50})
            if np.all(np.isfinite(res.x)):
                x = res.x
        except Exception:
            pass
    
    x_2d = x.reshape(N, 3)
    x_2d = np.clip(x_2d, -0.499, 0.499)
    
    # Phase 2: Local Active Set Polish
    # Identify critical pairs and walls
    r_current = _compute_radius(x_2d)
    if r_current > 0:
        # Find pairs with distance close to 2*r_current
        diff = x_2d[:, np.newaxis, :] - x_2d[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Critical pairs
        tol = 1e-4
        critical_pairs = np.argwhere(dists < 2.0 * (r_current + tol))
        # Remove diagonal
        critical_pairs = critical_pairs[critical_pairs[:, 0] != critical_pairs[:, 1]]
        # Keep unique (i < j)
        critical_pairs = critical_pairs[critical_pairs[:, 0] < critical_pairs[:, 1]]
        
        # Critical walls
        wall_dists = 0.5 - np.abs(x_2d)
        critical_walls = np.argwhere(wall_dists < r_current + tol)
        
        # Simple gradient step on critical constraints
        # We just do a few L-BFGS-B steps with high beta on the full set,
        # but we can also try to move points away from critical neighbors.
        # For now, rely on the high-beta L-BFGS-B above.
        pass
    
    return np.clip(x_2d, -0.499, 0.499)

def _refine_exact(pts, use_metropolis=True):
    """Refine using Metropolis MC + high-beta soft-min polish."""
    # Metropolis MC to escape local minima
    if use_metropolis and time.time() - START_TIME < TIME_LIMIT - 40.0:
        pts = _metropolis_mc(pts, n_steps=1500, temp_start=0.04, temp_end=0.003)
        pts = np.clip(pts, -0.499, 0.499)
    
    x = pts.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    # High beta polish
    for beta in [500.0, 1000.0]:
        if time.time() - START_TIME > TIME_LIMIT - 10.0:
            break
        try:
            def obj(x):
                val, _ = _soft_min_radius_and_grad(x, beta)
                return val
            def jac(x):
                _, g = _soft_min_radius_and_grad(x, beta)
                return g
            res = minimize(obj, x, method='L-BFGS-B', bounds=bounds, jac=jac,
                           options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-10})
            if np.all(np.isfinite(res.x)):
                x = res.x
        except Exception:
            pass
    return np.clip(x.reshape(N, 3), -0.499, 0.499)

def _final_polish(pts):
    """High-precision final polish using high-beta soft-min."""
    pts = np.clip(pts, -0.499, 0.499)
    best = pts.copy()
    best_r = _compute_radius(best)
    
    # Try smooth surrogate with very high beta as final touch
    if time.time() - START_TIME < TIME_LIMIT - 2.0:
        try:
            x = best.flatten().copy()
            bounds = [(-0.499, 0.499)] * (N * 3)
            for beta in [1000.0, 2000.0]:
                if time.time() - START_TIME > TIME_LIMIT - 1.0:
                    break
                def obj(x):
                    val, _ = _soft_min_radius_and_grad(x, beta)
                    return val
                def jac(x):
                    _, g = _soft_min_radius_and_grad(x, beta)
                    return g
                res = minimize(obj, x, method='L-BFGS-B', bounds=bounds, jac=jac,
                               options={'maxiter': 50, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 30})
                if np.all(np.isfinite(res.x)):
                    x = res.x
                    r = _compute_radius(x.reshape(N, 3))
                    if r > best_r:
                        best_r = r
                        best = np.clip(x.reshape(N, 3), -0.499, 0.499)
        except Exception:
            pass
    
    return best

def construct():
    best_pts = None
    best_r = -np.inf
    
    # Create diverse initializations including shell-core structure
    inits = [
        _init_shell_core(),
        _init_grid(),
        _init_fcc(),
        _init_bcc(),
        _init_random(),
        _init_random(),
    ]
    
    # Add more random inits if time permits
    while time.time() - START_TIME < TIME_LIMIT - 45.0:
        inits.append(_init_random())
    
    for i, init_pts in enumerate(inits):
        if time.time() - START_TIME > TIME_LIMIT - 20.0:
            break
        
        init_pts = np.clip(init_pts, -0.49, 0.49)
        
        # Alternate between different refinement strategies
        if i % 4 == 0:
            refined = _refine_smooth(init_pts, use_repulsion=True, use_metropolis=True)
        elif i % 4 == 1:
            refined = _refine_smooth(init_pts, use_repulsion=False, use_metropolis=True)
        elif i % 4 == 2:
            refined = _refine_exact(init_pts, use_metropolis=True)
        else:
            # Pure repulsion + smooth, no Metropolis (faster)
            refined = _refine_smooth(init_pts, use_repulsion=True, use_metropolis=False)
        
        refined = np.clip(refined, -0.499, 0.499)
        r = _compute_radius(refined)
        
        if r > best_r:
            best_r = r
            best_pts = refined.copy()
    
    # Local search on best solution to escape local minima
    if best_pts is not None and time.time() - START_TIME < TIME_LIMIT - 15.0:
        ls_pts = _local_search(best_pts, n_steps=1000, step_size_start=0.008, step_size_end=0.001)
        ls_r = _compute_radius(ls_pts)
        if ls_r > best_r:
            best_r = ls_r
            best_pts = ls_pts
    
    # Final high-precision polish on best solution
    if best_pts is not None and time.time() - START_TIME < TIME_LIMIT - 8.0:
        best_pts = _final_polish(best_pts)
        best_r = _compute_radius(best_pts)
    
    if best_pts is None:
        best_pts = _init_grid()
    
    return best_pts

# EVOLVE-BLOCK-END
