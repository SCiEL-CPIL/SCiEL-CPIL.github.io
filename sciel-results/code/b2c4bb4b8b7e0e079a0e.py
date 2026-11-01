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

def _soft_min_radius(x, beta=50.0):
    """
    Smooth surrogate for negative radius using log-sum-exp soft-min.
    Returns a differentiable approximation of -min(constraints).
    """
    pts = x.reshape(N, 3)
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e6
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    # Upper triangle indices for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_flat = pair_dists[iu]
    
    # Wall distances
    wall_dists = (0.5 - np.abs(pts)).flatten()
    
    all_dists = np.concatenate([pair_flat, wall_dists])
    
    # Stable soft-min: min(d) ≈ d_min - (1/beta)*log(sum(exp(-beta*(d-d_min))))
    d_min = np.min(all_dists)
    shifted = -beta * (all_dists - d_min)
    # Clip to prevent overflow (shifted <= 0)
    shifted = np.clip(shifted, -700, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = d_min - log_sum / beta
    
    return -soft_min_val

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

def _refine_smooth(pts, use_repulsion=True, use_metropolis=True):
    """Refine using optional repulsion annealing, Metropolis MC, smooth surrogate, then exact polish."""
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
    
    # Phase 1: Smooth surrogate with increasing beta
    betas = [10.0, 25.0, 60.0, 150.0, 300.0]
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT - 15.0:
            break
        try:
            def obj(x):
                return _soft_min_radius(x, beta)
            res = minimize(obj, x, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 50})
            if np.all(np.isfinite(res.x)):
                x = res.x
        except Exception:
            pass
    
    # Phase 2: Exact Nelder-Mead polish
    x_2d = x.reshape(N, 3)
    x_2d = np.clip(x_2d, -0.499, 0.499)
    try:
        res = minimize(_neg_radius, x_2d, method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
        if np.all(np.isfinite(res.x)):
            x_2d = res.x
    except Exception:
        pass
    
    return np.clip(x_2d, -0.499, 0.499)

def _refine_exact(pts, use_metropolis=True):
    """Refine using Metropolis MC + exact Nelder-Mead."""
    # Metropolis MC to escape local minima
    if use_metropolis and time.time() - START_TIME < TIME_LIMIT - 40.0:
        pts = _metropolis_mc(pts, n_steps=1500, temp_start=0.04, temp_end=0.003)
        pts = np.clip(pts, -0.499, 0.499)
    
    x = pts.flatten().copy()
    try:
        res = minimize(_neg_radius, x, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
        if np.all(np.isfinite(res.x)):
            x = res.x
    except Exception:
        pass
    return np.clip(x.reshape(N, 3), -0.499, 0.499)

def _final_polish(pts):
    """High-precision final polish with multiple methods."""
    pts = np.clip(pts, -0.499, 0.499)
    best = pts.copy()
    best_r = _compute_radius(best)
    
    # Try Nelder-Mead with very tight tolerance
    if time.time() - START_TIME < TIME_LIMIT - 5.0:
        try:
            res = minimize(_neg_radius, best, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True})
            if np.all(np.isfinite(res.x)):
                r = _compute_radius(res.x)
                if r > best_r:
                    best_r = r
                    best = np.clip(res.x, -0.499, 0.499)
        except Exception:
            pass
    
    # Try smooth surrogate with high beta as final touch
    if time.time() - START_TIME < TIME_LIMIT - 3.0:
        try:
            x = best.flatten().copy()
            bounds = [(-0.499, 0.499)] * (N * 3)
            for beta in [500.0, 1000.0]:
                if time.time() - START_TIME > TIME_LIMIT - 2.0:
                    break
                def obj(x):
                    return _soft_min_radius(x, beta)
                res = minimize(obj, x, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30})
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
    
    # Create diverse initializations
    inits = [
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
    
    # Final high-precision polish on best solution
    if best_pts is not None and time.time() - START_TIME < TIME_LIMIT - 8.0:
        best_pts = _final_polish(best_pts)
        best_r = _compute_radius(best_pts)
    
    if best_pts is None:
        best_pts = _init_grid()
    
    return best_pts

# EVOLVE-BLOCK-END
