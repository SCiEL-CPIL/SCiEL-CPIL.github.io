"""Improved solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s limit

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

def _softmin_radius_and_grad(x, beta=100.0):
    """
    Smooth surrogate for negative radius using log-sum-exp soft-min.
    Returns (value, gradient) for use with L-BFGS-B.
    """
    pts = x.reshape(N, 3)
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e6, np.zeros_like(x)
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    iu = np.triu_indices(N, k=1)
    pair_flat = pair_dists[iu]
    
    # Wall distances
    wall_dists = (0.5 - np.abs(pts)).flatten()
    
    all_dists = np.concatenate([pair_flat, wall_dists])
    
    # Stable soft-min
    d_min = np.min(all_dists)
    shifted = -beta * (all_dists - d_min)
    shifted = np.clip(shifted, -700, 0)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum = np.log(sum_exp)
    soft_min_val = d_min - log_sum / beta
    
    # Gradient computation
    grad = np.zeros_like(x)
    
    # Weights for soft-min
    weights = exp_shifted / sum_exp
    
    # Pairwise part
    pair_weights = weights[:len(pair_flat)]
    # Reconstruct pairwise gradient
    for idx in range(len(pair_flat)):
        i, j = iu[0][idx], iu[1][idx]
        d = pts[i] - pts[j]
        dist = dists[i, j]
        if dist < 1e-10:
            continue
        # d(pair_dist)/d(pts[i]) = d/dist, d(pair_dist)/d(pts[j]) = -d/dist
        # d(soft_min)/d(pts[i]) = -pair_weights[idx] * d/dist
        grad[i] += -pair_weights[idx] * d / dist
        grad[j] += pair_weights[idx] * d / dist
    
    # Wall part
    wall_weights = weights[len(pair_flat):]
    # Wall dist = 0.5 - |pts[i,k]|
    # d(wall_dist)/d(pts[i,k]) = -sign(pts[i,k])
    for idx in range(len(wall_weights)):
        i = idx // 3
        k = idx % 3
        sign = 1.0 if pts[i, k] >= 0 else -1.0
        grad[i, k] += wall_weights[idx] * sign
    
    return -soft_min_val, grad

def _epigraph_objective(x, t=0.13):
    """
    Epigraph formulation: minimize sum of violations.
    x: (N*3,) positions, t: radius
    Returns penalty for constraints not satisfied.
    """
    pts = x[:N*3].reshape(N, 3)
    r = x[N*3]
    
    # Pairwise constraints: |ci - cj| >= 2r
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_violations = np.maximum(0, 2*r - dists)
    
    # Wall constraints: |ci,k| + r <= 0.5
    wall_violations = np.maximum(0, np.abs(pts) + r - 0.5)
    
    # Smooth penalty
    penalty = np.sum(pair_violations**2) + np.sum(wall_violations**2)
    return penalty

def _epigraph_grad(x, t=0.13):
    """Gradient of epigraph objective."""
    pts = x[:N*3].reshape(N, 3)
    r = x[N*3]
    
    grad = np.zeros_like(x)
    
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    for i in range(N):
        for j in range(i+1, N):
            d = pts[i] - pts[j]
            dist = dists[i, j]
            if dist < 1e-10:
                continue
            violation = 2*r - dist
            if violation > 0:
                # d(violation^2)/d(pts[i]) = 2*violation * d(violation)/d(pts[i])
                # d(violation)/d(pts[i]) = -d/dist
                grad[i] += 2*violation * (-d / dist)
                grad[j] += 2*violation * (d / dist)
            # Also contribution from r
            # d(violation^2)/dr = 2*violation*2
            if violation > 0:
                grad[N*3] += 2*violation*2
    
    # Wall
    for i in range(N):
        for k in range(3):
            val = np.abs(pts[i, k]) + r - 0.5
            if val > 0:
                sign = 1.0 if pts[i, k] >= 0 else -1.0
                grad[i, k] += 2*val*sign
                grad[N*3] += 2*val
    
    return grad

def _repulsion_anneal(pts, n_steps=60, lr_start=0.01, lr_end=0.001):
    """
    Vectorized repulsion annealing.
    """
    pts = pts.copy().astype(np.float64)
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 40.0:
            break
        lr = lr_start * (lr_end / lr_start) ** (step / max(1, n_steps - 1))
        
        # Compute forces (vectorized)
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Force magnitude: 1/(dist^2 + eps)
        force_mag = 1.0 / (dists**2 + 1e-8)
        np.fill_diagonal(force_mag, 0)
        
        # Force vector = force_mag * diff / dist
        with np.errstate(divide='ignore', invalid='ignore'):
            force_vec = force_mag[:, :, np.newaxis] * diff / dists[:, :, np.newaxis]
        forces = np.sum(force_vec, axis=1)
        
        # Wall repulsion
        for k in range(3):
            wall_dist_pos = 0.5 - pts[:, k]
            wall_dist_neg = 0.5 + pts[:, k]
            # Only apply when close to wall
            close_pos = wall_dist_pos < 0.15
            close_neg = wall_dist_neg < 0.15
            forces[close_pos, k] -= 1.0 / (wall_dist_pos[close_pos] + 0.01)
            forces[close_neg, k] += 1.0 / (wall_dist_neg[close_neg] + 0.01)
        
        # Update
        pts += lr * forces
        pts = np.clip(pts, -0.495, 0.495)
    
    return pts

def _lloyd_equalize(pts, n_iters=10):
    """
    Lloyd's algorithm: move each point toward the centroid of its Voronoi cell.
    This promotes equal spacing.
    """
    pts = pts.copy()
    for _ in range(n_iters):
        if time.time() - START_TIME > TIME_LIMIT - 30.0:
            break
        try:
            # Compute Voronoi cells using Delaunay triangulation
            delaunay = Delaunay(pts)
            # For each point, compute centroid of its Voronoi cell
            # Approximation: use nearest neighbors
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            new_pts = pts.copy()
            for i in range(N):
                # Get 12 nearest neighbors
                dists, indices = tree.query(pts[i], k=13)
                neighbors = pts[indices[1:]]  # Exclude self
                # Weighted centroid (closer neighbors have more weight)
                weights = 1.0 / (dists[1:]**2 + 1e-8)
                centroid = np.sum(neighbors * weights[:, np.newaxis], axis=0) / np.sum(weights)
                # Move halfway toward centroid
                new_pts[i] = 0.5 * pts[i] + 0.5 * centroid
            pts = np.clip(new_pts, -0.495, 0.495)
        except Exception:
            break
    return pts

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
    """Initialize with some points near center, some near corners/edges."""
    pts = []
    # Center cluster
    for _ in range(10):
        pts.append(np.random.normal(0, 0.1, 3))
    # Edge clusters
    edges = [
        [0.4, 0, 0], [-0.4, 0, 0],
        [0, 0.4, 0], [0, -0.4, 0],
        [0, 0, 0.4], [0, 0, -0.4],
    ]
    for e in edges:
        for _ in range(3):
            pts.append(np.array(e) + np.random.normal(0, 0.05, 3))
    # Corner clusters
    corners = []
    for sx in [-0.35, 0.35]:
        for sy in [-0.35, 0.35]:
            for sz in [-0.35, 0.35]:
                corners.append([sx, sy, sz])
    for c in corners:
        for _ in range(3):
            pts.append(np.array(c) + np.random.normal(0, 0.04, 3))
    # Fill remaining
    while len(pts) < N:
        pts.append(np.random.uniform(-0.4, 0.4, 3))
    pts = np.array(pts[:N])
    return np.clip(pts, -0.49, 0.49)

def _smooth_polish(pts):
    """Polish using smooth surrogate with increasing beta."""
    x = pts.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    betas = [50.0, 100.0, 200.0, 400.0, 800.0]
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT - 20.0:
            break
        try:
            def obj(x):
                val, grad = _softmin_radius_and_grad(x, beta)
                return val, grad
            res = minimize(obj, x, method='L-BFGS-B', bounds=bounds,
                           jac=True,
                           options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 60})
            if np.all(np.isfinite(res.x)):
                x = res.x
        except Exception:
            pass
    
    return np.clip(x.reshape(N, 3), -0.499, 0.499)

def _epigraph_polish(pts):
    """Polish using epigraph formulation."""
    r_init = _compute_radius(pts)
    x = np.concatenate([pts.flatten(), [r_init]])
    bounds = [(-0.499, 0.499)] * (N * 3) + [(0.0, 0.5)]
    
    try:
        def obj(x):
            return _epigraph_objective(x)
        def grad(x):
            return _epigraph_grad(x)
        res = minimize(obj, x, method='L-BFGS-B', bounds=bounds, jac=grad,
                       options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 60})
        if np.all(np.isfinite(res.x)):
            new_pts = res.x[:N*3].reshape(N, 3)
            new_r = res.x[N*3]
            # Check if this is valid
            actual_r = _compute_radius(new_pts)
            if actual_r >= new_r - 1e-9:
                return np.clip(new_pts, -0.499, 0.499)
    except Exception:
        pass
    return pts

def _local_search(pts, n_trials=200):
    """Random local search: perturb and accept if better."""
    best = pts.copy()
    best_r = _compute_radius(best)
    
    for _ in range(n_trials):
        if time.time() - START_TIME > TIME_LIMIT - 15.0:
            break
        # Random perturbation
        trial = best + np.random.normal(0, 0.01, size=(N, 3))
        trial = np.clip(trial, -0.499, 0.499)
        r = _compute_radius(trial)
        if r > best_r:
            best_r = r
            best = trial.copy()
    
    return best

def _refine_full(pts):
    """Full refinement pipeline."""
    pts = np.clip(pts, -0.49, 0.49)
    
    # Phase 1: Repulsion annealing
    if time.time() - START_TIME < TIME_LIMIT - 50.0:
        pts = _repulsion_anneal(pts, n_steps=40, lr_start=0.008, lr_end=0.001)
        pts = np.clip(pts, -0.495, 0.495)
    
    # Phase 2: Lloyd equalization
    if time.time() - START_TIME < TIME_LIMIT - 40.0:
        pts = _lloyd_equalize(pts, n_iters=8)
    
    # Phase 3: Smooth polish
    if time.time() - START_TIME < TIME_LIMIT - 25.0:
        pts = _smooth_polish(pts)
    
    # Phase 4: Epigraph polish
    if time.time() - START_TIME < TIME_LIMIT - 15.0:
        pts = _epigraph_polish(pts)
    
    # Phase 5: Local search
    if time.time() - START_TIME < TIME_LIMIT - 10.0:
        pts = _local_search(pts, n_trials=100)
    
    # Phase 6: Final Nelder-Mead polish
    if time.time() - START_TIME < TIME_LIMIT - 5.0:
        try:
            res = minimize(_neg_radius, pts, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if np.all(np.isfinite(res.x)):
                pts = np.clip(res.x, -0.499, 0.499)
        except Exception:
            pass
    
    return np.clip(pts, -0.499, 0.499)

def _metropolis_mc(pts, n_steps=1000, t_start=0.01, t_end=0.001):
    """Metropolis Monte Carlo: random moves accepted with probability exp(-delta/E)."""
    pts = pts.copy()
    best = pts.copy()
    best_r = _compute_radius(best)
    
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 20.0:
            break
        t = t_start * (t_end / t_start) ** (step / max(1, n_steps - 1))
        
        # Random move
        idx = np.random.randint(N)
        trial = pts.copy()
        trial[idx] += np.random.normal(0, 0.02, 3)
        trial = np.clip(trial, -0.499, 0.499)
        
        r_trial = _compute_radius(trial)
        r_cur = _compute_radius(pts)
        
        # Accept if better or with probability exp((r_trial - r_cur)/t)
        if r_trial >= r_cur or np.random.random() < np.exp((r_trial - r_cur) / max(t, 1e-10)):
            pts = trial
            if r_trial > best_r:
                best_r = r_trial
                best = trial.copy()
    
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
        _init_shell_core(),
    ]
    
    # Add more random inits if time permits
    while time.time() - START_TIME < TIME_LIMIT - 60.0:
        inits.append(_init_random())
    
    for i, init_pts in enumerate(inits):
        if time.time() - START_TIME > TIME_LIMIT - 30.0:
            break
        
        init_pts = np.clip(init_pts, -0.49, 0.49)
        
        # Alternate between different refinement strategies
        if i % 3 == 0:
            refined = _refine_full(init_pts)
        elif i % 3 == 1:
            # Smooth polish only
            refined = _smooth_polish(init_pts)
            if time.time() - START_TIME < TIME_LIMIT - 10.0:
                refined = _local_search(refined, n_trials=50)
        else:
            # Metropolis + polish
            refined = _metropolis_mc(init_pts, n_steps=500)
            if time.time() - START_TIME < TIME_LIMIT - 10.0:
                refined = _smooth_polish(refined)
        
        refined = np.clip(refined, -0.499, 0.499)
        r = _compute_radius(refined)
        
        if r > best_r:
            best_r = r
            best_pts = refined.copy()
    
    # Final high-precision polish on best solution
    if best_pts is not None and time.time() - START_TIME < TIME_LIMIT - 15.0:
        # Multiple final polish attempts
        candidates = [best_pts]
        
        # Try local search variations
        if time.time() - START_TIME < TIME_LIMIT - 10.0:
            candidates.append(_local_search(best_pts, n_trials=200))
        
        # Try Metropolis
        if time.time() - START_TIME < TIME_LIMIT - 8.0:
            candidates.append(_metropolis_mc(best_pts, n_steps=300, t_start=0.005, t_end=0.001))
        
        # Pick best candidate
        for cand in candidates:
            cand = np.clip(cand, -0.499, 0.499)
            r = _compute_radius(cand)
            if r > best_r:
                best_r = r
                best_pts = cand.copy()
        
        # Final Nelder-Mead
        if time.time() - START_TIME < TIME_LIMIT - 5.0:
            try:
                res = minimize(_neg_radius, best_pts, method='Nelder-Mead',
                               options={'maxiter': 800, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True})
                if np.all(np.isfinite(res.x)):
                    r = _compute_radius(res.x)
                    if r > best_r:
                        best_r = r
                        best_pts = np.clip(res.x, -0.499, 0.499)
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = _init_grid()
    
    return best_pts

# EVOLVE-BLOCK-END
