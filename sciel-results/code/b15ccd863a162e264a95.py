import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_radius(c):
    """Compute the maximum common radius for configuration c."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(N, dtype=bool)
    min_pair_dist = np.min(dists[mask])
    
    max_coord = np.max(np.abs(c), axis=1)
    min_bound_dist = 0.5 - np.min(max_coord)
    
    return min(min_pair_dist / 2.0, min_bound_dist)

def _softmin_radius_and_grad(c, beta):
    """
    Compute soft-min radius and its gradient.
    Returns (soft_min_val, grad) where grad is (N, 3).
    """
    pts = c.reshape(N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Soft min of pair distances
    mask = ~np.eye(N, dtype=bool)
    dists_flat = dists[mask]
    
    min_d = np.min(dists_flat)
    shifted = dists_flat - min_d
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms)) - beta * min_d
    soft_min_val = -1.0 / beta * log_sum
    
    # Gradient computation
    dists_safe = np.where(dists > 1e-8, dists, 1e-8)
    exp_full = np.exp(-beta * dists)
    np.fill_diagonal(exp_full, 0)
    
    unit_vecs = diff / dists_safe[..., np.newaxis]
    np.fill_diagonal(unit_vecs, 0)
    
    G = np.sum(exp_full[..., np.newaxis] * unit_vecs, axis=1)
    K_val = 0.5 * np.sum(exp_full)
    
    if K_val < 1e-10:
        grad = np.zeros_like(pts)
    else:
        grad = G / K_val
    
    return soft_min_val, grad

def _softmin_obj(c, beta):
    """Objective: minimize -soft_min_radius."""
    val, _ = _softmin_radius_and_grad(c, beta)
    return -val

def _softmin_grad(c, beta):
    """Gradient of -soft_min_radius."""
    _, grad = _softmin_radius_and_grad(c, beta)
    return grad.flatten()

def _repulsion_anneal(c_init, deadline, n_iters=200):
    """
    Anneal a repulsion energy to spread spheres apart.
    Uses soft-min of (r - d_ij) where r is a target radius.
    """
    c = c_init.copy()
    current_time = time.time()
    remaining = deadline - current_time
    
    # Start with a small target and increase
    r_target = 0.05
    r_max = 0.14
    
    for it in range(n_iters):
        if time.time() > deadline - 5:  # Leave 5s margin
            break
        
        # Linearly increase target radius
        progress = it / n_iters
        r_cur = r_target + (r_max - r_target) * progress
        
        # Objective: minimize sum of max(0, r_cur - d_ij)^2 + boundary violations
        def repel_obj(x):
            pts = x.reshape(N, 3)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            mask = ~np.eye(N, dtype=bool)
            dists_flat = dists[mask]
            
            # Penalty for pairs closer than r_cur
            violations = np.maximum(0, r_cur - dists_flat)
            pair_energy = np.sum(violations ** 2)
            
            # Boundary penalty
            max_coord = np.max(np.abs(pts), axis=1)
            bound_viol = np.maximum(0, max_coord - 0.5)
            bound_energy = np.sum(bound_viol ** 2)
            
            return pair_energy + 10 * bound_energy
        
        def repel_grad(x):
            pts = x.reshape(N, 3)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            mask = ~np.eye(N, dtype=bool)
            
            # Gradient of pair energy
            grad = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    d = dists[i, j]
                    if d < r_cur and d > 1e-8:
                        vec = (pts[i] - pts[j]) / d
                        grad[i] += 2 * (r_cur - d) * vec
                        grad[j] -= 2 * (r_cur - d) * vec
            
            # Gradient of boundary energy
            for i in range(N):
                for k in range(3):
                    if pts[i, k] > 0.5:
                        grad[i, k] += 20 * (pts[i, k] - 0.5)
                    elif pts[i, k] < -0.5:
                        grad[i, k] += 20 * (pts[i, k] + 0.5)
            
            return grad.flatten()
        
        try:
            res = minimize(
                repel_obj,
                c.flatten(),
                jac=repel_grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 50, 'ftol': 1e-6}
            )
            c = res.x.reshape(N, 3)
        except Exception:
            continue
    
    return c

def _lloyd_step(c, k=8):
    """
    One Lloyd's algorithm step: for each sphere, move it to the center
    of its Voronoi cell (approximated by averaging k nearest neighbors' midpoints).
    """
    c_new = c.copy()
    
    # Find k nearest neighbors for each point
    for i in range(N):
        dists_to_i = np.linalg.norm(c - c[i], axis=1)
        dists_to_i[i] = np.inf
        k_indices = np.argsort(dists_to_i)[:k]
        
        # Center of mass of nearest neighbors
        center = np.mean(c[k_indices], axis=0)
        
        # Move halfway toward the center
        c_new[i] = c[i] + 0.5 * (center - c[i])
    
    # Clip to cube
    c_new = np.clip(c_new, -0.5, 0.5)
    
    return c_new

def _lloyd_optimize(c_init, deadline, max_iters=100):
    """Run Lloyd's algorithm until deadline."""
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    for it in range(max_iters):
        if time.time() > deadline - 10:
            break
        
        c = _lloyd_step(c)
        r = _compute_radius(c)
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    return best_c

def _softmin_polish(c_init, deadline, beta_start=20, beta_end=100, n_stages=5):
    """
    Polish using soft-min with increasing beta for higher precision.
    """
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    betas = np.linspace(beta_start, beta_end, n_stages)
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(
                _softmin_obj,
                c.flatten(),
                jac=_softmin_grad,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            c = res.x.reshape(N, 3)
        except Exception:
            continue
        
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    return best_c

def _initial_grid():
    """Generate a grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _initial_bcc():
    """Generate a BCC-like initialization."""
    # 3x3x3 grid with body center
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([(i - 1) / 3.0, (j - 1) / 3.0, (k - 1) / 3.0])
                # Add body center offset
                pts.append([(i - 1) / 3.0 + 1/6.0, (j - 1) / 3.0 + 1/6.0, (k - 1) / 3.0 + 1/6.0])
    
    pts = np.array(pts[:N])
    # Scale to fit in cube
    pts = pts * 0.8
    return pts

def _initial_fcc():
    """Generate an FCC-like initialization."""
    pts = []
    # Face-centered cubic lattice
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corners
                pts.append([(i - 1) / 3.0, (j - 1) / 3.0, (k - 1) / 3.0])
                # Face centers
                if i < 2:
                    pts.append([(i + 0.5) / 3.0, (j - 1) / 3.0, (k - 1) / 3.0])
                if j < 2:
                    pts.append([(i - 1) / 3.0, (j + 0.5) / 3.0, (k - 1) / 3.0])
                if k < 2:
                    pts.append([(i - 1) / 3.0, (j - 1) / 3.0, (k + 0.5) / 3.0])
    
    pts = np.array(pts[:N])
    pts = pts * 0.8
    return pts

def _local_search(c_init, deadline, n_restarts=10):
    """
    Local search with small random perturbations.
    """
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    for it in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Small random perturbation
        c_pert = c + np.random.normal(0, 0.005, size=(N, 3))
        c_pert = np.clip(c_pert, -0.5, 0.5)
        
        # Quick polish
        try:
            res = minimize(
                _softmin_obj,
                c_pert.flatten(),
                jac=_softmin_grad,
                args=(50,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            c_opt = res.x.reshape(N, 3)
        except Exception:
            continue
        
        r = _compute_radius(c_opt)
        if r > best_r:
            best_r = r
            best_c = c_opt.copy()
            c = c_opt.copy()
    
    return best_c

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_r = 0.0
    best_c = None
    
    # Multiple initialization strategies
    initializers = [
        _initial_grid,
        _initial_bcc,
        _initial_fcc,
        lambda: np.random.uniform(-0.45, 0.45, size=(N, 3))
    ]
    
    for init_func in initializers:
        if time.time() > deadline - 20:
            break
        
        c_init = init_func()
        
        # Stage 1: Repulsion annealing
        c_annealed = _repulsion_anneal(c_init, deadline - 10)
        
        # Stage 2: Lloyd's algorithm
        c_lloyd = _lloyd_optimize(c_annealed, deadline - 10)
        
        # Stage 3: Soft-min polish
        c_polished = _softmin_polish(c_lloyd, deadline - 5)
        
        # Stage 4: Local search
        c_final = _local_search(c_polished, deadline)
        
        r = _compute_radius(c_final)
        if r > best_r:
            best_r = r
            best_c = c_final.copy()
    
    if best_c is None:
        return _initial_grid()
    
    return best_c
