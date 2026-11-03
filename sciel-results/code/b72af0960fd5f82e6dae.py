"""Smooth energy-based SGLD + local polish for packing 100 equal spheres
in the unit cube [-1/2, 1/2]^3.

Strategy:
  1. Start from BCC or grid initializations.
  2. Use a smooth energy (log-sum-exp of constraint violations) with SGLD
     to explore the landscape and escape local minima of the hard-min objective.
  3. Iteratively increase target radius, solving smooth constrained optimization.
  4. Polish with direct local search on the exact min-distance objective.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
HALF = 0.5
DEADLINE = 150.0 - 8.0  # safety margin


# ---------------- exact objective ----------------
def _radius(pts):
    """Exact feasible common radius for a set of centres."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2))
    wall = HALF - np.max(np.abs(pts), axis=1)
    min_wall = np.min(wall)
    r = 0.5 * min(min_pair, 2.0 * min_wall)
    if not np.isfinite(r) or r < 0:
        return 0.0
    return float(r)


# ---------------- smooth energy with gradient ----------------
def _smooth_energy_and_grad(pts, r_target, beta=50.0):
    """Compute smooth energy and its gradient.
    
    Energy = sum over pairs: max(0, r_target - d_ij)^2  (smoothed)
           + sum over walls: max(0, r_target - wall_i)^2 (smoothed)
    
    Uses soft-max via log-sum-exp for differentiability.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]  # (n,n,3)
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(np.maximum(d2, 1e-15))
    
    # Violations: how much each pair/wall violates r_target
    pair_viol = np.maximum(0.0, r_target - d)
    wall_dist = HALF - np.max(np.abs(pts), axis=1)  # (n,)
    wall_viol = np.maximum(0.0, r_target - wall_dist)  # (n,)
    
    # Smooth energy: quadratic in violations
    E = 0.5 * np.sum(pair_viol ** 2) + 0.5 * np.sum(wall_viol ** 2)
    
    # Gradient
    # dE/dd_ij = -pair_viol_ij for d_ij < r_target
    # dE/dpts_i = sum_j (-pair_viol_ij) * (pts_i - pts_j) / d_ij
    grad = np.zeros_like(pts)
    
    mask_pair = d < r_target
    # For pairs: grad_i += pair_viol_ij * (pts_i - pts_j) / d_ij
    # Using einsum for efficiency
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_d = np.where(d > 1e-10, 1.0 / d, 0.0)
    
    # pair_viol is (n,n), inv_d is (n,n), diff is (n,n,3)
    # grad_i = sum_j pair_viol_ij * inv_d_ij * diff_ij
    pair_grad = np.einsum('ij,ij,ijk->ik', pair_viol, inv_d, diff)
    grad += pair_grad
    
    # Wall gradient: dE/dpts_ik = wall_viol_i * sign(pts_ik) for the max coordinate
    # wall_dist_i = HALF - max_k |pts_ik|
    # d(wall_dist_i)/dpts_ik = -sign(pts_ik) if k is argmax
    max_abs = np.max(np.abs(pts), axis=1, keepdims=True)  # (n,1)
    max_idx = np.argmax(np.abs(pts), axis=1)  # (n,)
    
    # Only for violated walls
    wall_mask = wall_viol > 0
    if np.any(wall_mask):
        wall_sign = np.sign(pts[wall_mask, max_idx[wall_mask]])
        grad[wall_mask, max_idx[wall_mask]] += wall_viol[wall_mask] * wall_sign
    
    return E, grad


# ---------------- SGLD dynamics ----------------
def _sgld(pts, r_target, rng, deadline, n_steps=2000, beta=50.0):
    """Run SGLD to minimize smooth energy for given target radius."""
    pts = np.clip(pts.copy(), -HALF, HALF)
    best_pts = pts.copy()
    best_E = _smooth_energy_and_grad(pts, r_target, beta)[0]
    
    lr0, lr1 = 0.01, 1e-5
    T0, T1 = 0.05, 1e-6
    
    for it in range(n_steps):
        if time.time() > deadline:
            break
        frac = it / max(1, n_steps - 1)
        lr = lr0 * (lr1 / lr0) ** frac
        T = T0 * (T1 / T0) ** frac
        
        E, g = _smooth_energy_and_grad(pts, r_target, beta)
        
        if E < best_E:
            best_E = E
            best_pts = pts.copy()
        
        noise = np.sqrt(2.0 * lr * T) * rng.standard_normal((N, 3))
        pts = np.clip(pts - lr * g + noise, -HALF, HALF)
        
        # Periodic exact eval
        if it % 300 == 0 or it == n_steps - 1:
            r = _radius(pts)
            if r > _radius(best_pts):
                best_pts = pts.copy()
    
    r = _radius(pts)
    if r > _radius(best_pts):
        best_pts = pts.copy()
    
    return best_pts


# ---------------- local polish ----------------
def _local_polish(pts, deadline, max_iter=300):
    """Local search: push apart the closest pair or wall violation."""
    pts = np.clip(pts.copy(), -HALF, HALF)
    best_pts = pts.copy()
    best_r = _radius(best_pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        
        wall = HALF - np.max(np.abs(pts), axis=1)
        
        min_pair_d2 = np.min(d2)
        min_wall = np.min(wall)
        
        r = 0.5 * min(np.sqrt(min_pair_d2), 2.0 * min_wall)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        
        if np.sqrt(min_pair_d2) < 2.0 * min_wall:
            idx = np.unravel_index(np.argmin(d2), d2.shape)
            i, j = idx[0], idx[1]
            if i == j:
                continue
            d_vec = pts[i] - pts[j]
            d_mag = np.sqrt(min_pair_d2)
            if d_mag < 1e-10:
                continue
            push = 0.003
            pts[i] += push * d_vec / d_mag
            pts[j] -= push * d_vec / d_mag
        else:
            wi = np.argmin(wall)
            coords = np.abs(pts[wi])
            k = np.argmax(coords)
            sign = np.sign(pts[wi, k])
            pts[wi, k] -= 0.003 * sign
        
        pts = np.clip(pts, -HALF, HALF)
    
    r = _radius(pts)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()
    
    return best_pts, best_r


# ---------------- initializers ----------------
def _init_bcc():
    """BCC lattice initialization."""
    m = 5
    pts = []
    s = 1.0 / m
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                cx = (ix + 0.5) * s - 0.5
                cy = (iy + 0.5) * s - 0.5
                cz = (iz + 0.5) * s - 0.5
                pts.append([cx - s/2, cy - s/2, cz - s/2])
                pts.append([cx + s/2, cy + s/2, cz + s/2])
    return np.array(pts[:N], dtype=np.float64)


def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _init_random(rng):
    return (rng.random((N, 3)) - 0.5) * 0.85


# ---------------- main construct ----------------
def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    rng = np.random.default_rng(20240517)
    
    # Start with BCC
    best_pts = _init_bcc()
    best_r = _radius(best_pts)
    
    grid = _init_grid()
    grid_r = _radius(grid)
    if grid_r > best_r:
        best_r = grid_r
        best_pts = grid.copy()
    
    # Phase 1: SGLD with smooth energy to explore
    # Try several target radii, starting from current best
    r_target = best_r * 1.05
    for trial in range(3):
        if time.time() > deadline:
            break
        start = best_pts.copy() if trial == 0 else _init_random(rng)
        p = _sgld(start, r_target, rng, deadline, n_steps=1500, beta=50.0)
        r = _radius(p)
        if r > best_r:
            best_r = r
            best_pts = p.copy()
            r_target = best_r * 1.03  # increase target
    
    # Phase 2: Local polish
    if time.time() < deadline - 5.0:
        p2, r2 = _local_polish(best_pts, deadline, max_iter=200)
        if r2 > best_r:
            best_r = r2
            best_pts = p2.copy()
    
    # Phase 3: More SGLD restarts if time permits
    guard = 0
    while time.time() < deadline - 10.0 and guard < 5:
        guard += 1
        start = _init_random(rng)
        p = _sgld(start, best_r * 1.02, rng, deadline, n_steps=1000, beta=50.0)
        r = _radius(p)
        if r > best_r:
            best_r = r
            best_pts = p.copy()
        
        if time.time() < deadline - 5.0:
            p2, r2 = _local_polish(best_pts, deadline, max_iter=100)
            if r2 > best_r:
                best_r = r2
                best_pts = p2.copy()
    
    # Final polish
    if time.time() < deadline - 2.0:
        p2, r2 = _local_polish(best_pts, deadline, max_iter=150)
        if r2 > best_r:
            best_r = r2
            best_pts = p2.copy()
    
    out = np.clip(np.asarray(best_pts, dtype=np.float64), -HALF, HALF)
    if not np.all(np.isfinite(out)):
        out = _init_grid()
    return out
