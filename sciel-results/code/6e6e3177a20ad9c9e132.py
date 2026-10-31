"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, QhullError

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _compute_voronoi_radii(pos):
    """
    Compute the maximum possible radius for each circle based on Voronoi diagram.
    The radius is limited by the distance to the nearest other center and the boundary.
    """
    radii = np.zeros(N)
    try:
        vor = Voronoi(pos)
    except QhullError:
        # Fallback to simple distance computation if Voronoi fails
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(N):
                if i != j:
                    d = np.linalg.norm(pos[i] - pos[j])
                    max_r = min(max_r, d / 2.0)
            radii[i] = max_r
        return radii

    for i in range(N):
        region = vor.regions[vor.point_region[i]]
        if -1 in region:
            # Unbounded region, limited by boundary
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        else:
            # Find minimum distance to any vertex in the region
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for v in region:
                vertex = vor.vertices[v]
                d = np.linalg.norm(vertex - pos[i])
                max_r = min(max_r, d)
        radii[i] = max_r
    return radii

def _voronoi_refine(pos, n_iters=100, shrink=0.98):
    """
    Refine positions and radii using Voronoi-based constraints.
    """
    pos = pos.copy()
    for _ in range(n_iters):
        radii = _compute_voronoi_radii(pos)
        radii *= shrink
        
        # Check validity
        if _is_valid(pos, radii, tol=1e-8):
            return pos, radii
        else:
            # If invalid, shrink more aggressively
            radii *= 0.95
            if _is_valid(pos, radii, tol=1e-8):
                return pos, radii
    return pos, radii

def _objective(x, penalty_weight=1000.0):
    """
    Objective: minimize -sum(radii) + penalty * violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _initial_grid():
    """Grid-based initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 5x5 grid
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N - 1:
                pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                radii[k] = 0.095
                k += 1
    pos[k] = [0.5, 0.5]
    radii[k] = 0.05
    return pos, radii

def _initial_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, (N, 2))
    radii = rng.uniform(0.05, 0.1, N)
    
    # Simple relaxation to fix overlaps
    for _ in range(50):
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Push apart
                    if dist > 1e-8:
                        push = (min_dist - dist) / (2 * dist)
                        pos[i, 0] += dx * push
                        pos[i, 1] += dy * push
                        pos[j, 0] -= dx * push
                        pos[j, 1] -= dy * push
                    # Clamp to bounds
                    pos[i, 0] = np.clip(pos[i, 0], radii[i], 1 - radii[i])
                    pos[i, 1] = np.clip(pos[i, 1], radii[i], 1 - radii[i])
                    pos[j, 0] = np.clip(pos[j, 0], radii[j], 1 - radii[j])
                    pos[j, 1] = np.clip(pos[j, 1], radii[j], 1 - radii[j])
    
    return pos, radii

def _voronoi_optimize(pos, radii, deadline):
    """
    Optimize using Voronoi-based radius computation and gradient descent.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Compute initial Voronoi radii
    radii = _compute_voronoi_radii(pos) * 0.99
    
    # Gradient descent on positions to maximize sum of Voronoi radii
    lr = 0.01
    n_steps = 2000
    
    for step in range(n_steps):
        if time.time() > deadline - 5:
            break
        
        # Compute gradient of sum of radii w.r.t. positions
        # Approximate by finite differences
        grad = np.zeros((N, 2))
        eps = 1e-4
        current_sum = np.sum(_compute_voronoi_radii(pos))
        
        for i in range(N):
            for d in range(2):
                pos_old = pos[i, d].copy()
                pos[i, d] += eps
                r_plus = np.sum(_compute_voronoi_radii(pos))
                pos[i, d] -= 2 * eps
                r_minus = np.sum(_compute_voronoi_radii(pos))
                pos[i, d] = pos_old
                grad[i, d] = (r_plus - r_minus) / (2 * eps)
        
        # Update positions
        pos += lr * grad
        pos = np.clip(pos, 0.05, 0.95)
        
        # Re-compute radii
        radii = _compute_voronoi_radii(pos) * 0.99
    
    return pos, radii

def _polish(pos, radii, deadline):
    """High-precision polishing with L-BFGS-B."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    try:
        result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                          args=(5000.0,),
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.fun < _objective(x, 5000.0) - 1e-12:
            x = result.x
    except Exception:
        pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-4)
    
    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    # Generate multiple initializations
    inits = []
    pos_grid, rad_grid = _initial_grid()
    inits.append((pos_grid, rad_grid))
    
    for seed in [42, 123, 456]:
        pos_r, rad_r = _initial_random(seed)
        inits.append((pos_r, rad_r))
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        # Voronoi refinement
        pos_v, rad_v = _voronoi_refine(pos_init, n_iters=50, shrink=0.98)
        score = np.sum(rad_v)
        if _is_valid(pos_v, rad_v) and score > best_score:
            best_score = score
            best_pos = pos_v.copy()
            best_radii = rad_v.copy()
        
        # Voronoi optimization
        if time.time() < deadline - 20:
            try:
                pos_opt, rad_opt = _voronoi_optimize(pos_v, rad_v, deadline)
                pos_pp, rad_pp = _voronoi_refine(pos_opt, n_iters=20, shrink=0.99)
                score = np.sum(rad_pp)
                if _is_valid(pos_pp, rad_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = rad_pp.copy()
            except Exception:
                pass
        
        # Polish with L-BFGS-B
        if time.time() < deadline - 10:
            try:
                pos_pol, rad_pol = _polish(pos_v, rad_v, deadline)
                pos_pp, rad_pp = _voronoi_refine(pos_pol, n_iters=20, shrink=0.99)
                score = np.sum(rad_pp)
                if _is_valid(pos_pp, rad_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = rad_pp.copy()
            except Exception:
                pass
    
    # Final validation
    if best_pos is None or not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.5]
        radii[k] = 0.05
        best_pos, best_radii = pos, radii
    
    # Final Voronoi refinement
    best_pos, best_radii = _voronoi_refine(best_pos, n_iters=50, shrink=0.99)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
