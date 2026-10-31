# EVOLVE-BLOCK-START

import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist[mask] < (min_dist[mask] - tol)):
        return False
    return True

def _repair(pos, radii):
    """
    Vectorized repair step: shrink radii to fix overlaps and boundary violations.
    This ensures the output is valid even if SLSQP fails slightly.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(100):
        valid = True
        
        # Boundary check
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-6)
                valid = False
        
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    if sum_r > 1e-10:
                        k = dist / sum_r * 0.9999
                        radii[i] = max(radii[i] * k, 1e-6)
                        radii[j] = max(radii[j] * k, 1e-6)
                        valid = False
                    else:
                        radii[i] = max(radii[i] * 0.99, 1e-6)
                        radii[j] = max(radii[j] * 0.99, 1e-6)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _init_greedy(rng):
    """
    Greedy initialization: place circles one by one at random positions
    with the maximum possible radius that fits.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    for i in range(N):
        placed = False
        for attempt in range(200):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                # r_i + r_j <= dist  => r_i <= dist - r_j
                potential_r = dist - radii[j]
                if potential_r < 0:
                    max_r = 0
                    break
                max_r = min(max_r, potential_r)
            
            if max_r > 0.005:
                pos[i] = [x, y]
                radii[i] = max_r
                placed = True
                break
        
        if not placed:
            # Fallback
            pos[i] = [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)]
            radii[i] = 0.01

    return pos, radii

def _solve_convex(pos_init, radii_init):
    """
    Solve the convex relaxation using SLSQP.
    Constraints are convex.
    """
    x0 = np.concatenate([pos_init.flatten(), radii_init])
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    def obj(x):
        return -np.sum(x[2*N:])
        
    def cons_fun(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        
        c_bound = np.concatenate([
            p[:, 0] - r,
            1.0 - p[:, 0] - r,
            p[:, 1] - r,
            1.0 - p[:, 1] - r
        ])
        
        dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
        dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        c_overlap = (dist - min_dist)[mask]
        
        return np.concatenate([c_bound, c_overlap])
        
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': cons_fun}],
                       options={'maxiter': 400, 'ftol': 1e-12})
        p = res.x[:2*N].reshape(N, 2)
        r = res.x[2*N:]
        r = np.maximum(r, 1e-6)
        
        # Repair to ensure strict validity
        p, r = _repair(p, r)
        
        if _is_valid(p, r):
            return p, r
    except Exception:
        pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_pos = None
    best_radii = None
    best_score = -1
    
    rng = np.random.RandomState(42)
    
    # Multi-start with Greedy Init
    for i in range(30):
        if time.time() > deadline - 5:
            break
            
        pos, radii = _init_greedy(rng)
        
        res = _solve_convex(pos, radii)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r
                
    if best_pos is None:
        pos, radii = _init_greedy(rng)
        pos, radii = _repair(pos, radii)
        best_pos = pos
        best_radii = radii

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
