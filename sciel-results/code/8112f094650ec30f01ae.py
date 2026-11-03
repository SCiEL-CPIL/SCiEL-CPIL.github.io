import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _compute_sum_radii(pos, w, h):
    """Compute sum of radii for given positions in rectangle w x h."""
    n = len(pos)
    radii = np.zeros(n)
    for i in range(n):
        # Distance to walls
        min_dist = min(pos[i, 0], w - pos[i,0], pos[i, 1], h - pos[i,1])
        # Distance to other circles
        for j in range(n):
            if i != j:
                d = np.sqrt((pos[i,0] - pos[j,0])**2 + (pos[i,1] - pos[j,1])**2)
                min_dist = min(min_dist, d / 2.0)
        radii[i] = max(0, min_dist)
    return radii

def _objective_smooth(x, w, h, n=N, beta=50.0):
    """
    Smooth objective: minimize -sum of radii.
    Uses log-sum-exp style soft-min for the radius computation.
    r_i = soft_min(distances_to_boundaries_and_other_circles)
    """
    pos = x[:n*2].reshape(n, 2)
    
    # Check bounds - if out of bounds, return large penalty
    if np.any(pos[:, 0] < -0.01) or np.any(pos[:, 0] > w + 0.01) or \
       np.any(pos[:, 1] < -0.01) or np.any(pos[:, 1] > h + 0.01):
        return 1e6
    
    sum_radii = 0.0
    for i in range(n):
        # Collect all distances that bound the radius of circle i
        dists = []
        # Wall distances
        dists.append(pos[i, 0])
        dists.append(w - pos[i, 0])
        dists.append(pos[i, 1])
        dists.append(h - pos[i, 1])
        # Distances to other circles (half the center-to-center distance)
        for j in range(n):
            if i != j:
                d = np.sqrt((pos[i,0] - pos[j,0])**2 + (pos[i,1] - pos[j,1])**2)
                dists.append(d / 2.0)
        
        dists = np.array(dists)
        
        # Soft-min using log-sum-exp
        # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
        # For numerical stability: soft_min(x) = x_min - 1/beta * log(sum(exp(-beta*(x - x_min))))
        x_min = np.min(dists)
        if x_min < -1e-6:
            # Circle is overlapping, penalize
            sum_radii += 0.0
        else:
            shifted = -beta * (dists - x_min)
            # Clip to avoid overflow
            shifted = np.clip(shifted, -700, 700)
            log_sum = np.log(np.sum(np.exp(shifted)))
            soft_min_val = x_min - log_sum / beta
            sum_radii += max(0, soft_min_val)
    
    return -sum_radii

def _objective_exact(x, w, h, n=N):
    """Exact objective for final evaluation."""
    pos = x[:n*2].reshape(n, 2)
    if np.any(pos[:, 0] < -1e-9) or np.any(pos[:, 0] > w + 1e-9) or \
       np.any(pos[:, 1] < -1e-9) or np.any(pos[:, 1] > h + 1e-9):
        return 1e6
    radii = _compute_sum_radii(pos, w, h)
    return -np.sum(radii)

def _pack_for_aspect(w, h, n_circles=N, deadline=None):
    """Pack circles in w x h rectangle, return (positions, radii) maximizing sum of radii."""
    
    # Grid initialization
    cols = int(np.ceil(np.sqrt(n_circles * w / h)))
    rows = int(np.ceil(n_circles / cols))
    
    positions = np.zeros((n_circles, 2))
    for i in range(n_circles):
        col = i % cols
        row = i // cols
        positions[i, 0] = (col + 0.5) * w / cols
        positions[i, 1] = (row + 0.5) * h / rows
    
    x0 = positions.flatten()
    
    # Bounds for positions
    bounds = []
    for i in range(n_circles):
        bounds.append((0, w))
        bounds.append((0, h))
    
    best_x = x0.copy()
    best_obj = _objective_smooth(x0, w, h, n_circles, beta=50.0)
    
    # Phase 1: Coarse optimization with smooth objective
    for beta in [30.0, 50.0, 100.0, 200.0]:
        if deadline is not None and time.time() > deadline:
            break
        
        result = minimize(
            _objective_smooth, x0, args=(w, h, n_circles, beta),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x.copy()
            x0 = result.x.copy()
    
    # Phase 2: Polish with exact objective using Nelder-Mead
    if deadline is not None and time.time() < deadline:
        result = minimize(
            _objective_exact, best_x, args=(w, h, n_circles),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x.copy()
    
    pos = best_x[:n_circles*2].reshape(n_circles, 2)
    radii = _compute_sum_radii(pos, w, h)
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_score = 0
    best_result = None
    
    # Try various aspect ratios - focus on promising ranges
    # Perimeter = 2(w+h) = 4, so w+h = 2
    # Aspect ratio r = w/h, so w = 2r/(1+r), h = 2/(1+r)
    aspect_ratios = np.array([
        0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
        1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0
    ])
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii = _pack_for_aspect(w, h, deadline=deadline)
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, radii])
    
    # Fallback: simple grid in square
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells[:N])
    
    return best_result
