"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 150.0
_TIME_MARGIN = 5.0  # seconds before deadline to stop


def _remaining_time():
    return _DEADLINE - time.time()


def _check_deadline():
    return time.time() > _DEADLINE - _TIME_MARGIN


def _evaluate(config, W):
    """
    config: (N, 3) array with [cx, cy, r]
    W: width of rectangle (height = 2 - W)
    Returns (score, valid) where score = sum of radii if valid, else 0.
    """
    cx, cy, r = config[:, 0], config[:, 1], config[:, 2]
    H = 2.0 - W
    
    # Check bounds
    if np.any(cx < r) or np.any(cy < r) or np.any(cx + r > W) or np.any(cy + r > H):
        return 0.0, False
    
    # Check pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - 1e-9:
                return 0.0, False
    
    return np.sum(r), True


def _optimize_positions(config, W, maxiter=2000):
    """
    Given initial config and W, optimize positions and radii to maximize sum(r)
    subject to non-overlap and boundary constraints.
    Uses log-sum-exp soft penalty for collisions.
    """
    H = 2.0 - W
    
    def objective(x):
        cx, cy, r = x[0:N], x[N:2*N], x[2*N:3*N]
        
        # Boundary violations (soft)
        bound_pen = 0.0
        for i in range(N):
            bound_pen += max(0, r[i] - cx[i]) ** 2
            bound_pen += max(0, r[i] - (W - cx[i])) ** 2
            bound_pen += max(0, r[i] - cy[i]) ** 2
            bound_pen += max(0, r[i] - (H - cy[i])) ** 2
        
        # Pairwise overlap (log-sum-exp for smoothness)
        overlap_pen = 0.0
        beta = 50.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx * dx + dy * dy + 1e-12)
                min_dist = r[i] + r[j]
                overlap = max(0.0, min_dist - dist)
                overlap_pen += overlap ** 2
        
        # Maximize sum(r) => minimize -sum(r) + penalties
        return -np.sum(r) + 10.0 * bound_pen + 10.0 * overlap_pen
    
    def jac(x):
        cx, cy, r = x[0:N], x[N:2*N], x[2*N:3*N]
        g = np.zeros(3 * N)
        
        # Gradient of -sum(r)
        for i in range(N):
            g[2*N + i] = -1.0
        
        # Boundary gradients
        for i in range(N):
            if cx[i] < r[i]:
                g[i] += 20.0 * (r[i] - cx[i])
                g[2*N + i] += -20.0 * (r[i] - cx[i])
            if W - cx[i] < r[i]:
                g[i] += -20.0 * (r[i] - (W - cx[i]))
                g[2*N + i] += 20.0 * (r[i] - (W - cx[i]))
            if cy[i] < r[i]:
                g[N + i] += 20.0 * (r[i] - cy[i])
                g[2*N + i] += -20.0 * (r[i] - cy[i])
            if H - cy[i] < r[i]:
                g[N + i] += -20.0 * (r[i] - (H - cy[i]))
                g[2*N + i] += 20.0 * (r[i] - (H - cy[i]))
        
        # Pairwise overlap gradients
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx * dx + dy * dy + 1e-12)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    fx = dx / dist
                    fy = dy / dist
                    g[i] += 20.0 * overlap * fx
                    g[N + i] += 20.0 * overlap * fy
                    g[j] += -20.0 * overlap * fx
                    g[N + j] += -20.0 * overlap * fy
                    g[2*N + i] += 20.0 * overlap
                    g[2*N + j] += 20.0 * overlap
        
        return g
    
    x0 = config.flatten()
    
    # Bounds: positions in [0, W] or [0, H], radii positive
    bounds = []
    for i in range(N):
        bounds.append((0.0, W))
    for i in range(N):
        bounds.append((0.0, H))
    for i in range(N):
        bounds.append((1e-6, min(W, H) / 2.0))
    
    result = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-10})
    
    return result.x.reshape(N, 3)


def _try_configuration(W, init_config, maxiter=2000):
    """Try to optimize from initial config for given W."""
    if _check_deadline():
        return None, 0.0
    
    H = 2.0 - W
    
    # Clamp initial positions to be inside
    config = init_config.copy()
    config[:, 0] = np.clip(config[:, 0], config[:, 2], W - config[:, 2])
    config[:, 1] = np.clip(config[:, 1], config[:, 2], H - config[:, 2])
    
    # Optimize
    optimized = _optimize_positions(config, W, maxiter=maxiter)
    
    # Clamp result
    optimized[:, 0] = np.clip(optimized[:, 0], 0, W)
    optimized[:, 1] = np.clip(optimized[:, 1], 0, H)
    optimized[:, 2] = np.maximum(optimized[:, 2], 1e-6)
    
    score, valid = _evaluate(optimized, W)
    if valid:
        return optimized, score
    return None, 0.0


def construct():
    start_time = time.time()
    
    best_config = None
    best_score = 0.0
    
    # Strategy: try different aspect ratios W (width), H = 2 - W
    # Perimeter = 2(W + H) = 2(W + 2 - W) = 4. Good.
    # W ranges from small to large. Optimal likely near square or moderate aspect.
    
    # Grid of aspect ratios to try
    W_values = np.linspace(0.5, 1.5, 11)  # W from 0.5 to 1.5, H from 1.5 to 0.5
    
    for W in W_values:
        if _check_deadline():
            break
        
        H = 2.0 - W
        
        # Initialize with grid-like placement
        # Place circles in a grid pattern scaled to fit
        cols = int(np.ceil(np.sqrt(N * W / H)))
        rows = int(np.ceil(N / cols))
        
        if rows == 0:
            rows = 1
        if cols == 0:
            cols = 1
        
        # Cell sizes
        cell_w = W / cols
        cell_h = H / rows
        
        # Initial radius estimate
        init_r = min(cell_w, cell_h) * 0.45
        
        init_config = np.zeros((N, 3))
        for i in range(N):
            row = i // cols
            col = i % cols
            init_config[i, 0] = (col + 0.5) * cell_w
            init_config[i, 1] = (row + 0.5) * cell_h
            init_config[i, 2] = init_r
        
        config, score = _try_configuration(W, init_config, maxiter=1500)
        
        if config is not None and score > best_score:
            best_score = score
            best_config = config.copy()
    
    # Second pass: try more refined W values around the best
    if best_config is not None and not _check_deadline():
        # Try to improve by perturbing W slightly
        for W_offset in [-0.05, -0.025, 0.0, 0.025, 0.05]:
            if _check_deadline():
                break
            W_new = best_config[:, 0].max() + 0.01  # rough guess
            # Actually, let's try W values around 1.0
            W_try = 1.0 + W_offset
            if W_try <= 0 or W_try >= 2:
                continue
            H_try = 2.0 - W_try
            
            # Use best config as init, rescale
            init_config = best_config.copy()
            init_config[:, 0] *= W_try / 1.0
            init_config[:, 1] *= H_try / 1.0
            # Re-estimate radii
            init_config[:, 2] = np.minimum(init_config[:, 2], 
                                           np.minimum(np.minimum(init_config[:, 0], 
                                                                  W_try - init_config[:, 0]),
                                                      np.minimum(init_config[:, 1], 
                                                                  H_try - init_config[:, 1])))
            init_config[:, 2] = np.maximum(init_config[:, 2], 1e-4)
            
            config, score = _try_configuration(W_try, init_config, maxiter=1000)
            
            if config is not None and score > best_score:
                best_score = score
                best_config = config.copy()
    
    # If we still have time, do a final polish on the best
    if best_config is not None and not _check_deadline():
        # Determine W from best config
        max_x = best_config[:, 0].max()
        max_y = best_config[:, 1].max()
        min_x = best_config[:, 0].min()
        min_y = best_config[:, 1].max()  # typo, should be min
        min_y = best_config[:, 1].min()
        
        W_est = max_x - min_x + best_config[:, 2].max() * 2
        # Actually, just use the W that produced it. Let's re-derive:
        # We need W such that all circles fit. Use max extent.
        # For now, just do a local polish assuming W from the best trial.
        # This is complex; skip for now if no clear W.
        
        pass
    
    if best_config is None:
        # Fallback: simple grid in a 1x2 rectangle
        W = 1.0
        H = 2.0
        cols = 3
        rows = 7
        cell_w = W / cols
        cell_h = H / rows
        r = min(cell_w, cell_h) * 0.45
        config = np.zeros((N, 3))
        for i in range(N):
            row = i // cols
            col = i % cols
            config[i, 0] = (col + 0.5) * cell_w
            config[i, 1] = (row + 0.5) * cell_h
            config[i, 2] = r
        return config
    
    return best_config
# EVOLVE-BLOCK-END
