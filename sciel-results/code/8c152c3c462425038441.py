import numpy as np
from scipy.optimize import minimize
import time

N = 21

def soft_min(x, beta=20.0):
    """
    Smooth approximation of the minimum of x.
    x: array of values
    beta: temperature parameter (higher is sharper)
    """
    # Shift to avoid overflow in exp
    x_min = np.min(x)
    # If x_min is very large positive, exp(-beta*(x-x_min)) is safe
    # If x_min is very large negative, exp(-beta*(x-x_min)) might overflow if x is large positive?
    # Actually, x - x_min >= 0, so exp(-beta * (x - x_min)) <= 1. No overflow.
    # Underflow is possible for large differences, which is fine (contributes 0 to sum).
    
    # Handle case where x has very large values to maintain numerical stability
    # x_shifted = x - x_min
    # log_sum = np.log(np.sum(np.exp(-beta * x_shifted)))
    # return x_min - (1.0 / beta) * log_sum
    
    # Simpler stable version:
    x_shifted = x - x_min
    # exp(-beta * x_shifted) is in (0, 1]
    terms = np.exp(-beta * x_shifted)
    log_sum = np.log(np.sum(terms))
    return x_min - (1.0 / beta) * log_sum

def _pack_for_aspect(w, h, n_circles=N, beta=20.0, max_iter=500):
    """
    Pack circles in w x h rectangle using soft-min objective and L-BFGS-B.
    Returns (positions, radii, score).
    """
    # Initialize in a grid
    # Adjust grid aspect ratio to match rectangle
    aspect = w / h
    cols = int(np.ceil(np.sqrt(n_circles * aspect)))
    rows = int(np.ceil(n_circles / cols))
    
    positions = np.zeros((n_circles, 2))
    for i in range(n_circles):
        col = i % cols
        row = i // cols
        positions[i, 0] = (col + 0.5) * w / cols
        positions[i, 1] = (row + 0.5) * h / rows
        
    # Flatten for optimizer
    x0 = positions.flatten()
    bounds = [(0, w)] * n_circles + [(0, h)] * n_circles
    
    def objective(x):
        pos = x[:n_circles * 2].reshape(n_circles, 2)
        
        # Check if out of bounds (L-BFGS-B handles bounds, but just in case)
        if np.any(pos[:, 0] < 0) or np.any(pos[:, 0] > w) or \
           np.any(pos[:, 1] < 0) or np.any(pos[:, 1] > h):
            return 1e6 # Large penalty
            
        total_soft_r = 0.0
        for i in range(n_circles):
            dists = []
            # Wall distances
            dists.append(pos[i, 0])
            dists.append(w - pos[i, 0])
            dists.append(pos[i, 1])
            dists.append(h - pos[i, 1])
            
            # Circle distances (half distance to center)
            for j in range(n_circles):
                if i != j:
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx*dx + dy*dy) / 2.0
                    dists.append(d)
            
            dists_arr = np.array(dists)
            r_soft = soft_min(dists_arr, beta)
            total_soft_r += r_soft
            
        return -total_soft_r
        
    # Run optimization
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
    except Exception:
        # Fallback if optimization fails
        return positions, np.zeros(n_circles), 0.0
        
    pos_opt = result.x[:n_circles * 2].reshape(n_circles, 2)
    
    # Calculate true radii and score
    radii = np.zeros(n_circles)
    for i in range(n_circles):
        min_dist = min(
            pos_opt[i, 0],
            w - pos_opt[i, 0],
            pos_opt[i, 1],
            h - pos_opt[i, 1]
        )
        for j in range(n_circles):
            if i != j:
                dx = pos_opt[i, 0] - pos_opt[j, 0]
                dy = pos_opt[i, 1] - pos_opt[j, 1]
                d = np.sqrt(dx*dx + dy*dy) / 2.0
                min_dist = min(min_dist, d)
        radii[i] = max(0, min_dist - 1e-9)
        
    score = np.sum(radii)
    return pos_opt, radii, score

def construct():
    start_time = time.time()
    deadline = start_time + 100 # Safety margin for 120s limit
    
    best_score = 0
    best_result = None
    
    # Perimeter 4 => w + h = 2.
    # w ranges from small to 2.
    # Symmetry: w and 2-w are equivalent (just rotate/flip).
    # So we only need to check w in (0, 1].
    # However, the "shape" of the packing might differ.
    # Let's sample w in [0.2, 1.0].
    
    # We can use a finer grid or adaptive sampling.
    # Given L-BFGS-B is fast, we can try ~30 points.
    w_values = np.linspace(0.2, 1.0, 30)
    
    for w in w_values:
        if time.time() > deadline:
            break
            
        h = 2.0 - w
        pos, radii, score = _pack_for_aspect(w, h)
        
        if score > best_score:
            best_score = score
            # Store result as (x, y, r)
            best_result = np.column_stack([pos, radii])
            
    # Fallback to seed if nothing works
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
