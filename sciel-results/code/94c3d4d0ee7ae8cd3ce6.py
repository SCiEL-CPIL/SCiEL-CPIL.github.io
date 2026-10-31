import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_score = 0.0
    best_config = None
    
    # Try a few aspect ratios
    # Perimeter 4 => w + h = 2 => h = 2 - w
    # w can range from small to large. 
    # Square is w=1. Elongated might be better for packing?
    # Let's try w in [0.7, 1.3]
    aspect_ratios = np.linspace(0.7, 1.3, 7)
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Run multiple restarts for this aspect ratio
        # We have limited time, so maybe 2-3 restarts per aspect ratio
        n_restarts = 3
        time_per_restart = (deadline - time.time()) / (n_restarts * len(aspect_ratios))
        
        for _ in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize positions randomly
            # Radii start small
            r = np.full(N, 0.05)
            # Positions
            # x in [r, w-r], y in [r, h-r]
            # Since r is small, just [0.05, w-0.05]
            x = np.random.uniform(0.05, w - 0.05, N)
            y = np.random.uniform(0.05, h - 0.05, N)
            
            current_r_sum = np.sum(r)
            
            # Inflation loop
            # Increase radii and optimize positions
            max_iter_inflation = 200
            delta_r = 0.01
            
            for k in range(max_iter_inflation):
                if time.time() > deadline:
                    break
                
                # Increase radii
                r += delta_r
                
                # Define objective for position optimization
                # Minimize overlap penalty
                def overlap_penalty(pos):
                    # pos is array of shape (2*N,)
                    # x = pos[0:N], y = pos[N:2*N]
                    px = pos[:N]
                    py = pos[N:]
                    
                    # Boundaries penalty
                    # x < r or x > w-r
                    # y < r or y > h-r
                    # Use quadratic penalty
                    bound_pen = 0.0
                    # Left
                    bound_pen += np.sum(np.maximum(0, r - px)**2)
                    # Right
                    bound_pen += np.sum(np.maximum(0, px + r - w)**2)
                    # Bottom
                    bound_pen += np.sum(np.maximum(0, r - py)**2)
                    # Top
                    bound_pen += np.sum(np.maximum(0, py + r - h)**2)
                    
                    # Overlap penalty
                    # Calculate all pair distances
                    # N is small (21), so N^2 is fine
                    dx = np.subtract.outer(px, px)
                    dy = np.subtract.outer(py, py)
                    d = np.sqrt(dx**2 + dy**2)
                    
                    # Sum radii
                    sum_r = np.add.outer(r, r)
                    
                    # Overlap amount
                    overlap = np.maximum(0, sum_r - d)
                    
                    # Quadratic penalty
                    # Mask diagonal to avoid self-overlap (though d=0, sum_r>0)
                    np.fill_diagonal(overlap, 0)
                    
                    overlap_pen = np.sum(overlap**2)
                    
                    return bound_pen + 10.0 * overlap_pen
                
                # Initial guess for optimizer is current x, y
                pos0 = np.concatenate([x, y])
                
                # Bounds for x, y
                # x_i must be in [r_i, w - r_i]
                # y_i must be in [r_i, h - r_i]
                # Since r changes, bounds change.
                # L-BFGS-B supports bounds.
                bounds = []
                for i in range(N):
                    bounds.append((r[i], w - r[i]))
                    bounds.append((r[i], h - r[i]))
                
                # Optimize
                # Use L-BFGS-B
                # maxiter limited to keep speed up
                res = minimize(
                    overlap_penalty, 
                    pos0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 20, 'ftol': 1e-6}
                )
                
                if res.success:
                    x, y = res.x[:N], res.x[N:]
                    # Check if valid
                    # If penalty is low, we are good
                    if res.fun < 1e-4:
                        # Valid packing found for these radii
                        current_r_sum = np.sum(r)
                        if current_r_sum > best_score:
                            best_score = current_r_sum
                            best_config = np.column_stack([x, y, r])
                    else:
                        # Stuck, randomize positions
                        x = np.random.uniform(r, w - r, N)
                        y = np.random.uniform(r, h - r, N)
                else:
                    # Optimization failed, randomize
                    x = np.random.uniform(r, w - r, N)
                    y = np.random.uniform(r, h - r, N)
            
            # End inflation loop
        # End restarts
    # End aspect ratios
    
    # If no valid config found (should not happen), return seed-like
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_config = np.array(cells[:N])
        
    return best_config

# EVOLVE-BLOCK-END
