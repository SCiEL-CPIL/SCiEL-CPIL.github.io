import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = -np.inf
    best_solution = None
    
    # We search over aspect ratio W. H = 2 - W.
    # W can range from 0.5 (square 1x1) to 1.5 (1.5x0.5).
    # Symmetry means W and H are interchangeable, so we can just search W in [0.5, 1.5].
    w_values = np.linspace(0.5, 1.5, 11)
    
    # Penalty weight
    lambda_pen = 100.0
    
    def objective(z, W):
        H = 2.0 - W
        # z is array of size 3*N
        z = z.reshape(N, 3)
        x = z[:, 0]
        y = z[:, 1]
        r = z[:, 2]
        
        # Wall constraints
        # r <= x, r <= W-x, r <= y, r <= H-y
        wall_viol = np.maximum(0, r - x) + np.maximum(0, r - (W - x)) + np.maximum(0, r - y) + np.maximum(0, r - (H - y))
        wall_pen = np.sum(wall_viol**2)
        
        # Overlap constraints
        # d_ij >= r_i + r_j
        # Vectorized calculation of pairwise distances
        # diff = x_i - x_j
        x_diff = x[:, np.newaxis] - x[np.newaxis, :]
        y_diff = y[:, np.newaxis] - y[np.newaxis, :]
        d = np.sqrt(x_diff**2 + y_diff**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Only consider i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d_upper = d[mask]
        r_sum_upper = r_sum[mask]
        
        overlap_viol = np.maximum(0, r_sum_upper - d_upper)
        overlap_pen = np.sum(overlap_viol**2)
        
        total_pen = wall_pen + overlap_pen
        
        sum_r = np.sum(r)
        
        return -sum_r + lambda_pen * total_pen
    
    for W in w_values:
        if time.time() > deadline:
            break
            
        H = 2.0 - W
        # Bounds: [xmin, xmax, ymin, ymax, rmin, rmax] for each circle
        # x in [0, W], y in [0, H], r in [0, min(W,H)/2]
        r_max = min(W, H) / 2.0
        
        bounds = []
        for _ in range(N):
            bounds.append((0, W))
            bounds.append((0, H))
            bounds.append((0, r_max))
            
        # Run Differential Evolution
        # maxiter and popsize tuned for speed and quality
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                args=(W,),
                maxiter=50, 
                popsize=10, 
                tol=0.01, 
                seed=42,
                polish=False # We will polish manually if needed, or just use DE result
            )
        except Exception:
            continue
            
        if res.success or res.fun < best_score:
            # Check if the solution is valid (penalty is small)
            z_best = res.x
            val = objective(z_best, W)
            # Extract penalty to check validity
            z_check = z_best.reshape(N, 3)
            x = z_check[:, 0]
            y = z_check[:, 1]
            r = z_check[:, 2]
            
            wall_viol = np.maximum(0, r - x) + np.maximum(0, r - (W - x)) + np.maximum(0, r - y) + np.maximum(0, r - (H - y))
            wall_pen = np.sum(wall_viol**2)
            
            x_diff = x[:, np.newaxis] - x[np.newaxis, :]
            y_diff = y[:, np.newaxis] - y[np.newaxis, :]
            d = np.sqrt(x_diff**2 + y_diff**2)
            r_sum = r[:, np.newaxis] + r[np.newaxis, :]
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            d_upper = d[mask]
            r_sum_upper = r_sum[mask]
            overlap_viol = np.maximum(0, r_sum_upper - d_upper)
            overlap_pen = np.sum(overlap_viol**2)
            
            total_pen = wall_pen + overlap_pen
            
            if total_pen < 1e-6:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_solution = z_best.copy()
            else:
                # If penalty is not zero, we might still keep it if it's the best we have,
                # but we should try to polish it.
                # For now, only accept valid solutions.
                # We can also try a local polish on the DE result.
                pass

    if best_solution is None:
        # Fallback to seed program logic if optimization fails
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_solution
