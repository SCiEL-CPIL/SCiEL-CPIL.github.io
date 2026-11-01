import numpy as np
from scipy.optimize import minimize
import time

N = 32
START_TIME = time.time()
TIME_LIMIT = 110.0

def construct():
    global START_TIME
    start_time = time.time()
    
    # Initial grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side
                cells.append([x, y, r])
    init_grid = np.array(cells)
    
    def objective(params, mu):
        if time.time() - start_time > TIME_LIMIT:
            return 1e10
            
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        obj = -np.sum(r)
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        
        min_dist = r[:, None] + r[None, :]
        diff = min_dist - dist
        smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-6))
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlap_pen = np.sum(smooth_max[mask])
        
        diff_x_neg = r - x
        diff_x_pos = x - (1.0 - r)
        pen_x_neg = 0.5 * (diff_x_neg + np.sqrt(diff_x_neg*diff_x_neg + 1e-6))
        pen_x_pos = 0.5 * (diff_x_pos + np.sqrt(diff_x_pos*diff_x_pos + 1e-6))
        
        diff_y_neg = r - y
        diff_y_pos = y - (1.0 - r)
        pen_y_neg = 0.5 * (diff_y_neg + np.sqrt(diff_y_neg*diff_y_neg + 1e-6))
        pen_y_pos = 0.5 * (diff_y_pos + np.sqrt(diff_y_pos*diff_y_pos + 1e-6))
        
        bound_pen = np.sum(pen_x_neg) + np.sum(pen_x_pos) + np.sum(pen_y_neg) + np.sum(pen_y_pos)
        
        total_obj = obj + mu * (overlap_pen + bound_pen)
        
        return total_obj

    def make_valid(params):
        x = params[:N].copy()
        y = params[N:2*N].copy()
        r = params[2*N:3*N].copy()
        
        # Iteratively shrink radii to ensure validity
        for _ in range(100):
            valid = True
            for i in range(N):
                # Boundary check
                min_bound = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                if r[i] > min_bound:
                    r[i] = min_bound * 0.99
                    valid = False
            
            if valid:
                # Overlap check
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist_sq = dx*dx + dy*dy
                        min_dist = r[i] + r[j]
                        if dist_sq < min_dist*min_dist:
                            # Shrink both
                            factor = np.sqrt(dist_sq) / min_dist * 0.99
                            r[i] *= factor
                            r[j] *= factor
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                break
                
        return np.concatenate([x, y, r])

    best_params = init_grid.flatten()
    best_score = np.sum(init_grid[:, 2])
    
    np.random.seed(42)
    
    # Adaptive penalty strategy
    # Start with lower penalty, increase over time
    base_mu = 10.0
    max_mu = 1000.0
    
    num_restarts = 100
    
    for k in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Determine penalty for this restart
        # Early restarts use lower penalty to explore
        # Later restarts use higher penalty to refine
        if k < 20:
            mu = base_mu
        elif k < 50:
            mu = 100.0
        else:
            mu = max_mu
            
        if k == 0:
            x0 = init_grid.flatten()
        else:
            # Generate diverse initializations
            # Mix of grid-based and random
            if np.random.random() < 0.5:
                # Jittered grid
                x0 = init_grid.flatten().copy()
                noise_scale = 0.1
                x0[:N] += np.random.uniform(-noise_scale, noise_scale, N)
                x0[N:2*N] += np.random.uniform(-noise_scale, noise_scale, N)
                x0[2*N:3*N] = np.maximum(0.01, x0[2*N:3*N] + np.random.uniform(-0.05, 0.05, N))
            else:
                # Random initialization
                x0 = np.zeros(3*N)
                x0[:N] = np.random.uniform(0.1, 0.9, N)
                x0[N:2*N] = np.random.uniform(0.1, 0.9, N)
                x0[2*N:3*N] = np.random.uniform(0.02, 0.1, N)
            
            x0[:N] = np.clip(x0[:N], 0, 1)
            x0[N:2*N] = np.clip(x0[N:2*N], 0, 1)

        try:
            # Use L-BFGS-B for speed
            res = minimize(objective, x0, args=(mu,), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            
            valid_params = make_valid(res.x)
            score = np.sum(valid_params[2*N:3*N])
            
            if score > best_score:
                best_score = score
                best_params = valid_params
        except Exception as e:
            pass

    # Final polish with very high penalty and tight tolerance
    if time.time() - start_time < TIME_LIMIT - 10:
        try:
            # Use Powell or Nelder-Mead for final polish? L-BFGS-B might be stuck.
            # Let's try L-BFGS-B again with high mu and tight ftol
            res = minimize(objective, best_params, args=(1000.0,), method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})
            valid_params = make_valid(res.x)
            score = np.sum(valid_params[2*N:3*N])
            if score > best_score:
                best_score = score
                best_params = valid_params
        except Exception as e:
            pass
        
        # Another polish with different method?
        if time.time() - start_time < TIME_LIMIT - 5:
            try:
                res = minimize(objective, best_params, args=(1000.0,), method='Powell', options={'maxiter': 500, 'ftol': 1e-12})
                valid_params = make_valid(res.x)
                score = np.sum(valid_params[2*N:3*N])
                if score > best_score:
                    best_score = score
                    best_params = valid_params
            except Exception as e:
                pass

    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    result = np.column_stack((x, y, r))
    return result
