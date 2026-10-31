import numpy as np
import random
from scipy.optimize import minimize

N = 21

def construct():
    # Objective: Maximize sum of radii.
    # Variables: u_0..u_20, v_0..v_20, W
    # x_i = u_i * W, y_i = v_i * H, H = 2 - W
    # Bounds: u_i, v_i in [0, 1], W in [0.5, 1.5]
    
    def objective_normalized(vars):
        u = vars[:N]
        v = vars[N:2*N]
        W = vars[2*N]
        H = 2.0 - W
        
        if W < 0.2 or H < 0.2:
            return 100.0

        x = u * W
        y = v * H
        
        # Vectorized distance calculation
        d_wall_x = np.minimum(x, W - x)
        d_wall_y = np.minimum(y, H - y)
        r = np.minimum(d_wall_x, d_wall_y)
        
        # Pairwise distances
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Exclude self
        np.fill_diagonal(dist, np.inf)
        
        # Min distance to any neighbor
        min_dist = np.min(dist, axis=1)
        
        # Update radii
        r = np.minimum(r, min_dist / 2.0)
        
        return -np.sum(r)

    bounds = [(0, 1) for _ in range(2*N)] + [(0.5, 1.5)]
    
    best_score = -100.0
    best_result = None
    
    # Multi-start
    starts = []
    
    # Grid starts
    for cols in [4, 5]:
        rows = (N + cols - 1) // cols
        u_grid = []
        v_grid = []
        for i in range(cols):
            for j in range(rows):
                if len(u_grid) >= N: break
                u_grid.append((i + 0.5) / cols)
                v_grid.append((j + 0.5) / rows)
        while len(u_grid) < N:
            u_grid.append(0.5)
            v_grid.append(0.5)
            
        u_grid = np.array(u_grid[:N])
        v_grid = np.array(v_grid[:N])
        
        for W_init in [0.8, 1.0, 1.2]:
            starts.append(np.concatenate([u_grid, v_grid, [W_init]]))
            
    # Random starts
    for _ in range(5):
        u_rand = np.random.rand(N)
        v_rand = np.random.rand(N)
        W_rand = np.random.uniform(0.7, 1.3)
        starts.append(np.concatenate([u_rand, v_rand, [W_rand]]))
        
    for start in starts:
        try:
            res = minimize(objective_normalized, start, method='Powell', bounds=bounds, options={'maxiter': 100, 'disp': False})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_u = res.x[:N]
                best_v = res.x[N:2*N]
                best_W = res.x[2*N]
                best_H = 2.0 - best_W
                
                best_x = best_u * best_W
                best_y = best_v * best_H
                
                # Recalculate radii for output
                d_wall_x = np.minimum(best_x, best_W - best_x)
                d_wall_y = np.minimum(best_y, best_H - best_y)
                r = np.minimum(d_wall_x, d_wall_y)
                
                dx = best_x[:, np.newaxis] - best_x[np.newaxis, :]
                dy = best_y[:, np.newaxis] - best_y[np.newaxis, :]
                dist = np.sqrt(dx**2 + dy**2)
                np.fill_diagonal(dist, np.inf)
                min_dist = np.min(dist, axis=1)
                r = np.minimum(r, min_dist / 2.0)
                
                best_result = np.column_stack((best_x, best_y, r))
        except Exception:
            continue
            
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
