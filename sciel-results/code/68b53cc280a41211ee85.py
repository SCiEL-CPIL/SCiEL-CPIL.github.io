import numpy as np
from scipy.optimize import minimize
import random

N = 21

def construct():
    best_score = -1
    best_result = None
    
    # Perimeter 4 => W + H = 2
    # Try a range of aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2]
    
    # Number of restarts per aspect ratio
    n_restarts = 5
    
    for w_ratio in aspect_ratios:
        W = w_ratio
        H = 2.0 - W
        
        for _ in range(n_restarts):
            # Initialize variables: x, y, r
            # Strategy: Grid initialization for positions, small radii
            cols = 5
            rows = 4
            # 20 spots. Add 1 extra.
            # Let's just use random positions for diversity, or grid + random.
            # Grid is safer for SLSQP convergence.
            x_init = []
            y_init = []
            r_init = []
            
            # Create 21 positions
            # 5x4 grid = 20.
            for i in range(cols):
                for j in range(rows):
                    x_init.append((i + 0.5) * W / cols)
                    y_init.append((j + 0.5) * H / rows)
                    # Initial radius estimate
                    r_init.append(0.5 * min(W/cols, H/rows) * 0.8)
            
            # Add 21st circle in center or random
            x_init.append(W/2)
            y_init.append(H/2)
            r_init.append(0.5 * min(W/cols, H/rows) * 0.8)
            
            # Shuffle to break symmetry? No, grid is symmetric.
            # Randomize slightly to help escape local minima
            x_init = np.array(x_init) + np.random.uniform(-0.01, 0.01, N)
            y_init = np.array(y_init) + np.random.uniform(-0.01, 0.01, N)
            r_init = np.array(r_init)
            
            # Ensure initial radii are positive and small enough not to violate bounds immediately
            # If a circle is too close to a wall, r must be smaller.
            for i in range(N):
                max_r = min(x_init[i], W - x_init[i], y_init[i], H - y_init[i])
                r_init[i] = min(r_init[i], max_r * 0.9)
                r_init[i] = max(r_init[i], 0.01)
            
            # Combine into vector z = [x_1..x_N, y_1..y_N, r_1..r_N]
            z0 = np.concatenate([x_init, y_init, r_init])
            
            # Define objective function
            def objective(z):
                x = z[:N]
                y = z[N:2*N]
                r = z[2*N:]
                # Maximize sum of r => Minimize negative sum
                return -np.sum(r)
            
            # Define constraints
            # SLSQP requires g(z) >= 0
            constraints = []
            
            # Boundary constraints
            for i in range(N):
                # x_i >= r_i
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda z, i=i: z[i] - z[2*N+i]
                })
                # x_i <= W - r_i  => W - x_i - r_i >= 0
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda z, i=i: W - z[i] - z[2*N+i]
                })
                # y_i >= r_i
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda z, i=i: z[N+i] - z[2*N+i]
                })
                # y_i <= H - r_i  => H - y_i - r_i >= 0
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda z, i=i: H - z[N+i] - z[2*N+i]
                })
                # r_i >= 0
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda z, i=i: z[2*N+i]
                })
            
            # Non-overlap constraints
            for i in range(N):
                for j in range(i+1, N):
                    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
                    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda z, i=i, j=j: (z[i] - z[j])**2 + (z[N+i] - z[N+j])**2 - (z[2*N+i] + z[2*N+j])**2
                    })
            
            # Bounds for SLSQP
            # x in [0, W], y in [0, H], r in [0, min(W,H)/2]
            bounds = []
            for i in range(N):
                bounds.append((0, W))
            for i in range(N):
                bounds.append((0, H))
            for i in range(N):
                bounds.append((0, min(W, H)/2))
            
            # Run optimization
            try:
                res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
                
                if res.success:
                    z_opt = res.x
                    r_opt = z_opt[2*N:]
                    score = np.sum(r_opt)
                    
                    # Verify validity manually
                    x_opt = z_opt[:N]
                    y_opt = z_opt[N:2*N]
                    valid = True
                    for i in range(N):
                        if x_opt[i] < r_opt[i] - 1e-6 or x_opt[i] > W - r_opt[i] + 1e-6:
                            valid = False
                        if y_opt[i] < r_opt[i] - 1e-6 or y_opt[i] > H - r_opt[i] + 1e-6:
                            valid = False
                        if r_opt[i] < -1e-6:
                            valid = False
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                            if dist < r_opt[i] + r_opt[j] - 1e-6:
                                valid = False
                            
                    if valid and score > best_score:
                        best_score = score
                        # Format result: (x, y, r)
                        best_result = np.column_stack((x_opt, y_opt, r_opt))
                        
            except Exception:
                pass
        
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
