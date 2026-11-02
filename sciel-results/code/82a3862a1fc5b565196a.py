import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Define the objective function
    # Variables: x1, y1, r1, ..., xN, yN, rN, W
    # H = 2 - W
    def objective(vars):
        if time.time() > deadline:
            return 1e10 # Penalty to stop early if possible, though minimize might not check this often
            # Better to handle timeout in the outer loop or check inside minimize callback? 
            # Minimize doesn't easily support callbacks that abort. 
            # We will rely on maxiter and checking time in the outer loop between restarts.
            
        x = vars[:N]
        y = vars[N:2*N]
        r = vars[2*N:3*N]
        W = vars[3*N]
        H = 2.0 - W
        
        # Check validity of W
        if W <= 0 or W >= 2:
            return 1e10
            
        # Penalty for negative radii
        pen_r = np.sum(np.maximum(0, -r)**2)
        
        # Penalty for boundary violations
        # x_i >= r_i, x_i <= W - r_i
        # y_i >= r_i, y_i <= H - r_i
        pen_x = np.sum(np.maximum(0, r - x)**2) + np.sum(np.maximum(0, x - (W - r))**2)
        pen_y = np.sum(np.maximum(0, r - y)**2) + np.sum(np.maximum(0, y - (H - r))**2)
        
        # Penalty for circle-circle overlaps
        # d_ij >= r_i + r_j
        # Use pairwise distances
        # Vectorized calculation for speed
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        d = np.sqrt(dx**2 + dy**2)
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Overlap is max(0, r_sum - d)
        # We only need i < j
        iu = np.triu_indices(N, k=1)
        overlaps = np.maximum(0, r_sum[iu] - d[iu])
        pen_overlap = np.sum(overlaps**2)
        
        # Maximize sum of radii -> Minimize -sum(r)
        score = -np.sum(r)
        
        # Total cost
        cost = score + 10.0 * (pen_r + pen_x + pen_y) + 10.0 * pen_overlap
        return cost

    # Initial guesses
    # We try a few different aspect ratios and grid layouts
    initial_guesses = []
    
    # 1. Grid 5x5 (with 4 missing)
    # Side length approx sqrt(21) ~ 4.58
    # Let's try W=1.0, H=1.0 (Square)
    # Grid 5x5
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.15) for i in range(side) for j in range(side)]
    cells = cells[:N]
    # Scale to fit in 1x1 box? No, let optimizer do it.
    # Initial radii 0.15 might be too big or small.
    # Let's just provide centers and small radii.
    init_vars = []
    for cx, cy, r in cells:
        init_vars.extend([cx, cy, r])
    init_vars.append(1.0) # W
    initial_guesses.append(np.array(init_vars))
    
    # 2. Rectangle 1.5 x 0.5
    # Grid 7x3 = 21
    W0 = 1.5
    H0 = 0.5
    side_x = 7
    side_y = 3
    cells = [((i + 0.5) / side_x, (j + 0.5) / side_y, 0.1) for i in range(side_x) for j in range(side_y)]
    init_vars = []
    for cx, cy, r in cells:
        # Map to [0,1]x[0,1] for initial guess? 
        # The optimizer works in actual coordinates.
        # If W=1.5, H=0.5, we can place them in that box.
        init_vars.extend([cx * W0, cy * H0, r])
    init_vars.append(W0)
    initial_guesses.append(np.array(init_vars))
    
    # 3. Rectangle 0.5 x 1.5
    W0 = 0.5
    H0 = 1.5
    side_x = 3
    side_y = 7
    cells = [((i + 0.5) / side_x, (j + 0.5) / side_y, 0.1) for i in range(side_x) for j in range(side_y)]
    init_vars = []
    for cx, cy, r in cells:
        init_vars.extend([cx * W0, cy * H0, r])
    init_vars.append(W0)
    initial_guesses.append(np.array(init_vars))

    # Run optimization for each guess
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 10:
            break
            
        try:
            # Nelder-Mead is robust
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res.success or res.fun < best_score:
                # Extract best result
                x = res.x[:N]
                y = res.x[N:2*N]
                r = res.x[2*N:3*N]
                W = res.x[3*N]
                H = 2.0 - W
                
                # Verify validity
                valid = True
                if W <= 0 or W >= 2:
                    valid = False
                else:
                    for k in range(N):
                        if r[k] <= 0 or x[k] < r[k] or x[k] > W - r[k] or y[k] < r[k] or y[k] > H - r[k]:
                            valid = False
                            break
                    if valid:
                        # Check overlaps
                        for i_c in range(N):
                            for j_c in range(i_c + 1, N):
                                dx = x[i_c] - x[j_c]
                                dy = y[i_c] - y[j_c]
                                d = np.sqrt(dx**2 + dy**2)
                                if d < r[i_c] + r[j_c] - 1e-5:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x, y, r))
                        
        except Exception as e:
            pass # Ignore errors in individual runs

    # Fallback to seed if no valid result found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
