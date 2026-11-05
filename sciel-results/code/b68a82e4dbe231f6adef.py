import numpy as np
import time as _time

class ExercisePolicy:
    def __init__(self, thresholds):
        self.thresholds = thresholds
    
    def exercise(self, current_state, time_index):
        if time_index < 0 or time_index >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
        return current_state < self.thresholds[time_index]

def fit_exercise_policy(training_paths, contract):
    start_time = _time.time()
    deadline = start_time + 110  # Safety margin
    
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, n_steps = paths.shape
    n_exercise = 12  # exercise dates at steps 1..12
    
    dt = 1.0 / 12
    r = 0.05
    K = 100.0
    discount = np.exp(-r * dt)
    
    V = np.zeros((n_exercise, n_paths))
    
    # Last exercise date: j=11, step=12
    V[11, :] = np.maximum(K - paths[:, 12], 0.0)
    
    # Grid for threshold finding
    # S ranges from 0 to K. 1000 points for high precision.
    S_grid = np.linspace(0.0, K, 1000)
    S_grid_norm = S_grid / K
    
    # Pre-compute grid features for Degree 3
    deg = 3
    X_grid = np.ones((len(S_grid), deg + 1))
    for d in range(1, deg + 1):
        X_grid[:, d] = S_grid_norm ** d
        
    thresholds = np.zeros(n_exercise)
    
    # Backward: j=10 down to 0
    for j in range(10, -1, -1):
        # Check time budget
        if _time.time() > deadline - 5:
            break
        
        S_j = paths[:, j+1]
        
        # Fit regression on ITM paths only
        itm_mask = S_j < K
        if not np.any(itm_mask):
            # No ITM paths, assume threshold is 0
            thresholds[j] = 0.0
            V[j, :] = np.maximum(K - S_j, 0.0)
            continue
            
        S_itm = S_j[itm_mask]
        V_next_itm = V[j+1, itm_mask]
        
        # Normalize S_itm to [0, 1] using fixed range [0, K]
        S_itm_norm = S_itm / K
        
        # Build features
        X_itm = np.ones((len(S_itm), deg + 1))
        for d in range(1, deg + 1):
            X_itm[:, d] = S_itm_norm ** d
            
        # Ridge Regression
        lambda_ridge = 1e-4
        A = X_itm.T @ X_itm + lambda_ridge * np.eye(deg + 1)
        b = X_itm.T @ V_next_itm
        try:
            coeffs = np.linalg.solve(A, b)
        except:
            coeffs = np.linalg.lstsq(X_itm, V_next_itm, rcond=None)[0]
            
        # Evaluate Continuation Value on Grid
        C_grid = X_grid @ coeffs
        C_grid = np.maximum(C_grid, 0.0)
        
        # Find Threshold: K - S = C(S)
        # diff = Payoff - Continuation
        # We want the largest S where diff > 0
        diff = (K - S_grid) - C_grid
        
        pos_mask = diff > 0
        if not np.any(pos_mask):
            thresholds[j] = 0.0
        else:
            idx = np.where(pos_mask)[0]
            last_pos_idx = idx[-1]
            if last_pos_idx == len(S_grid) - 1:
                thresholds[j] = S_grid[-1]
            else:
                # Interpolate between last_pos_idx and last_pos_idx + 1
                S1, S2 = S_grid[last_pos_idx], S_grid[last_pos_idx + 1]
                d1, d2 = diff[last_pos_idx], diff[last_pos_idx + 1]
                if d2 - d1 != 0:
                    t = d1 / (d1 - d2)
                    thresholds[j] = S1 + t * (S2 - S1)
                else:
                    thresholds[j] = S1
        
        # Calculate V[j] for all paths
        # V[j] = max(Payoff, Continuation)
        S_j_norm = S_j / K
        X_j = np.ones((n_paths, deg + 1))
        for d in range(1, deg + 1):
            X_j[:, d] = S_j_norm ** d
            
        C_j = X_j @ coeffs
        C_j = np.maximum(C_j, 0.0)
        
        V[j, :] = np.maximum(K - S_j, C_j)
        
    # Safety: ensure thresholds are reasonable
    thresholds = np.clip(thresholds, 0.0, K)
    
    return ExercisePolicy(thresholds)
