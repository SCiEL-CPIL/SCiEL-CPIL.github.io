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
    
    # Use log-spot for regression basis (GBM natural variable)
    # Backward: j=10 down to 0
    for j in range(10, -1, -1):
        if _time.time() > deadline - 5:
            break
        
        S_j = paths[:, j+1]
        payoff = np.maximum(K - S_j, 0.0)
        
        # Use log(S) as the regression variable for better conditioning
        log_S = np.log(S_j + 1e-10)
        log_min = log_S.min()
        log_max = log_S.max()
        log_norm = (log_S - log_min) / (log_max - log_min + 1e-10)
        
        # Polynomial features degree 6 on normalized log-spot
        deg = 6
        X = np.ones((n_paths, deg + 1))
        for d in range(1, deg + 1):
            X[:, d] = log_norm ** d
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, V[j+1, :], rcond=None)
            continuation = X @ coeffs
            continuation = np.maximum(continuation, 0.0)
        except:
            continuation = np.zeros(n_paths)
        
        V[j, :] = np.maximum(payoff, discount * continuation)
    
    # Determine exercise thresholds via crossing point
    thresholds = np.zeros(n_exercise)
    
    for j in range(n_exercise):
        if _time.time() > deadline - 2:
            break
        
        S_j = paths[:, j+1]
        payoff = np.maximum(K - S_j, 0.0)
        
        if j == n_exercise - 1:
            thresholds[j] = K
            continue
        
        log_S = np.log(S_j + 1e-10)
        log_min = log_S.min()
        log_max = log_S.max()
        log_norm = (log_S - log_min) / (log_max - log_min + 1e-10)
        
        deg = 6
        X = np.ones((n_paths, deg + 1))
        for d in range(1, deg + 1):
            X[:, d] = log_norm ** d
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, V[j+1, :], rcond=None)
            continuation = X @ coeffs
            continuation = np.maximum(continuation, 0.0)
        except:
            continuation = np.zeros(n_paths)
        
        diff = payoff - discount * continuation
        
        # Sort by S and find crossing in ITM region
        sort_idx = np.argsort(S_j)
        S_sorted = S_j[sort_idx]
        diff_sorted = diff[sort_idx]
        
        itm_mask = S_sorted < K
        if not itm_mask.any():
            thresholds[j] = 0.0
            continue
        
        itm_indices = np.where(itm_mask)[0]
        S_itm = S_sorted[itm_indices]
        diff_itm = diff_sorted[itm_indices]
        
        if np.all(diff_itm > 0):
            thresholds[j] = S_itm[-1]
        elif np.all(diff_itm < 0):
            thresholds[j] = 0.0
        else:
            pos_mask = diff_itm > 0
            if pos_mask.any():
                last_pos = np.where(pos_mask)[0][-1]
                if last_pos + 1 < len(diff_itm):
                    S1, S2 = S_itm[last_pos], S_itm[last_pos + 1]
                    d1, d2 = diff_itm[last_pos], diff_itm[last_pos + 1]
                    if d2 - d1 != 0:
                        t = d1 / (d1 - d2)
                        thresholds[j] = S1 + t * (S2 - S1)
                    else:
                        thresholds[j] = S1
                else:
                    thresholds[j] = S_itm[last_pos]
            else:
                thresholds[j] = 0.0
    
    thresholds = np.clip(thresholds, 0.0, K)
    
    return ExercisePolicy(thresholds)
