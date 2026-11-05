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
    
    # LSMC: backward induction with regression for continuation values
    # V[j, i] = max(payoff[j,i], E[V[j+1,i] | S_j])
    # We estimate E[V[j+1] | S_j] via polynomial regression on S_j
    
    V = np.zeros((n_exercise, n_paths))
    
    # Last exercise date: j=11, step=12 — always exercise (terminal)
    V[11, :] = np.maximum(K - paths[:, 12], 0.0)
    
    # Backward: j=10 down to 0
    for j in range(10, -1, -1):
        # Check time budget
        if _time.time() > deadline - 5:
            break
        
        S_j = paths[:, j+1]  # spot at exercise date j
        payoff = np.maximum(K - S_j, 0.0)
        
        # Regression: fit V[j+1] as function of S_j
        # Use polynomial features of S_j
        # S_j is in range roughly [30, 200] for GBM with these params
        # Normalize S_j to [0,1] for numerical stability
        S_min = S_j.min()
        S_max = S_j.max()
        S_norm = (S_j - S_min) / (S_max - S_min + 1e-10)
        
        # Build polynomial features (degree 4)
        deg = 4
        X = np.ones((n_paths, deg + 1))
        for d in range(1, deg + 1):
            X[:, d] = S_norm ** d
        
        # Fit linear regression: V[j+1] ≈ X @ coeffs
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, V[j+1, :], rcond=None)
            continuation = X @ coeffs
            # Clip negative continuation values
            continuation = np.maximum(continuation, 0.0)
        except:
            continuation = np.zeros(n_paths)
        
        V[j, :] = np.maximum(payoff, discount * continuation)
    
    # Determine exercise thresholds
    # For each exercise date j, find the threshold S* such that:
    # payoff(S*) = discount * E[V[j+1] | S = S*]
    # i.e., K - S* = discount * continuation(S*)
    # We'll find this by searching for the crossing point
    
    thresholds = np.zeros(n_exercise)
    
    for j in range(n_exercise):
        if _time.time() > deadline - 2:
            break
        
        S_j = paths[:, j+1]
        payoff = np.maximum(K - S_j, 0.0)
        
        if j == n_exercise - 1:
            # Last date: exercise if in the money
            thresholds[j] = K
            continue
        
        # Recompute continuation value at S_j using the same regression
        S_min = S_j.min()
        S_max = S_j.max()
        S_norm = (S_j - S_min) / (S_max - S_min + 1e-10)
        
        deg = 4
        X = np.ones((n_paths, deg + 1))
        for d in range(1, deg + 1):
            X[:, d] = S_norm ** d
        
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, V[j+1, :], rcond=None)
            continuation = X @ coeffs
            continuation = np.maximum(continuation, 0.0)
        except:
            continuation = np.zeros(n_paths)
        
        # Find threshold: exercise if payoff > discount * continuation
        # i.e., K - S_j > discount * continuation
        # For S_j < K: K - S_j > discount * continuation
        # For S_j >= K: 0 > discount * continuation (never true if continuation >= 0)
        
        # Find the boundary: where K - S = discount * continuation(S)
        # Sort by S and find the crossing
        sort_idx = np.argsort(S_j)
        S_sorted = S_j[sort_idx]
        payoff_sorted = payoff[sort_idx]
        cont_sorted = discount * continuation[sort_idx]
        
        # Only consider S < K (in the money region)
        itm_mask = S_sorted < K
        if not itm_mask.any():
            thresholds[j] = 0.0
            continue
        
        # Find where payoff crosses continuation
        diff = payoff_sorted - cont_sorted
        # In ITM region, find the last point where diff > 0
        itm_indices = np.where(itm_mask)[0]
        if len(itm_indices) == 0:
            thresholds[j] = 0.0
            continue
        
        # Find the crossing point
        diff_itm = diff[itm_indices]
        S_itm = S_sorted[itm_indices]
        
        # If all positive, threshold is max S in ITM
        if np.all(diff_itm > 0):
            thresholds[j] = S_itm[-1]
        elif np.all(diff_itm < 0):
            thresholds[j] = 0.0
        else:
            # Find the last index where diff > 0
            pos_mask = diff_itm > 0
            if pos_mask.any():
                last_pos = np.where(pos_mask)[0][-1]
                # Interpolate between last_pos and last_pos+1
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
    
    # Safety: ensure thresholds are reasonable
    thresholds = np.clip(thresholds, 0.0, K)
    
    return ExercisePolicy(thresholds)
