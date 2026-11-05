"""Bermudan put exercise-policy research with threshold-based policy and coordinate ascent optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.stats import norm

_DEADLINE = 145.0  # seconds; safety margin under 150 s wall-clock limit

def _bs_put_price(S, K, r, sigma, T):
    """
    Calculate European put price using Black-Scholes formula.
    S: array of spot prices
    K: strike
    r: risk-free rate
    sigma: volatility
    T: time to maturity
    """
    if T <= 0:
        return np.maximum(K - S, 0.0)
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Put price = K * exp(-r*T) * N(-d2) - S * N(-d1)
    N = norm.cdf
    price = K * np.exp(-r * T) * N(-d2) - S * N(-d1)
    return price

def _find_bs_threshold(K, r, sigma, T_remaining, dt, S_grid):
    """
    Find the spot price S* such that Intrinsic(S*) = Discounted Continuation(S*).
    Intrinsic: K - S
    Continuation: exp(-r*dt) * BS_Put(S, K, r, sigma, T_remaining - dt)
    """
    T_next = T_remaining - dt
    if T_next < 0: T_next = 0
    
    # Vectorized calculation over S_grid
    intrinsic = K - S_grid
    cont_val = np.exp(-r * dt) * _bs_put_price(S_grid, K, r, sigma, T_next)
    
    diff = intrinsic - cont_val
    # Find sign change
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        if np.all(diff > 0):
            return np.max(S_grid)
        else:
            return np.min(S_grid)
            
    idx = sign_change[0]
    S1, S2 = S_grid[idx], S_grid[idx+1]
    d1, d2 = diff[idx], diff[idx+1]
    if d2 == d1:
        return S1
    S_star = S1 - d1 * (S2 - S1) / (d2 - d1)
    return S_star

class ExercisePolicy:
    def __init__(self, thresholds, r, dt):
        self.thresholds = thresholds
        self.r = r
        self.dt = dt
        self.n_dates = len(thresholds)

    def exercise(self, current_state, time):
        """
        current_state: array of stock prices
        time: integer index of the exercise date (0 to 11) or float time in [0,1]
        Returns: boolean array
        """
        if isinstance(time, (int, np.integer)):
            idx = int(time)
        else:
            # Assuming time is normalized to [0, 1]
            idx = int(np.round(time * 12)) - 1
            if idx < 0:
                idx = 0
            if idx >= self.n_dates:
                idx = self.n_dates - 1

        if idx < 0 or idx >= self.n_dates:
            return np.zeros(len(current_state), dtype=bool)
        
        return current_state < self.thresholds[idx]

def _propagate_back(V_t, S, K, r, dt, thresholds, t_end):
    """
    Propagate value V_t back to V_0 using thresholds[0:t_end].
    V_t: value at time t
    S: (n_dates, n_paths)
    thresholds: (n_dates,)
    t_end: index of the last threshold to apply (exclusive, so applies 0 to t_end-1)
    Returns V_0
    """
    V = V_t.copy()
    # Iterate from t_end-1 down to 0
    for t in range(t_end - 1, -1, -1):
        intrinsic = np.maximum(K - S[t], 0.0)
        ex_mask = S[t] < thresholds[t]
        V = intrinsic * ex_mask + np.exp(-r * dt) * V * (~ex_mask)
    return V

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy.
    1. Initialize thresholds using Black-Scholes analytical boundary approximation.
    2. Refine thresholds using coordinate ascent on training paths with efficient propagation.
    """
    t_start = _time.time()

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    S = training_paths
    if S.ndim != 2:
        S = S.reshape(n_dates, -1)
        
    n_paths = S.shape[1]
    dt = T / n_dates
    
    # 1. Initialize thresholds using BS heuristic
    thresholds = np.zeros(n_dates)
    S_min = np.min(S)
    S_max = np.max(S)
    S_grid = np.linspace(max(S_min * 0.5, 1.0), S_max * 1.5, 300)
    
    for t in range(n_dates):
        T_remaining = (n_dates - t) * dt
        thresholds[t] = _find_bs_threshold(K, r, sigma, T_remaining, dt, S_grid)
        
    # Ensure thresholds are within reasonable bounds
    thresholds = np.clip(thresholds, 0.0, K + 10.0)
    
    # Precompute discount factor
    disc = np.exp(-r * dt)
    
    # Helper to evaluate full policy value
    def _evaluate_full(thr):
        V = np.zeros(n_paths)
        for t in range(n_dates - 1, -1, -1):
            intrinsic = np.maximum(K - S[t], 0.0)
            ex_mask = S[t] < thr[t]
            V = intrinsic * ex_mask + disc * V * (~ex_mask)
        return np.mean(V)
    
    best_score = _evaluate_full(thresholds)
    
    # Generate candidate thresholds for each date
    # Use percentiles of the spot distribution at that date
    n_candidates = 100
    candidate_thresholds = []
    for t in range(n_dates):
        pcts = np.linspace(0.01, 99.99, n_candidates)
        vals = np.percentile(S[t], pcts)
        vals = np.sort(np.unique(vals))
        candidate_thresholds.append(vals)
        
    # Coordinate Ascent Optimization
    max_passes = 20
    for pass_num in range(max_passes):
        if _time.time() - t_start > _DEADLINE - 5.0:
            break
            
        improved = False
        for t in range(n_dates):
            if _time.time() - t_start > _DEADLINE - 5.0:
                break
                
            # Calculate V_{t+1} using current thresholds for t+1..n_dates-1
            V_next = np.zeros(n_paths)
            for tt in range(n_dates - 1, t, -1):
                intrinsic = np.maximum(K - S[tt], 0.0)
                ex_mask = S[tt] < thresholds[tt]
                V_next = intrinsic * ex_mask + disc * V_next * (~ex_mask)
            
            best_local_score = -np.inf
            best_local_thresh = thresholds[t]
            
            candidates = candidate_thresholds[t]
            
            # Evaluate each candidate
            for c in candidates:
                # Calculate V_t with candidate threshold c
                intrinsic = np.maximum(K - S[t], 0.0)
                ex_mask = S[t] < c
                V_t = intrinsic * ex_mask + disc * V_next * (~ex_mask)
                
                # Propagate back to 0
                if t > 0:
                    V_0 = _propagate_back(V_t, S, K, r, dt, thresholds, t)
                else:
                    V_0 = V_t
                    
                score = np.mean(V_0)
                
                if score > best_local_score:
                    best_local_score = score
                    best_local_thresh = c
            
            if best_local_thresh != thresholds[t]:
                thresholds[t] = best_local_thresh
                if best_local_score > best_score:
                    best_score = best_local_score
                    improved = True
            
        if not improved:
            break
            
    # Final refinement pass with finer grid around current thresholds
    if _time.time() - t_start < _DEADLINE - 10.0:
        for t in range(n_dates):
            if _time.time() - t_start > _DEADLINE - 5.0:
                break
            
            current = thresholds[t]
            # Create a finer grid around the current threshold
            # Use a small range around the current value
            lower = max(0.0, current - 5.0)
            upper = min(K + 10.0, current + 5.0)
            fine_candidates = np.linspace(lower, upper, 50)
            
            # Calculate V_{t+1}
            V_next = np.zeros(n_paths)
            for tt in range(n_dates - 1, t, -1):
                intrinsic = np.maximum(K - S[tt], 0.0)
                ex_mask = S[tt] < thresholds[tt]
                V_next = intrinsic * ex_mask + disc * V_next * (~ex_mask)
            
            best_local_score = -np.inf
            best_local_thresh = current
            
            for c in fine_candidates:
                intrinsic = np.maximum(K - S[t], 0.0)
                ex_mask = S[t] < c
                V_t = intrinsic * ex_mask + disc * V_next * (~ex_mask)
                
                if t > 0:
                    V_0 = _propagate_back(V_t, S, K, r, dt, thresholds, t)
                else:
                    V_0 = V_t
                    
                score = np.mean(V_0)
                
                if score > best_local_score:
                    best_local_score = score
                    best_local_thresh = c
            
            thresholds[t] = best_local_thresh
            
    return ExercisePolicy(thresholds, r, dt)
# EVOLVE-BLOCK-END
