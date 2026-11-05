"""Bermudan exercise-policy research with hybrid BS-initialization and continuous coordinate ascent optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.stats import norm
from scipy.optimize import minimize_scalar

_DEADLINE = 115.0  # seconds; safety margin under 120s wall-clock limit (150s kill)

def _bs_put_price(S, K, r, sigma, T):
    if T <= 0:
        return np.maximum(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N = norm.cdf
    price = K * np.exp(-r * T) * N(-d2) - S * N(-d1)
    return price

def _find_bs_threshold(K, r, sigma, T_remaining, dt, S_grid):
    T_next = T_remaining - dt
    if T_next < 0: T_next = 0
    intrinsic = K - S_grid
    cont_val = np.exp(-r * dt) * _bs_put_price(S_grid, K, r, sigma, T_next)
    diff = intrinsic - cont_val
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
        if isinstance(time, (int, np.integer)):
            idx = int(time)
        else:
            idx = int(np.round(time * 12)) - 1
            if idx < 0:
                idx = 0
            if idx >= self.n_dates:
                idx = self.n_dates - 1
        if idx < 0 or idx >= self.n_dates:
            return np.zeros(len(current_state), dtype=bool)
        return current_state < self.thresholds[idx]

def _evaluate_policy_fast(S, K, r, T, n_dates, thresholds, Intrinsic=None, disc=None):
    dt = T / n_dates
    n_paths = S.shape[1]
    if Intrinsic is None:
        Intrinsic = np.maximum(K - S, 0.0)
    if disc is None:
        disc = np.exp(-r * dt)
    V = np.zeros(n_paths)
    for t in range(n_dates - 1, -1, -1):
        ex_mask = S[t] < thresholds[t]
        V = Intrinsic[t] * ex_mask + disc * V * (~ex_mask)
    return np.mean(V)

def _precompute_prefix_affine(S, K, r, T, n_dates, thresholds, t_end, Intrinsic=None, disc=None):
    dt = T / n_dates
    n_paths = S.shape[1]
    if Intrinsic is None:
        Intrinsic = np.maximum(K - S, 0.0)
    if disc is None:
        disc = np.exp(-r * dt)
    
    C = np.ones(n_paths)
    D = np.zeros(n_paths)
    
    for k in range(t_end - 1, -1, -1):
        ex_mask = S[k] < thresholds[k]
        A_k = (~ex_mask) * disc
        B_k = ex_mask * Intrinsic[k]
        
        C = A_k * C
        D = A_k * D + B_k
        
    return C, D

def fit_exercise_policy(training_paths, contract):
    t_start = _time.time()

    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    S = training_paths
    if S.ndim != 2:
        if S.shape[1] == n_dates:
            S = S.T
        else:
            S = S.reshape(n_dates, -1)
            
    n_paths = S.shape[1]
    dt = T / n_dates
    disc = np.exp(-r * dt)
    Intrinsic = np.maximum(K - S, 0.0)
    
    # 1. Initialize thresholds using BS heuristic
    thresholds = np.zeros(n_dates)
    S_min = np.min(S)
    S_max = np.max(S)
    S_grid = np.linspace(S_min * 0.5, S_max * 1.5, 200)
    
    for t in range(n_dates):
        T_remaining = (n_dates - t) * dt
        thresholds[t] = _find_bs_threshold(K, r, sigma, T_remaining, dt, S_grid)
        
    thresholds = np.clip(thresholds, 0.0, K + 10.0)
    
    # 2. Coordinate Ascent Optimization
    n_candidates = 100
    candidate_thresholds = []
    for t in range(n_dates):
        pcts = np.linspace(0.01, 99.99, n_candidates)
        vals = np.percentile(S[t], pcts)
        vals = np.concatenate([[0.0], vals, [K + 100.0]])
        vals = np.sort(np.unique(vals))
        candidate_thresholds.append(vals)
        
    best_score = _evaluate_policy_fast(S, K, r, T, n_dates, thresholds, Intrinsic, disc)
    
    while _time.time() - t_start < _DEADLINE - 10.0:
        improved = False
        for t in range(n_dates):
            if _time.time() - t_start > _DEADLINE - 10.0:
                break
                
            # Precompute V_{t+1}
            V_next = np.zeros(n_paths)
            for k in range(n_dates - 1, t, -1):
                ex_mask = S[k] < thresholds[k]
                V_next = Intrinsic[k] * ex_mask + disc * V_next * (~ex_mask)
                
            # Precompute C, D
            C, D = _precompute_prefix_affine(S, K, r, T, n_dates, thresholds, t, Intrinsic, disc)
            
            candidates = candidate_thresholds[t]
            best_local_score = -np.inf
            best_local_thresh = thresholds[t]
            
            for c in candidates:
                ex_mask = S[t] < c
                V_t = Intrinsic[t] * ex_mask + disc * V_next * (~ex_mask)
                V_0 = C * V_t + D
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

    # Phase 2: Continuous Coordinate Ascent (Polish)
    if _time.time() - t_start < _DEADLINE - 2.0:
        bounds = (0.0, 2.0 * K)
        
        while _time.time() - t_start < _DEADLINE - 2.0:
            progress = False
            for t in range(n_dates):
                if _time.time() - t_start > _DEADLINE - 2.0:
                    break
                
                V_next = np.zeros(n_paths)
                for k in range(n_dates - 1, t, -1):
                    ex_mask = S[k] < thresholds[k]
                    V_next = Intrinsic[k] * ex_mask + disc * V_next * (~ex_mask)
                
                C, D = _precompute_prefix_affine(S, K, r, T, n_dates, thresholds, t, Intrinsic, disc)
                
                def fast_obj(theta):
                    ex_mask = S[t] < theta
                    V_t = Intrinsic[t] * ex_mask + disc * V_next * (~ex_mask)
                    V_0 = C * V_t + D
                    return -np.mean(V_0)
                
                try:
                    res = minimize_scalar(fast_obj, bounds=bounds, method='bounded', 
                                          options={'xatol': 1e-4})
                    if res.success:
                        new_thresh = res.x
                        old_thresh = thresholds[t]
                        thresholds[t] = new_thresh
                        new_score = _evaluate_policy_fast(S, K, r, T, n_dates, thresholds, Intrinsic, disc)
                        if new_score > best_score:
                            best_score = new_score
                            progress = True
                        else:
                            thresholds[t] = old_thresh
                except Exception:
                    pass
            
            if not progress:
                break

    return ExercisePolicy(thresholds, r, dt)
# EVOLVE-BLOCK-END
