"""Optimized program for second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 50

# EVOLVE-BLOCK-START
def _find_sidon_set(n_max, start=0):
    """
    Find a Sidon set (B2 set) of maximum size in range [start, n_max).
    A Sidon set is a set of integers where all pairwise sums a+b (a<=b) are distinct.
    Uses greedy with backtracking for better density.
    """
    best_sidon = []
    best_used_sums = set()
    
    def _greedy(start_idx, current_sidon, current_sums):
        nonlocal best_sidon, best_used_sums
        if len(current_sidon) > len(best_sidon):
            best_sidon = list(current_sidon)
            best_used_sums = set(current_sums)
        
        for i in range(start_idx, n_max):
            valid = True
            new_sums = set()
            for j in current_sidon:
                s = i + j
                if s in current_sums:
                    valid = False
                    break
                new_sums.add(s)
            if valid:
                s_ii = 2 * i
                if s_ii in current_sums:
                    valid = False
                else:
                    new_sums.add(s_ii)
            if valid:
                current_sidon.append(i)
                current_sums.update(new_sums)
                _greedy(i + 1, current_sidon, current_sums)
                current_sidon.pop()
                current_sums.difference_update(new_sums)
    
    _greedy(start, [], set())
    return best_sidon

def _compute_score(heights):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(heights, dtype=np.float64)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    conv_linear = np.correlate(h, h, mode='full')
    sup_norm = np.max(conv_linear)
    if sup_norm < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def _scale_to_min(h):
    """Scale h so that min positive amplitude is 1.0 (maximizes the ratio)."""
    pos = h[h > 1e-15]
    if len(pos) == 0:
        return h
    min_pos = np.min(pos)
    if min_pos < 1e-15:
        return h
    return h / min_pos

def _log_ratio_neg(h):
    """Negative log of the score for minimization."""
    score = _compute_score(h)
    if score <= 0:
        return 1e10
    return -np.log(score)

def _optimize_amplitudes(sidon, x0, max_iter=500):
    """Optimize amplitudes on a Sidon set using L-BFGS-B."""
    k = len(sidon)
    N_local = N
    
    def objective(x):
        if np.any(x < -1e-10):
            return 1e10
        x_clipped = np.maximum(x, 0.0)
        h_opt = np.zeros(N_local)
        h_opt[sidon] = x_clipped
        # Scale to minimize
        h_scaled = _scale_to_min(h_opt)
        return _log_ratio_neg(h_scaled)
    
    bounds = [(0, None) for _ in range(k)]
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < _log_ratio_neg(_scale_to_min(np.concatenate([np.zeros(sidon[0]), x0, np.zeros(N - sidon[-1] - 1)]))):
            x_opt = np.maximum(res.x, 0.0)
            h_opt = np.zeros(N_local)
            h_opt[sidon] = x_opt
            return h_opt
    except Exception:
        pass
    
    # Fallback to initial
    h_init = np.zeros(N_local)
    h_init[sidon] = x0
    return h_init

def construct():
    """
    Construct 50 nonnegative heights to maximize the ratio.
    Strategy: Use a Sidon set as support. Optimize amplitudes using gradient ascent 
    on the log-ratio to break symmetry and suppress off-diagonal peaks.
    Scale to minimum amplitude to maximize the ratio.
    """
    # Find the largest Sidon set in [0, 50)
    sidon = _find_sidon_set(N)
    
    if not sidon:
        # Fallback: single point
        h = np.zeros(N)
        h[0] = 1.0
        return list(h)
        
    k = len(sidon)
    
    # Try multiple initializations
    best_h = None
    best_score = 0.0
    
    # Initial guess 1: equal amplitudes
    x0_equal = np.ones(k)
    h_opt1 = _optimize_amplitudes(sidon, x0_equal, max_iter=500)
    score1 = _compute_score(h_opt1)
    if score1 > best_score:
        best_score = score1
        best_h = h_opt1
    
    # Initial guess 2: geometric progression
    x0_geo = np.linspace(1.0, 2.0, k)
    h_opt2 = _optimize_amplitudes(sidon, x0_geo, max_iter=500)
    score2 = _compute_score(h_opt2)
    if score2 > best_score:
        best_score = score2
        best_h = h_opt2
    
    # Initial guess 3: random
    np.random.seed(42)
    x0_rand = np.random.rand(k) + 0.1
    h_opt3 = _optimize_amplitudes(sidon, x0_rand, max_iter=500)
    score3 = _compute_score(h_opt3)
    if score3 > best_score:
        best_score = score3
        best_h = h_opt3
    
    # Initial guess 4: alternating high/low
    x0_alt = np.ones(k)
    x0_alt[::2] = 1.5
    x0_alt[1::2] = 0.5
    h_opt4 = _optimize_amplitudes(sidon, x0_alt, max_iter=500)
    score4 = _compute_score(h_opt4)
    if score4 > best_score:
        best_score = score4
        best_h = h_opt4
    
    # Scale the best result to minimum amplitude = 1
    if best_h is not None:
        best_h = _scale_to_min(best_h)
        final_score = _compute_score(best_h)
        if final_score > best_score:
            best_score = final_score
        return list(best_h)
    
    # Fallback
    h = np.zeros(N)
    h[sidon] = 1.0
    return list(h)

# EVOLVE-BLOCK-END
