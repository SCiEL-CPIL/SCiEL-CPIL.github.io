"""Optimized program for second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _find_sidon_set(n_max):
    """
    Find a Sidon set (B2 set) of maximum size in range [0, n_max).
    A Sidon set is a set of integers where all pairwise sums a+b (a<=b) are distinct.
    """
    sidon = []
    used_sums = set()
    
    # Greedy construction
    for i in range(n_max):
        # Check if i can be added
        valid = True
        new_sums = set()
        # Sums involving i and existing elements
        for j in sidon:
            s = i + j
            if s in used_sums:
                valid = False
                break
            new_sums.add(s)
        
        if valid:
            # Check sum of i with itself
            s_ii = 2 * i
            if s_ii in used_sums:
                valid = False
            else:
                new_sums.add(s_ii)
        
        if valid:
            sidon.append(i)
            used_sums.update(new_sums)
            
    return sidon

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

def _neg_log_ratio(amplitudes, sidon_indices):
    """
    Negative log of the ratio for optimization.
    We optimize over amplitudes on the Sidon set support.
    """
    h = np.zeros(N)
    h[sidon_indices] = amplitudes
    
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 1e10
    
    conv_linear = np.correlate(h, h, mode='full')
    sup_norm = np.max(conv_linear)
    if sup_norm < 1e-15:
        return 1e10
    
    ratio = l2_sq / (l1 * sup_norm)
    if ratio <= 0:
        return 1e10
    
    return -np.log(ratio)

def _ratio_from_amplitudes(amplitudes, sidon_indices):
    """Compute the ratio from amplitudes on Sidon set."""
    h = np.zeros(N)
    h[sidon_indices] = amplitudes
    
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    
    conv_linear = np.correlate(h, h, mode='full')
    sup_norm = np.max(conv_linear)
    if sup_norm < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * sup_norm)

def construct():
    """
    Construct 50 nonnegative heights to maximize the ratio.
    Strategy: Use a Sidon set and optimize amplitudes via gradient ascent on log-ratio.
    """
    # Find the largest Sidon set in [0, 50)
    sidon = _find_sidon_set(N)
    
    if not sidon:
        # Fallback: single point
        h = np.zeros(N)
        h[0] = 1.0
        return list(h)
        
    k = len(sidon)
    sidon_arr = np.array(sidon)
    
    # Initialize with equal amplitudes
    init_amplitudes = np.ones(k) / k
    
    # Optimize amplitudes to maximize the ratio
    # Use SLSQP with bounds to keep amplitudes non-negative
    bounds = [(0, None)] * k
    
    # Try multiple initializations to avoid local minima
    best_ratio = 0.0
    best_amplitudes = init_amplitudes.copy()
    
    # Initial score with equal amplitudes
    init_ratio = _ratio_from_amplitudes(init_amplitudes, sidon_arr)
    best_ratio = max(best_ratio, init_ratio)
    
    # Run optimization from several starting points
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if restart == 0:
            x0 = init_amplitudes.copy()
        else:
            # Perturb the initial amplitudes
            x0 = init_amplitudes + rng.normal(0, 0.1, size=k)
            x0 = np.maximum(x0, 0.01)  # Keep positive for log
        
        # Normalize to avoid scale issues (ratio is scale-invariant)
        x0 = x0 / np.sum(x0)
        
        result = minimize(
            _neg_log_ratio,
            x0,
            args=(sidon_arr,),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        
        if result.success:
            # Ensure non-negativity
            x_opt = np.maximum(result.x, 0)
            x_opt = x_opt / np.sum(x_opt)  # Normalize
            
            ratio = _ratio_from_amplitudes(x_opt, sidon_arr)
            if ratio > best_ratio:
                best_ratio = ratio
                best_amplitudes = x_opt
    
    # Construct final heights
    h = np.zeros(N)
    h[sidon_arr] = best_amplitudes
    
    # Final score check
    final_score = _compute_score(h)
    
    return list(h)
# EVOLVE-BLOCK-END
