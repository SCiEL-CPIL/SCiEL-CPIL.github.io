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

def _compute_score_normalized(x):
    """
    Compute the score for amplitudes x, normalized to L1=1.
    This removes scale ambiguity.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    l1 = np.sum(x)
    if l1 < 1e-15:
        return 0.0
    h = x / l1
    l2_sq = np.sum(h * h)
    conv_linear = np.correlate(h, h, mode='full')
    sup_norm = np.max(conv_linear)
    if sup_norm < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_norm) # Note: l1 is 1 here, so effectively l2_sq / sup_norm

def _neg_log_score_norm(x):
    """Negative log of the normalized score for minimization."""
    score = _compute_score_normalized(x)
    if score <= 1e-15:
        return 1e10
    return -np.log(score)

def construct():
    """
    Construct 50 nonnegative heights to maximize the ratio.
    Strategy: Use a Sidon set as support. Optimize amplitudes using Nelder-Mead 
    on the normalized log-ratio to break symmetry and suppress off-diagonal peaks.
    """
    # Find the largest Sidon set in [0, 50)
    sidon = _find_sidon_set(N)
    
    if not sidon:
        # Fallback: single point
        h = np.zeros(N)
        h[0] = 1.0
        return list(h)
        
    k = len(sidon)
    
    # Initialize with equal amplitudes on the Sidon set
    x0 = np.ones(k) / k
    
    # We will optimize x such that h is supported on sidon and L1=1
    # The objective function handles normalization internally
    
    best_score = 0.0
    best_x = x0.copy()
    
    # Run multiple restarts to avoid local minima
    np.random.seed(42)
    for restart in range(10):
        if restart == 0:
            x_start = x0.copy()
        else:
            # Random perturbation
            x_start = np.random.dirichlet(np.ones(k))
        
        try:
            # Nelder-Mead is robust to non-smoothness (max in sup norm)
            res = minimize(_neg_log_score_norm, x_start, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.success or res.fun < -np.log(best_score) if best_score > 0 else True:
                current_score = _compute_score_normalized(res.x)
                if current_score > best_score:
                    best_score = current_score
                    best_x = res.x
        except Exception:
            continue
            
    # Construct the final vector
    h_final = np.zeros(N)
    h_final[sidon] = best_x
    
    # Normalize to L1=1 to be consistent with standard conventions
    l1 = np.sum(h_final)
    if l1 > 1e-15:
        h_final /= l1
    else:
        h_final = np.zeros(N)
        h_final[0] = 1.0
        
    return list(h_final)

# EVOLVE-BLOCK-END
