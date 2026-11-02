"""Optimized program for second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _find_sidon_set(n_max):
    """
    Find a Sidon set (B2 set) of maximum size in range [0, n_max).
    A Sidon set is a set of integers where all pairwise sums a+b (a<=b) are distinct.
    Uses greedy construction with backtracking for better quality.
    """
    best_sidon = []
    best_size = 0
    
    def backtrack(start, current_sidon, used_sums):
        nonlocal best_sidon, best_size
        
        if len(current_sidon) > best_size:
            best_size = len(current_sidon)
            best_sidon = list(current_sidon)
        
        if start >= n_max:
            return
        
        # Try adding each candidate from start to n_max-1
        for i in range(start, n_max):
            # Check if i can be added
            valid = True
            new_sums = set()
            # Sums involving i and existing elements
            for j in current_sidon:
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
                # Add i and recurse
                current_sidon.append(i)
                used_sums.update(new_sums)
                backtrack(i + 1, current_sidon, used_sums)
                # Remove i and undo
                current_sidon.pop()
                used_sums.difference_update(new_sums)
                
                # Early termination: if we've already found a large enough set
                # and continuing won't help, stop
                if best_size >= 8:  # For N=50, we expect around 7-8 elements
                    return
    
    backtrack(0, [], set())
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

def _log_ratio_neg(h):
    """Negative log of the score for minimization."""
    score = _compute_score(h)
    if score <= 0:
        return 1e10
    return -np.log(score)

def _optimize_amplitudes(sidon, x0, max_iter=500):
    """Optimize amplitudes on the given Sidon set."""
    k = len(sidon)
    
    def objective(x):
        h_opt = np.zeros(N)
        h_opt[sidon] = x
        # Ensure non-negativity (minimize penalty if negative)
        if np.any(x < 0):
            return 1e10 + np.sum(x[x < 0]**2)
        return _log_ratio_neg(h_opt)
    
    bounds = [(0, None) for _ in range(k)]
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})
        if res.success:
            x_opt = res.x
            h_opt = np.zeros(N)
            h_opt[sidon] = x_opt
            score_opt = _compute_score(h_opt)
            return h_opt, score_opt
        else:
            return None, 0.0
    except Exception:
        return None, 0.0

def construct():
    """
    Construct 50 nonnegative heights to maximize the ratio.
    Strategy: Use a Sidon set as support. Optimize amplitudes using multi-start 
    gradient ascent on the log-ratio to break symmetry and suppress off-diagonal peaks.
    """
    # Find the largest Sidon set in [0, 50)
    sidon = _find_sidon_set(N)
    
    if not sidon:
        # Fallback: single point
        h = np.zeros(N)
        h[0] = 1.0
        return list(h)
        
    k = len(sidon)
    
    # Generate multiple initializations
    np.random.seed(42)
    initializations = []
    
    # 1. Equal amplitudes
    h_eq = np.zeros(N)
    h_eq[sidon] = 1.0 / k
    initializations.append(h_eq[sidon].copy())
    
    # 2. Random perturbations of equal amplitudes
    for _ in range(5):
        x_rand = np.ones(k) / k
        x_rand += np.random.randn(k) * 0.1
        x_rand = np.maximum(x_rand, 0)
        x_rand /= np.sum(x_rand) if np.sum(x_rand) > 0 else 1
        initializations.append(x_rand)
    
    # 3. Geometric progression
    x_geo = np.geomspace(0.5, 2.0, k)
    x_geo /= np.sum(x_geo)
    initializations.append(x_geo)
    
    # 4. Alternating high/low
    x_alt = np.zeros(k)
    x_alt[0::2] = 1.5
    x_alt[1::2] = 0.5
    x_alt /= np.sum(x_alt)
    initializations.append(x_alt)
    
    # Optimize from each initialization and keep the best
    best_h = None
    best_score = 0.0
    
    for x0 in initializations:
        h_opt, score_opt = _optimize_amplitudes(sidon, x0, max_iter=300)
        if h_opt is not None and score_opt > best_score:
            best_score = score_opt
            best_h = h_opt
    
    # If optimization failed, use the best initial
    if best_h is None:
        best_h = h_eq
    
    return list(best_h)

# EVOLVE-BLOCK-END
