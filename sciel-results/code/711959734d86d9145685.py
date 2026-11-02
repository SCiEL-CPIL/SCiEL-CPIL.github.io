"""
Strategy: Combinatorial search over Sidon sets for support, followed by Nelder-Mead
optimization of log-amplitudes.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    h is a vector of length N.
    """
    if np.max(h) < 1e-12:
        return 0.0
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    if l1 < 1e-12:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-12:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _is_sidon(indices):
    """
    Check if a set of indices forms a Sidon set (B_2 sequence).
    A Sidon set has the property that all pairwise sums (with repetition) are distinct.
    For indices sorted, we check if all i+j (i<=j) are unique.
    """
    sums = set()
    n = len(indices)
    for i in range(n):
        for j in range(i, n):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _generate_random_sidon_candidates(num_candidates, min_size, max_size, seed_offset=0):
    """
    Generate random subsets of [0, N-1] that are Sidon sets.
    """
    candidates = []
    np.random.seed(seed_offset)
    attempts = 0
    max_attempts = 2000
    while len(candidates) < num_candidates and attempts < max_attempts:
        attempts += 1
        # Random size between min and max
        k = np.random.randint(min_size, max_size + 1)
        # Random sample
        support = np.random.choice(N, k, replace=False)
        support.sort()
        if _is_sidon(support):
            # Avoid duplicates in candidates
            if support not in candidates:
                candidates.append(support)
    return candidates

def _optimize_amplitudes_for_support(support, h_full, max_iter=100, timeout=5.0):
    """
    Optimize amplitudes on a fixed support using Nelder-Mead on log-amplitudes.
    Returns the optimized h_full and the best value found.
    """
    k = len(support)
    if k == 0:
        return h_full, 0.0
    
    # Initial amplitudes: uniform
    # We optimize in log-space to maintain positivity.
    # Let x be the log-amplitudes.
    # h_i = exp(x_i) for i in support.
    # The ratio is scale invariant, so we can fix one parameter or just let NM handle it.
    # However, NM works better with bounded or well-scaled variables.
    # Let's normalize the initial amplitudes to sum to 1.
    
    h_init = np.zeros(N)
    h_init[support] = 1.0 / k
    
    # Define the objective function for minimization (negative of log ratio)
    def objective(x):
        h = np.zeros(N)
        # x are log amplitudes. To avoid overflow/underflow issues in NM,
        # we can use a softplus or just exp. Since NM moves locally, exp is fine.
        # We normalize h to sum to 1 inside the objective to remove scale invariance
        # and keep numbers stable.
        h_vals = np.exp(x)
        s = np.sum(h_vals)
        if s < 1e-12:
            return 1e6
        h[support] = h_vals / s
        val = _compute_ratio(h)
        if val < 1e-12:
            return 1e6
        return -np.log(val)
    
    # Initial x
    x0 = np.full(k, np.log(1.0 / k))
    
    # Options for Nelder-Mead
    options = {
        'maxiter': max_iter,
        'xatol': 1e-6,
        'fatol': 1e-6,
        'adaptive': True
    }
    
    try:
        start_time = time.time()
        # NM doesn't have a built-in timeout, so we rely on maxiter and the outer loop check.
        # To strictly enforce timeout, we could use a callback, but for small k, maxiter is sufficient.
        res = minimize(objective, x0, method='Nelder-Mead', options=options)
        
        # Reconstruct h from the result
        h_opt = np.zeros(N)
        h_vals = np.exp(res.x)
        s = np.sum(h_vals)
        if s > 1e-12:
            h_opt[support] = h_vals / s
        
        val = _compute_ratio(h_opt)
        return h_opt, val
        
    except Exception:
        return h_full, _compute_ratio(h_full)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_val = 0.0
    best_h = np.ones(N)
    
    # 1. Generate a pool of Sidon set supports
    # We want supports of various sizes. Larger sizes generally allow for higher L2/L1 ratio
    # if the autoconv peak doesn't grow too fast.
    # For N=50, max Sidon set size is around 7-8 (since N ~ K^2/2 -> K ~ sqrt(2N) ~ 10).
    # Actually, max B2 sequence in [1, N] has size ~ sqrt(N).
    # Let's try sizes 5 to 15. Some might not exist or be rare, but random generation handles it.
    
    sidon_supports = _generate_random_sidon_candidates(
        num_candidates=50, 
        min_size=5, 
        max_size=15, 
        seed_offset=42
    )
    
    # Also add some random non-Sidon supports for comparison, just in case
    # but prioritize Sidon.
    random_supports = []
    np.random.seed(123)
    for _ in range(20):
        k = np.random.randint(5, 16)
        s = np.random.choice(N, k, replace=False)
        s.sort()
        random_supports.append(s)
        
    all_supports = sidon_supports + random_supports
    
    # Shuffle to mix them up, but ensure we try them all if time permits
    np.random.shuffle(all_supports)
    
    for support in all_supports:
        if time.time() > deadline:
            break
            
        h_init = np.zeros(N)
        h_init[support] = 1.0
        
        # Run Nelder-Mead optimization
        h_opt, val = _optimize_amplitudes_for_support(support, h_init, max_iter=50)
        
        if val > best_val:
            best_val = val
            best_h = h_opt.copy()
            
        # Check time
        if time.time() > deadline - 10.0:
            break
            
    # Final check: if best_val is still 0 (shouldn't happen), return uniform
    if best_val < 1e-6:
        best_h = np.ones(N)
        best_val = _compute_ratio(best_h)
        
    return list(best_h)
