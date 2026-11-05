"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3(a):
    """Compute C3 upper bound: max over k of |sum_n a_n * a_{n+k} * a_{n+2k}|"""
    n = len(a)
    best = 0.0
    for k in range(1, n):
        # For lag k, valid n range: 0 <= n, n+k < n, n+2k < n
        # So n < n - 2k, i.e., n < N - 2k
        valid_len = n - 2 * k
        if valid_len <= 0:
            break
        vals = a[:valid_len] * a[k:k + valid_len] * a[2 * k:2 * k + valid_len]
        s = np.sum(vals)
        if abs(s) > best:
            best = abs(s)
    return best

def _logistic_sign_sequence(mu, x0, n=N):
    """Generate ±1 sequence from logistic map."""
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = mu * x[i - 1] * (1.0 - x[i - 1])
        # Keep in (0,1) to avoid numerical issues
        if x[i] <= 0.0 or x[i] >= 1.0:
            x[i] = 0.5 + (x[i] - 0.5) * 0.999
    # Sign: +1 if x > 0.5, -1 otherwise
    return np.where(x > 0.5, 1.0, -1.0)

def _local_search(a, max_flips=500, deadline=None):
    """Local search: flip signs to reduce C3."""
    a = a.copy()
    current_c3 = _compute_c3(a)
    n = len(a)
    
    # Precompute for efficiency
    def c3_after_flip(idx):
        """Compute change in C3 if we flip a[idx]."""
        # This is expensive; just compute full C3 for simplicity in small iterations
        a[idx] *= -1
        val = _compute_c3(a)
        a[idx] *= -1
        return val
    
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try random flips
        np.random.shuffle(np.arange(n))
        for idx in np.arange(n):
            if deadline is not None and time.time() > deadline:
                break
            a[idx] *= -1
            new_c3 = _compute_c3(a)
            if new_c3 < current_c3:
                current_c3 = new_c3
                improved = True
            else:
                a[idx] *= -1
    return a, current_c3

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    best_a = np.ones(N)
    best_c3 = _compute_c3(best_a)
    
    # Try multiple mu values near 4.0 where chaos is strongest
    mu_candidates = [3.99, 3.995, 3.999, 4.0, 3.9999, 3.99999, 3.98, 3.97, 3.95, 3.9]
    
    for mu in mu_candidates:
        if time.time() > deadline:
            break
        for x0 in [0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9]:
            if time.time() > deadline:
                break
            a = _logistic_sign_sequence(mu, x0)
            c3 = _compute_c3(a)
            if c3 < best_c3:
                best_c3 = c3
                best_a = a.copy()
        
        # Local search on best candidate for this mu
        if time.time() < deadline - 10.0:
            a_ls, c3_ls = _local_search(best_a, deadline=deadline - 5.0)
            if c3_ls < best_c3:
                best_c3 = c3_ls
                best_a = a_ls
    
    # Final local search if time permits
    if time.time() < deadline - 5.0:
        a_final, c3_final = _local_search(best_a, deadline=deadline - 1.0)
        if c3_final < best_c3:
            best_c3 = c3_final
            best_a = a_final
    
    return best_a
# EVOLVE-BLOCK-END
