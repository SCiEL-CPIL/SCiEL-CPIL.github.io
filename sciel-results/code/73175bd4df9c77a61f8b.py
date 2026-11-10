# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_mobius(N):
    """Compute Moebius function mu(k) for k = 1..N using linear sieve."""
    mu = np.zeros(N + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > N:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def _verify_feasibility(domain, values, K, max_x):
    """
    Check M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for x in [1, max_x].
    Uses the identity: M(x) = sum_{n=1}^{x} g(n) where g(n) = sum_{k|n} f(k).
    """
    m = len(domain)
    if m == 0:
        return True

    # Compute g(n) for n up to max_x
    # g(n) = sum_{k|n} f(k)
    g = np.zeros(max_x + 1, dtype=np.float64)
    
    # We only need to add f(k) to multiples of k up to max_x
    # domain and values are aligned
    for i in range(m):
        k = int(domain[i])
        v = values[i]
        if v != 0.0:
            # Add v to g(n) for all n that are multiples of k
            # n = k, 2k, 3k, ... <= max_x
            g[k:max_x+1:k] += v

    # M(x) = cumsum(g)[x]
    # We need M(x) <= 1.0001 for all x in [1, max_x]
    # M(0) is 0, we check indices 1 to max_x
    M = np.cumsum(g)
    
    # Optimization: We know M(x)=1 for x<=K if f is mu truncated.
    # But we must check the whole range [1, 10K] as per contract.
    # However, for x <= K, M(x) is exactly 1 (or 0 for x=0).
    # The deviation happens for x > K.
    # We can check max(M[K+1:max_x+1]) <= 1.0001
    # But to be safe and generic, check the whole range.
    # Actually, M(x) for x<=K is 1. 1 <= 1.0001 is true.
    # So we only strictly need to check x > K.
    if max_x > K:
        if np.any(M[K+1:max_x+1] > 1.0001):
            return False
    else:
        if np.any(M[1:max_x+1] > 1.0001):
            return False
            
    return True

def _compute_objective(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k"""
    # domain is int array, values is float array
    log_k = np.log(domain.astype(np.float64))
    # f(k) * log(k) / k
    terms = values * log_k / domain.astype(np.float64)
    a = -np.sum(terms)
    return a

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = 600.0
    
    # Strategy: Use the Mobius function mu(k) for k = 1..K.
    # We want the largest K such that M(x) <= 1.0001 for all x in [1, 10K].
    
    # Pre-compute mu up to a large limit.
    # Let's estimate the max K we can afford.
    # Verification is O(K log K) roughly.
    # 600s limit.
    # Let's try to find the max feasible K.
    
    # Start with a small K and expand.
    # Or start with a large K and shrink.
    # Shrinking is safer for time if large K is infeasible.
    # But we don't know the infeasibility threshold.
    
    # Let's try a few candidate Ks.
    # The deviation of M(x) from 1 for truncated Mobius is related to the Mertens function M(n) = sum_{i=1}^n mu(i).
    # M(x) for x > K is 1 + sum_{n=K+1}^x (sum_{d|n} mu(d) - [d<=K part]) ...
    # Actually, M(x) = sum_{k=1}^K mu(k) floor(x/k).
    # We know sum_{k=1}^x mu(k) floor(x/k) = 1.
    # So sum_{k=1}^K mu(k) floor(x/k) = 1 - sum_{k=K+1}^x mu(k) floor(x/k).
    # For this to be <= 1.0001, we need - sum_{k=K+1}^x mu(k) floor(x/k) <= 0.0001
    # i.e., sum_{k=K+1}^x mu(k) floor(x/k) >= -0.0001.
    
    # Let's just use the direct verification method which is robust.
    
    best_domain = None
    best_values = None
    best_a = -np.inf
    
    # We will try K values.
    # Let's precompute mu up to 200,000.
    N_max = 200000
    mu = _compute_mobius(N_max)
    
    # Candidate Ks.
    # We know K=30 is feasible (seed).
    # Let's try larger Ks.
    # The computation of mu is fast.
    # The verification is the cost.
    
    # Let's try a binary search on K?
    # Feasibility is not necessarily monotonic with K?
    # If K is feasible, is K-1 feasible?
    # M_K(x) = sum_{k=1}^K mu(k) floor(x/k).
    # M_{K-1}(x) = M_K(x) - mu(K) floor(x/K).
    # If M_K(x) <= 1.0001, M_{K-1}(x) could be larger if mu(K) > 0.
    # So it's not strictly monotonic.
    
    # However, the "error" term E_K(x) = 1 - M_K(x) = sum_{k=K+1}^x mu(k) floor(x/k).
    # We need M_K(x) <= 1.0001 => 1 - E_K(x) <= 1.0001 => E_K(x) >= -0.0001.
    
    # Let's just scan K from small to large, keeping the best feasible one.
    # But scanning all K up to 200,000 with verification each time is too slow.
    # Verification is O(K log K). Sum of K log K is huge.
    
    # Alternative: Compute M(x) for the full range for a specific K.
    # We can check feasibility for a specific K in O(K log K).
    # We can afford to check maybe 10-20 different K values.
    
    # Let's pick K values: 100, 500, 1000, 5000, 10000, 50000, 100000, 200000.
    # If K is infeasible, try smaller.
    
    K_candidates = [200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000, 500, 100, 50, 30]
    
    for K in K_candidates:
        if time.time() > deadline - 30:
            break
            
        if K > N_max:
            continue
            
        # Extract domain and values for this K
        # domain: 1..K
        # values: mu[1..K]
        # We can optimize by only including non-zero mu?
        # The contract says "domain holds distinct positive integers".
        # If f(k)=0, it doesn't affect M(x) or a.
        # But including them increases m.
        # "m >= 1".
        # Smaller m is better for "efficiency" but doesn't affect score directly.
        # However, the evaluator cost is m * 10K.
        # If we include only non-zero mu, m is ~ 6/pi^2 K.
        # Let's include only non-zero mu to reduce m.
        
        nonzero_indices = np.where(mu[1:K+1] != 0)[0] + 1
        if len(nonzero_indices) == 0:
            continue
            
        domain = nonzero_indices.astype(np.int64)
        values = mu[domain].astype(np.float64)
        
        max_x = 10 * K
        
        # Verify
        if not _verify_feasibility(domain, values, K, max_x):
            continue
            
        # Compute objective
        a = _compute_objective(domain, values)
        
        if a > best_a:
            best_a = a
            best_domain = domain.copy()
            best_values = values.copy()
            
        # If we found a feasible K, we might stop if a is very close to 1?
        # Or continue to smaller Ks to see if a larger K was missed?
        # We are iterating from large to small.
        # The first feasible K is the largest.
        # Since a increases with K (generally), the largest feasible K gives the best a.
        # So we can break after the first feasible K found in descending order.
        break
    
    # Fallback
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values
        
    return domain, values

# EVOLVE-BLOCK-END
