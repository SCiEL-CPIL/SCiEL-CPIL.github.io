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

def _verify_feasibility_fast(mu, K, max_x):
    """
    Check M(x) = sum_{k=1}^K mu(k) floor(x/k) <= 1.0001 for x in [1, max_x].
    
    We compute g(n) = sum_{d|n, d<=K} mu(d) for n=1..max_x.
    Then M(x) = sum_{n=1}^x g(n).
    
    We can compute g(n) by adding mu(d) to all multiples of d.
    g = zeros(max_x+1)
    for d in 1..K:
        if mu[d] != 0:
            g[d::d] += mu[d]
    
    This is O(K * (max_x/K)) = O(max_x) operations if we sum over d, but the slicing is fast.
    Actually, the total number of additions is sum_{d=1}^K max_x/d = max_x * H_K ~ max_x * log K.
    For max_x = 10K, this is 10K log K.
    """
    # mu is defined up to N, we only care up to K
    # We need to accumulate into an array of size max_x + 1
    g = np.zeros(max_x + 1, dtype=np.float64)
    
    # We only add non-zero mu values
    # mu is int8, so we can filter
    nonzero_idx = np.where(mu[1:K+1] != 0)[0] + 1
    
    for d in nonzero_idx:
        g[d:max_x+1:d] += mu[d]
        
    M = np.cumsum(g)
    
    # Check constraint: M(x) <= 1.0001 for x in [1, max_x]
    # Note: M(0)=0. We check 1..max_x
    if np.any(M[1:max_x+1] > 1.0001):
        return False
    return True

def _compute_objective(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k"""
    log_k = np.log(domain.astype(np.float64))
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
    # Verification is O(10K log K).
    # If K=100,000, max_x=1,000,000. 1,000,000 * log(100,000) ~ 10^6 * 11.5 ~ 10^7 ops.
    # NumPy can do 10^7 ops in < 1s.
    # So we can test K up to 200,000 easily.
    
    N_max = 200000
    mu = _compute_mobius(N_max)
    
    best_domain = None
    best_values = None
    best_a = -np.inf
    
    # We will scan K from large to small.
    # The objective a increases with K (approaches 1).
    # So the largest feasible K is the best.
    
    # Generate a list of candidate Ks.
    # We want to find the threshold where feasibility breaks.
    # Let's try a dense scan near the expected limit or a logarithmic scan.
    # Given the speed, we can afford a linear scan if we are smart, or just a few steps.
    # Let's try K values: 200000, 190000, ..., down to a safe floor.
    
    # However, scanning 200000 steps is too much.
    # Let's use a binary search-like approach or just a few strategic points.
    # Actually, since the cost is linear in K (roughly), we can estimate the total budget.
    # Cost(K) ~ C * K * log K.
    # If we test K1, K2, ..., Kn, total cost ~ C * sum Ki log Ki.
    
    # Let's try a geometric sequence of Ks to find the feasible region, then refine.
    # Or just a set of candidates.
    
    # Based on previous runs, K=30 was the seed. 
    # Let's try to push K much higher.
    
    # Candidate Ks:
    # Start high, go down.
    K_candidates = list(range(200000, 0, -1000)) # Step 1000
    # Add some finer steps near the end if needed? 
    # Or just let the loop break if it finds a feasible one?
    # No, we want the LARGEST feasible.
    # So we should iterate from Large to Small and stop at the first feasible one?
    # Wait, is feasibility monotonic?
    # If K is feasible, is K-1 feasible?
    # M_K(x) = M_{K-1}(x) + mu(K) floor(x/K).
    # If mu(K) > 0, M_K(x) >= M_{K-1}(x) for x >= K.
    # If M_K(x) <= 1.0001, M_{K-1}(x) could be smaller, so likely feasible.
    # If mu(K) < 0, M_K(x) <= M_{K-1}(x).
    # If M_K(x) <= 1.0001, M_{K-1}(x) could be larger, so might be infeasible.
    # So feasibility is NOT strictly monotonic decreasing with K.
    # However, the "error" term E_K(x) = 1 - M_K(x) = sum_{k=K+1}^x mu(k) floor(x/k).
    # We need E_K(x) >= -0.0001.
    # As K increases, the sum has fewer terms, so the magnitude of the error generally decreases?
    # Not necessarily, because the terms can be large.
    # But typically, truncating more of the tail reduces the error in the PNT context.
    # Let's assume that if a large K is infeasible, a slightly smaller one might be feasible, 
    # but a much smaller one is definitely feasible.
    # To be safe, we should find the MAXIMUM feasible K.
    # We can scan downwards from 200,000.
    # If we find a feasible K, we can't immediately stop because a larger K might have been feasible 
    # but we skipped it? No, we are scanning downwards. 
    # If 200,000 is infeasible, we check 199,000.
    # If 199,000 is feasible, is 200,000 necessarily infeasible? 
    # We already checked it and it was infeasible.
    # So the first feasible K we find in a descending scan is the largest feasible K 
    # *among the candidates we test*.
    # If the step size is small enough, we find the true max.
    
    # Let's use a step size that balances resolution and speed.
    # Verification for K=200,000 takes ~ 1s?
    # Let's estimate: 
    # g = zeros(2,000,001)
    # Loop over ~ 120,000 non-zero mu values.
    # Each slice addition is fast.
    # Total ops ~ 2,000,000 * log(200,000) ~ 2.4 * 10^7.
    # NumPy can do this in < 0.1s.
    # So we can afford many checks.
    
    # Let's scan with step 1000.
    # 200 steps. 200 * 0.1s = 20s. Very safe.
    
    # But what if the transition is sharp? 
    # Let's start with step 1000.
    # If we find a feasible K, we can then refine upwards with step 10 or 1 to find the exact max.
    
    coarse_step = 1000
    fine_step = 10
    
    # First pass: Coarse scan
    found_feasible_K = None
    for K in range(N_max, 0, -coarse_step):
        if time.time() > deadline - 60:
            break
            
        max_x = 10 * K
        # Optimization: if max_x is too large for memory? 
        # 10 * 200,000 = 2,000,000 floats = 16 MB. Fine.
        
        if not _verify_feasibility_fast(mu, K, max_x):
            continue
            
        found_feasible_K = K
        break
        
    # Second pass: Fine scan upwards from found_feasible_K
    if found_feasible_K is not None:
        # We know found_feasible_K is feasible.
        # We want to go higher.
        # The previous candidate (found_feasible_K + coarse_step) was infeasible.
        # Let's scan from found_feasible_K + 1 up to found_feasible_K + coarse_step - 1
        # Actually, we can just scan upwards from found_feasible_K + 1.
        # But we don't know how far up is feasible.
        # We know found_feasible_K + coarse_step is infeasible.
        # So the max feasible K is in [found_feasible_K, found_feasible_K + coarse_step].
        
        start_K = found_feasible_K
        end_K = min(N_max, found_feasible_K + coarse_step)
        
        best_K_in_range = start_K
        
        # Scan upwards
        for K in range(start_K + 1, end_K + 1, fine_step):
            if time.time() > deadline - 30:
                break
                
            max_x = 10 * K
            if not _verify_feasibility_fast(mu, K, max_x):
                continue
                
            best_K_in_range = K
            
        # Final fine-tune around best_K_in_range?
        # If fine_step was 10, we might have missed K+1..K+9.
        # Let's do a very fine scan around best_K_in_range.
        # Range: [best_K_in_range - fine_step, best_K_in_range + fine_step]
        # But we know best_K_in_range is feasible.
        # We want to push it higher.
        # Let's check best_K_in_range + 1, +2, ... up to +fine_step
        
        final_K = best_K_in_range
        for K in range(best_K_in_range + 1, best_K_in_range + fine_step + 1):
            if time.time() > deadline - 10:
                break
            if K > N_max:
                break
            max_x = 10 * K
            if not _verify_feasibility_fast(mu, K, max_x):
                break
            final_K = K
            
        best_K = final_K
        
    else:
        # No feasible K found in coarse scan? 
        # This shouldn't happen for small K, but just in case.
        # Try very small Ks.
        best_K = 30
        for K in range(100, 0, -1):
             if time.time() > deadline - 10:
                 break
             max_x = 10 * K
             if _verify_feasibility_fast(mu, K, max_x):
                 best_K = K
                 break

    # Construct domain and values for best_K
    # domain: k where mu(k) != 0 for k in 1..best_K
    nonzero_indices = np.where(mu[1:best_K+1] != 0)[0] + 1
    if len(nonzero_indices) == 0:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = nonzero_indices.astype(np.int64)
        values = mu[domain].astype(np.float64)
        
    return domain, values

# EVOLVE-BLOCK-END
