# EVOLVE-BLOCK-START
import numpy as np
import time

def compute_mobius(n):
    """Compute Möbius function mu(k) for k = 0..n using linear sieve."""
    mu = np.zeros(n + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def check_feasibility_fast(f, K, max_x):
    """Check if M(x) <= 1.0001 for all x in [1, max_x] using difference array."""
    diff = np.zeros(max_x + 2, dtype=np.float64)
    for k in range(1, K + 1):
        fk = f[k]
        if fk == 0:
            continue
        j_max = max_x // k
        if j_max == 0:
            continue
        for j in range(1, j_max + 1):
            start = j * k
            end = min((j + 1) * k - 1, max_x)
            diff[start] += fk * j
            diff[end + 1] -= fk * j
    M = np.cumsum(diff)
    return np.max(M[1:max_x + 1]) <= 1.0001

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    Uses a Two-Block Heuristic: f(k) = -10 for k in [2, M], f(k) = v for k in [M+1, K].
    """
    start_time = time.time()
    deadline = start_time + 580
    
    # We will try increasing K
    # The heuristic is O(K) per (M, v) check, so we can afford large K
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try K values
    # K=100000 is 100k. 10K = 1M. Check feasibility is O(K * 10K / K) = O(10K)?
    # No, check_feasibility_fast is O(K * (10K/K)) = O(10K) if we use the diff array efficiently?
    # Actually, the inner loop runs 10K/K = 10 times on average? No, sum_{k=1}^K 10K/k = 10K log K.
    # For K=10000, 10K log K ~ 100000 * 10 = 1M ops. Fast.
    # For K=100000, 1M * 12 = 12M ops. Fast.
    
    K_list = [1000, 5000, 10000, 20000, 50000, 100000]
    
    for K in K_list:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
            
        max_x = 10 * K
        
        # We search over M (cutoff for negative block)
        # M should be small relative to K
        # Let's try M from 2 to 100
        best_M = -1
        best_v = -1
        best_a = -np.inf
        
        # Precompute log_k / k for score
        ks = np.arange(2, K + 1, dtype=np.float64)
        log_ks = np.log(ks)
        weights = log_ks / ks
        
        # We need to estimate max_x R(x) for a given M
        # R(x) = (10 * sum_{k=2}^M {x/k} - 1) / sum_{k=M+1}^K {x/k}
        # We will sample x values
        # x in [1, 10K]
        # Critical x are likely small or multiples of large numbers
        
        # To speed up, we can just try a few M values and a few x values
        # But we must ensure feasibility.
        
        # Let's try M in [2, 5, 10, 20, 50, 100]
        M_candidates = [2, 3, 5, 10, 20, 50, 100, 200]
        
        for M in M_candidates:
            if M >= K:
                continue
            
            # Estimate v_min
            # We need max over x of (10 * S1(x) - 1) / S2(x)
            # S1(x) = sum_{k=2}^M {x/k}
            # S2(x) = sum_{k=M+1}^K {x/k}
            
            # Sample x
            # x = 1..100
            # x = K, 2K, ..., 10K
            # x = random?
            x_samples = list(range(1, 101))
            x_samples.extend(range(K, 10*K + 1, K)) # K, 2K, ...
            # Add some randoms
            if time.time() - start_time < deadline - 50:
                x_samples.extend(np.random.randint(1, 10*K, 100).tolist())
            
            v_min_required = 0.0
            for x in x_samples:
                # Compute S1 and S2
                # S1 is small M
                s1 = 0.0
                for k in range(2, M + 1):
                    s1 += (x % k) / k
                
                # S2 is large
                # Vectorize?
                ks2 = np.arange(M + 1, K + 1, dtype=np.float64)
                # {x/k} = x/k - floor(x/k)
                # floor(x/k) = x // k
                s2_vals = (x % ks2.astype(np.int64)) / ks2
                s2 = np.sum(s2_vals)
                
                if s2 > 1e-6:
                    req_v = (10.0 * s1 - 1.0) / s2
                    if req_v > v_min_required:
                        v_min_required = req_v
            
            # If v_min_required > 10, infeasible
            if v_min_required > 10.0:
                continue
            
            # We can set v = v_min_required + epsilon
            # But we want to maximize score, so we want v as small as possible
            # However, if we set v too small, we might violate constraints for x not sampled
            # Let's add a safety margin
            v = v_min_required + 0.1
            if v > 10.0:
                v = 10.0
            
            # Construct f
            f = np.zeros(K + 1, dtype=np.float64)
            f[2:M+1] = -10.0
            f[M+1:K+1] = v
            
            # Calculate f(1)
            S = np.sum(f[2:K+1] / ks)
            f[1] = -S
            
            # Check feasibility
            if check_feasibility_fast(f, K, max_x):
                # Calculate score
                a = -np.sum(f[2:K+1] * weights)
                
                if a > best_a:
                    best_a = a
                    best_M = M
                    best_v = v
                    best_domain = np.arange(1, K + 1, dtype=np.int64)
                    best_values = f[1:K+1]
                    
                    # Update global best
                    if best_a > best_score:
                        best_score = best_a
                        # Save best result
                        global_best_domain = best_domain
                        global_best_values = best_values
        
        # If we found a good M, we might try to refine v?
        # Or try other M?
        
    # Fallback to Möbius if heuristic failed
    if best_domain is None or best_score < 0.5:
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            max_K = 200000
            mu = compute_mobius(max_K)
            
            for K in [200000, 100000, 50000]:
                if time.time() > deadline - 10:
                    break
                max_x = 10 * K
                f = np.zeros(K + 1, dtype=np.float64)
                f[1:K + 1] = mu[1:K + 1].astype(np.float64)
                
                if check_feasibility_fast(f, K, max_x):
                    mask = mu[1:K + 1] != 0
                    domain = np.nonzero(mask)[0] + 1
                    values = mu[1:K + 1][mask].astype(np.float64)
                    
                    log_k = np.log(domain.astype(np.float64))
                    a = -np.sum(values * log_k / domain.astype(np.float64))
                    
                    if a > best_score:
                        best_score = a
                        best_domain = domain
                        best_values = values
                    break
    
    # Fallback
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values

# EVOLVE-BLOCK-END
