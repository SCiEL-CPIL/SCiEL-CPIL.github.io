# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600

    def mobius_sieve(n):
        """Compute Mobius function mu(k) for k=1..n using a linear sieve."""
        mu = np.ones(n + 1, dtype=np.int8)
        mu[0] = 0
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
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

    def check_feasibility(domain, values, K, batch_size=20000):
        """
        Check M(x) <= 1.0001 for all x in [1, 10K] using vectorized ops.
        For x <= K, M(x) = sum_{k<=x} f(k) * floor(x/k).
        For f = mu, this sum is 1 for x >= 1 (identity sum_{d|x} mu(d) = 0 for x>1, 1 for x=1).
        Wait, the identity is sum_{d|x} mu(d) = 0 for x > 1.
        M(x) = sum_{k} mu(k) floor(x/k).
        Let's check the identity: sum_{k=1}^x mu(k) floor(x/k) = 1.
        Yes, this is a standard identity.
        So for x <= K, M(x) should be 1 exactly (if we have all k up to x).
        Since we have all k up to K, for x <= K we have all k <= x, so M(x) = 1.
        Thus we only need to check x in [K+1, 10K].
        """
        max_x = 10 * K
        if K >= max_x:
            return True
        
        check_start = K + 1
        check_end = max_x
        
        # We can optimize the check by noting that M(x) is a step function.
        # However, a full vectorized check is safer.
        # To save time, we can check in batches.
        
        # Note: The evaluator computes M(x) for EVERY integer x in [1, 10K].
        # Our check must verify this.
        
        feasible = True
        for batch_start in range(check_start, check_end + 1, batch_size):
            if time.time() - start_time > deadline - 60:
                return False
            
            batch_end = min(batch_start + batch_size - 1, check_end)
            xs = np.arange(batch_start, batch_end + 1, dtype=np.int64)
            
            # Compute M(x) for each x in batch
            # To avoid huge memory usage, process xs in sub-batches
            sub_batch_size = 5000
            for sb_start in range(0, len(xs), sub_batch_size):
                if time.time() - start_time > deadline - 60:
                    return False
                sb_end = min(sb_start + sub_batch_size, len(xs))
                xs_sub = xs[sb_start:sb_end]
                
                # floor_mat shape: (n_x, m)
                # domain shape: (m,)
                floor_mat = xs_sub[:, None] // domain[None, :]
                M_batch = np.sum(values[None, :] * floor_mat, axis=1)
                
                if np.any(M_batch > 1.0001):
                    return False
        
        return True

    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try increasing K values
    # K=200000 is the max allowed.
    # m for K=200000 is approx 6/pi^2 * 200000 ~ 127000.
    # 10K = 2,000,000.
    # Cost of check: m * 10K = 127000 * 2000000 = 2.54e11 operations.
    # This is too slow for a Python loop, even vectorized?
    # NumPy vectorized multiplication of (5000, 127000) matrix is fast.
    # 5000 * 127000 = 6.35e8 elements.
    # Double precision: 8 bytes * 6.35e8 = 5 GB. This might exceed memory or be slow.
    
    # Let's estimate time for K=200000.
    # If we use smaller sub-batches, e.g., 1000 x 127000 = 1.27e8 elements.
    # 1.27e8 * 8 bytes = 1 GB.
    # Number of batches for 10K=2e6: 2000 batches.
    # Each batch takes ~0.1s? 200s total.
    # This is risky.
    
    # Let's try K=100000 first.
    # m ~ 63000.
    # 10K = 1,000,000.
    # Sub-batch 5000 x 63000 = 3.15e8 elements. 2.5 GB.
    # Batches: 200.
    # Time: ~20s?
    
    # We should try K values in descending order of expected score (larger K is better)
    # but stop if time is running out.
    
    # Actually, the score a = -sum f(k) log(k)/k converges to 1.
    # For K=100000, a is very close to 1.
    # For K=200000, a is slightly closer to 1.
    # The gain is small. The risk of timeout is high.
    
    # Let's try K=100000, then 200000 if time permits.
    
    for K in [100000, 200000]:
        if time.time() - start_time > deadline - 30:
            break

        mu = mobius_sieve(K)
        nonzero_idx = np.where(mu[1:] != 0)[0] + 1
        domain = nonzero_idx.astype(np.int64)
        values = mu[nonzero_idx].astype(np.float64)

        m = len(domain)
        if m < 1:
            continue

        # Compute the objective a = -sum f(k) log(k) / k
        log_k = np.log(domain.astype(np.float64))
        a = -np.sum(values * log_k / domain.astype(np.float64))

        # Check feasibility
        feasible = check_feasibility(domain, values, K)

        if not feasible:
            continue

        if a > best_score:
            best_score = a
            best_domain = domain.copy()
            best_values = values.copy()

    if best_domain is None:
        # Fallback to a small feasible solution
        # We can use K=1000
        K = 1000
        mu = mobius_sieve(K)
        nonzero_idx = np.where(mu[1:] != 0)[0] + 1
        domain = nonzero_idx.astype(np.int64)
        values = mu[nonzero_idx].astype(np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values


# EVOLVE-BLOCK-END
