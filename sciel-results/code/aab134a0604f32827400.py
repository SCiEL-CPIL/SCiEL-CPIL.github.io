# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    # We want to approximate the Möbius function on a dense set of square-free integers.
    # The key insight: M(x) = sum_k f(k) * floor(x/k) must be <= 1 for all x in [1, 10K].
    # If we take f(k) = mu(k) for k up to K, then M(x) = sum_{k<=K} mu(k) floor(x/k).
    # For x <= K, this is exactly sum_{d|x, d<=K} 1... no wait.
    # Actually, sum_{k=1}^{floor(x)} mu(k) floor(x/k) = 1 for all x >= 1 (this is the identity sum_{d|n} mu(d) = 1 if n=1 else 0, but here it's different).
    # Let me re-examine: M(x) = sum_k f(k) floor(x/k). If f(k) = mu(k), then M(x) = sum_{k=1}^K mu(k) floor(x/k).
    # For x <= K, this equals sum_{k=1}^x mu(k) floor(x/k) + sum_{k=x+1}^K mu(k) * 0 = sum_{k=1}^x mu(k) floor(x/k).
    # But the classic identity is sum_{d|n} mu(d) = [n=1]. This is not the same as sum_{k<=n} mu(k) floor(n/k).
    # Actually, sum_{k=1}^n mu(k) floor(n/k) = sum_{k=1}^n mu(k) * (number of multiples of k up to n) = sum_{m=1}^n sum_{k|m} mu(k) = sum_{m=1}^n [m=1] = 1.
    # So for x <= K, M(x) = 1 exactly. For x > K, M(x) = sum_{k=1}^K mu(k) floor(x/k), which is not necessarily 1.
    
    # The constraint is M(x) <= 1.0001 for all x in [1, 10K].
    # For x in [1, K], M(x) = 1 exactly.
    # For x in [K+1, 10K], M(x) = sum_{k=1}^K mu(k) floor(x/k). This can exceed 1.
    
    # To ensure feasibility, we need to be careful. One approach: use a smaller K where the tail behavior is well-controlled, or adjust values.
    # However, the task says the optimum is approached by mu(k), and the seed has a small domain.
    # Let's try to use mu(k) for k up to some K, but we need to check feasibility.
    
    # A safer approach: use only square-free k with mu(k) = +/-1, and choose K such that the maximum of M(x) for x in [K+1, 10K] is <= 1.
    # In practice, for moderate K, the fluctuations of M(x) for x > K are small. But to be safe, let's compute M(x) and verify.
    
    # Given the 600s time limit and K <= 200000, we need an efficient method.
    # Computing M(x) for all x in [1, 10K] naively is O(m * 10K) which is too slow for large m.
    # But we can use the fact that floor(x/k) changes only O(sqrt(x)) times, or use a sieve-like approach.
    
    # Actually, the evaluator computes M(x) for every integer x in [1, 10K]. The cost is about m * 10K.
    # If m is large (e.g., ~150000 square-free numbers up to 200000), then m * 10K is huge.
    # But wait, the problem says "cost about m * 10K" — this is the evaluator's cost, not ours.
    # Our job is just to return (domain, values). The evaluator will check feasibility.
    
    # So we need to choose f such that M(x) <= 1.0001 for all x in [1, 10K].
    # If we set f(k) = mu(k) for k in some set, we need to verify this.
    
    # Strategy: Use mu(k) for all square-free k up to K, but we need to handle the normalization.
    # The evaluator sets f(1) = f(1) - S where S = sum f(k)/k, so that sum f(k)/k = 0.
    # If we set f(k) = mu(k) for all k, then sum_{k=1}^K mu(k)/k is some value, and f(1) gets adjusted.
    
    # But the key issue is M(x) <= 1. For x <= K, M(x) = 1 (if we use all mu(k) for k <= K).
    # For x > K, M(x) = sum_{k=1}^K mu(k) floor(x/k). We need this <= 1.0001.
    
    # Let's try K = 1000 first and see if it's feasible. We can compute M(x) for x in [1, 10000] to check.
    # But doing this in the function is expensive. Instead, let's use a known result: for the Möbius function,
    # the maximum of |sum_{k=1}^K mu(k) floor(x/k) - 1| for x in [K, 10K] is small for moderate K.
    
    # Alternatively, we can use a truncated version: only include k where mu(k) != 0, i.e., square-free k.
    # And we can scale or adjust to ensure feasibility.
    
    # Let's try a practical approach: use mu(k) for square-free k up to K, with K chosen to balance
    # the objective a = -sum f(k) log(k)/k and feasibility.
    
    # Compute mu(k) for k up to K using a sieve.
    K = 50000  # Choose a moderate K
    
    # Sieve for Möbius function
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    # Select square-free k (mu(k) != 0)
    square_free = np.where(mu[1:] != 0)[0] + 1
    m = len(square_free)
    
    # Set f(k) = mu(k) for these k
    values = mu[square_free].astype(np.float64)
    domain = square_free.astype(np.int64)
    
    # Now we need to check feasibility: M(x) <= 1.0001 for all x in [1, 10*K]
    # But computing this naively is O(m * 10K) which is too slow.
    # We can use a more efficient method: for each k, add mu(k) to M(x) for x in [k, 10K] in chunks.
    # But even this is O(m * 10K / k) total, which is O(10K * log(K)) ≈ 500000 * 11 ≈ 5.5M, which is feasible.
    
    # Let's compute M(x) for x in [1, 10*K] using an efficient method.
    max_x = 10 * K
    M = np.zeros(max_x + 1, dtype=np.float64)
    
    # For each k in domain, add mu(k) * floor(x/k) to M(x) for x in [k, max_x]
    # We can do this by iterating over k and adding to ranges where floor(x/k) is constant.
    for k, val in zip(domain, values):
        # For x in [k, max_x], floor(x/k) = q for x in [q*k, (q+1)*k - 1]
        q = 1
        while q * k <= max_x:
            start = q * k
            end = min((q + 1) * k - 1, max_x)
            M[start:end+1] += val * q
            q += 1
    
    # Check feasibility
    if np.any(M[1:max_x+1] > 1.0001):
        # Not feasible, fall back to a smaller K
        K = 10000
        mu = np.ones(K + 1, dtype=np.int64)
        is_prime = np.ones(K + 1, dtype=bool)
        primes = []
        for i in range(2, K + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > K:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        
        square_free = np.where(mu[1:] != 0)[0] + 1
        m = len(square_free)
        values = mu[square_free].astype(np.float64)
        domain = square_free.astype(np.int64)
        
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        for k, val in zip(domain, values):
            q = 1
            while q * k <= max_x:
                start = q * k
                end = min((q + 1) * k - 1, max_x)
                M[start:end+1] += val * q
                q += 1
        
        if np.any(M[1:max_x+1] > 1.0001):
            # Fall back to the seed
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
    
    return domain, values


# EVOLVE-BLOCK-END
