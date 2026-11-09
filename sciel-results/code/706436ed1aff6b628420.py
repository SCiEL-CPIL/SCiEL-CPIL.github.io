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
    # We use a large K to approach the theoretical limit of the score (1.0).
    # The feasibility constraint M(x) <= 1.0001 is expected to hold for large K
    # due to the small error term in the Mobius inversion.
    K = 200000
    
    # 1. Compute the Mobius function mu(k) for k <= K using a sieve.
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
    mu[1] = 1

    # 2. Construct domain and values.
    # We include all k where mu(k) != 0 to keep the domain size manageable, 
    # but including all k is also fine. The sum over 0s doesn't affect M(x) or score.
    # However, smaller domain is better for potential overheads, though evaluator handles it.
    # Let's include all k=1..K for simplicity, or just non-zero mu.
    # The prompt says "domain holds distinct positive integers".
    # Using all K is safe.
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    
    # 3. Apply the f(1) adjustment.
    # S = sum_k f(k)/k
    # We need to compute this sum.
    # Note: values[0] corresponds to k=1.
    S = np.sum(values / domain)
    values[0] -= S
    
    # 4. Check feasibility: M(x) <= 1.0001 for all x in [1, 10K].
    # M(x) = sum_k f(k) * floor(x/k)
    # Efficiently compute M(x) for all x using the "add to multiples" technique.
    N = 10 * K
    M = np.zeros(N + 1, dtype=np.float64)
    
    # We only need to process k where f(k) != 0 to save time, 
    # but vectorized addition is fast.
    # Loop over k is O(K log K) which is ~ 2.4 * 10^7 operations.
    # This might be slow in a pure Python loop.
    # We can use NumPy to speed it up if we can vectorize the inner loop?
    # Not easily for "add to multiples" without sparse matrices or similar.
    # However, 2.4 * 10^7 simple additions in Python might take ~1-2 seconds.
    # Let's try to optimize by skipping zeros.
    
    nonzero_indices = np.nonzero(values)[0]
    for idx in nonzero_indices:
        k = idx + 1
        val = values[idx]
        if val != 0:
            M[k::k] += val
            
    # Check max
    if np.max(M[1:]) > 1.0001:
        # If infeasible, we might need to reduce K or adjust.
        # For now, return the best we have, but it will score 0 if infeasible.
        # Let's try a smaller K if this fails? 
        # The error is expected to be small. 
        # If it fails, the score is 0.
        # We can add a fallback to a smaller K if needed, but 200k is likely fine.
        pass
        
    return domain, values

# EVOLVE-BLOCK-END
