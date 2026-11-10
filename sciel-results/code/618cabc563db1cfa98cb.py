# EVOLVE-BLOCK-START
import numpy as np
import time

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
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = start_time + 500.0  # Safety margin before 600s limit

    # We use the Mobius function for k up to a certain limit.
    # To keep the evaluator fast (cost ~ m * 10K), we limit m.
    # Let's try m=1000. 1000 * 2,000,000 = 2e9 ops.
    # If evaluator is vectorized NumPy, this is fast. If not, it might be slow.
    # Let's be conservative with m=500.
    
    K_limit = 500
    
    # Compute Mobius function up to K_limit
    mu = np.ones(K_limit + 1, dtype=np.int64)
    mu[0] = 0
    mu[1] = 1
    
    # Sieve of Eratosthenes style for Mobius
    # mu[n] = 0 if n has a squared prime factor
    # mu[n] = (-1)^k if n has k distinct prime factors
    
    is_prime = np.ones(K_limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # Simple sieve to find primes
    for i in range(2, int(np.sqrt(K_limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, K_limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, K_limit + 1) if is_prime[i]]
    
    # Initialize mu with 1
    # For each prime p, multiply mu[j] by -1 for multiples of p
    # If j is divisible by p^2, set mu[j] = 0
    
    # Efficient way:
    # mu = np.ones(K_limit + 1)
    # for p in primes:
    #   for j in range(p, K_limit+1, p):
    #     mu[j] *= -1
    #     if j % (p*p) == 0:
    #        mu[j] = 0
        
    # Let's use the standard linear sieve logic or simple iteration
    mu = np.ones(K_limit + 1, dtype=np.int64)
    mu[0] = 0
    # 1 is 1
    for p in primes:
        for j in range(p, K_limit + 1, p):
            mu[j] *= -1
        p_sq = p * p
        if p_sq <= K_limit:
            for j in range(p_sq, K_limit + 1, p_sq):
                mu[j] = 0
                
    # Construct domain and values
    # We include all k from 1 to K_limit
    domain = np.arange(1, K_limit + 1, dtype=np.int64)
    values = mu[1:K_limit+1].astype(np.float64)
    
    # Check time
    if time.time() > deadline:
        # Return a smaller, safe fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return domain, values

# EVOLVE-BLOCK-END
