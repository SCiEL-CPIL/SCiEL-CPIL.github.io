# EVOLVE-BLOCK-START
import numpy as np
import time


def _mobius(n):
    """Compute Mobius function for 1..n using sieve."""
    mu = np.ones(n + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
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


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Uses a truncated Mobius function to approximate the ideal solution.
    """
    start_time = time.time()
    deadline = start_time + 550  # Safety margin before 600s limit
    
    # Start with a larger set of Mobius values
    K = 500
    mu = _mobius(K)
    
    # Include all k where mu(k) != 0 (square-free numbers)
    mask = mu != 0
    domain = np.where(mask)[0]
    values = mu[domain].astype(np.float64)
    
    # Ensure 1 is in domain
    if domain[0] != 1:
        domain = np.concatenate([[1], domain])
        values = np.concatenate([[1.0], values])
    
    # Compute the f(1) adjustment: S = sum_k f(k)/k
    S = np.sum(values / domain.astype(np.float64))
    # After adjustment, f(1) becomes f(1) - S
    # We want to maximize a = -sum_k f(k)*log(k)/k
    # But we need to check feasibility: M(x) <= 1.0001
    
    # For the Mobius function, M(x) = sum mu(k)*floor(x/k) = 1 for all x >= 1
    # So it should be feasible. Let's verify with a sample.
    
    # Check feasibility on a subset of x values
    K_max = domain[-1]
    x_max = min(10 * K_max, 5000)  # Check up to 10K but cap for speed
    
    # Compute M(x) for x = 1 to x_max
    # M(x) = sum_k f(k) * floor(x/k)
    # For efficiency, use the fact that floor(x/k) changes at most O(sqrt(x)) times
    
    # Simple check for small x_max
    if x_max <= 10000:
        x_vals = np.arange(1, x_max + 1)
        # M(x) = sum_k f(k) * floor(x/k)
        # Vectorized: for each k, add f(k) * floor(x/k) for all x
        M = np.zeros(x_max)
        for k, f in zip(domain, values):
            M += f * (x_vals // k)
        
        if np.max(M) > 1.0001:
            # Not feasible, fall back to smaller set
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
    
    # Compute the objective a = -sum_k f(k)*log(k)/k
    # Note: the evaluator will adjust f(1) by subtracting S
    # So we need to account for that in our calculation
    log_k = np.log(domain.astype(np.float64))
    a_raw = -np.sum(values * log_k / domain.astype(np.float64))
    
    # After f(1) adjustment: f(1) -> f(1) - S
    # The new sum is a_new = a_raw + S * log(1)/1 = a_raw (since log(1) = 0)
    # Wait, that's not right. Let me recalculate.
    # a = -sum_k f(k)*log(k)/k
    # After adjustment: f(1) becomes f(1) - S
    # a_new = -((f(1)-S)*log(1)/1 + sum_{k>1} f(k)*log(k)/k)
    #        = -(0 + sum_{k>1} f(k)*log(k)/k)  [since log(1) = 0]
    #        = a_raw  [same as before since log(1) = 0]
    
    # So the objective doesn't change with the f(1) adjustment!
    # The constraint is just that M(x) <= 1.0001
    
    return domain, values


# EVOLVE-BLOCK-END
