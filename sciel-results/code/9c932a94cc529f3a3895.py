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
    
    def compute_mobius(K):
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
        return mu

    def check_feasibility(domain, values, K):
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        for k, val in zip(domain, values):
            q = 1
            while q * k <= max_x:
                start = q * k
                end = min((q + 1) * k - 1, max_x)
                M[start:end+1] += val * q
                q += 1
        
        # Calculate S = sum f(k)/k
        S = np.sum(values / domain.astype(np.float64))
        
        # The evaluator sets f(1) = f(1) - S.
        # So the new M(x) is M_old(x) - S * floor(x/1) = M_old(x) - S * x.
        M_final = M - S * np.arange(max_x + 1, dtype=np.float64)
        
        if np.any(M_final[1:max_x+1] > 1.0001):
            return False
        return True

    max_K = 20000
    # We want the largest K that is feasible.
    # Start from max_K and go down.
    # To save time, we can check a few candidates.
    # The feasibility depends on S and fluctuations.
    
    best_domain = None
    best_values = None
    
    # Try a range of K values.
    # S(K) oscillates. We want S(K) > 0 to help with M(x) <= 1 for large x.
    # Let's just test K from max_K downwards until we find a feasible one.
    # But checking every K is slow.
    # Let's test K = 20000, 15000, 10000, 5000, 2000, 1000.
    
    candidates = [20000, 15000, 10000, 5000, 2000, 1000, 500, 100]
    
    for K in candidates:
        mu = compute_mobius(K)
        square_free = np.where(mu[1:] != 0)[0] + 1
        if len(square_free) == 0:
            continue
        values = mu[square_free].astype(np.float64)
        domain = square_free.astype(np.int64)
        
        if check_feasibility(domain, values, K):
            best_domain = domain
            best_values = values
            break
            
    if best_domain is None:
        # Fallback to a very small K
        K = 10
        mu = compute_mobius(K)
        square_free = np.where(mu[1:] != 0)[0] + 1
        values = mu[square_free].astype(np.float64)
        domain = square_free.astype(np.int64)
        best_domain = domain
        best_values = values

    return best_domain, best_values


# EVOLVE-BLOCK-END
