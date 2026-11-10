# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute Mobius function for 1..K using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
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


def _evaluate(domain, values, K_max=None):
    """
    Apply clipping, S-adjustment, check M(x) constraint, compute score.
    Returns (feasible, score).
    """
    # Clip to [-10, 10]
    vals = np.clip(values, -10.0, 10.0)
    
    # Compute S = sum_k f(k)/k
    S = np.sum(vals / domain.astype(np.float64))
    
    # Adjust f(1)
    if 1 in domain:
        idx1 = np.where(domain == 1)[0][0]
        vals[idx1] -= S
    else:
        # Add 1 to domain with value -S
        domain = np.insert(domain, 0, 1)
        vals = np.insert(vals, 0, -S)
    
    # Check M(x) for x in [1, 10*K_max]
    K = domain.max()
    x_max = 10 * K
    
    g = np.zeros(x_max + 1, dtype=np.float64)
    for i, k in enumerate(domain):
        if k > x_max:
            continue
        g[k::k] += vals[i]
    
    M = np.cumsum(g[1:])  # M(x) for x=1..x_max
    
    if np.max(M) > 1.0001:
        return False, -np.inf
    
    # Compute score a = -sum_k f(k) * log(k) / k
    score = -np.sum(vals * np.log(domain.astype(np.float64)) / domain.astype(np.float64))
    return True, score


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # Try increasing K, using Mobius function values
    for K in range(1, 200001):
        if time.time() > deadline - 30:  # Safety margin
            break
        
        mu = _compute_mobius(K)
        
        # Use mu(k) for k=1..K, but skip k where mu(k)=0 to keep domain small
        non_zero = np.where(mu[1:] != 0)[0] + 1
        if len(non_zero) == 0:
            continue
        
        domain = non_zero.astype(np.int64)
        values = mu[1:][mu[1:] != 0].astype(np.float64)
        
        feasible, score = _evaluate(domain, values, K)
        
        if feasible and score > best_score:
            best_score = score
            best_domain = domain.copy()
            best_values = values.copy()
        
        # If score is already close to 1, we can stop
        if best_score > 0.999:
            break
    
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback to seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


# EVOLVE-BLOCK-END
