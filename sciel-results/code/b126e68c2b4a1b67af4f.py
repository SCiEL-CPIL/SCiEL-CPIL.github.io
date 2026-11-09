import numpy as np
import time

def compute_mobius(n):
    mu = np.ones(n + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(n + 1, dtype=bool)
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

def check_feasibility(mu, K):
    """Check if M(x) <= 1.0001 for all x in [1, 10K]."""
    max_x = 10 * K
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for k in range(1, K + 1):
        if mu[k] == 0:
            continue
        mk = float(mu[k])
        j_max = max_x // k
        if j_max == 0:
            continue
        j = np.arange(1, j_max + 1, dtype=np.int64)
        starts = j * k
        ends = np.minimum((j + 1) * k - 1, max_x)
        diff[starts] += mk * j
        diff[ends + 1] -= mk * j
    
    M = np.cumsum(diff)
    M = M[1:max_x + 1]
    return np.max(M) <= 1.0001, M

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580
    
    # Compute Möbius up to max K
    max_K = 200000
    mu = compute_mobius(max_K)
    
    # Try largest K first, reduce if infeasible
    for K in [200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000]:
        if time.time() > deadline - 10:
            break
        feasible, M = check_feasibility(mu, K)
        if feasible:
            mask = mu[1:K + 1] != 0
            domain = np.nonzero(mask)[0] + 1
            values = mu[1:K + 1][mask].astype(np.float64)
            return domain, values
    
    # Fallback: use the seed program's result
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values
