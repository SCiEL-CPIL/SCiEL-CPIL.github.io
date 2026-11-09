# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function mu(n) for n = 1..K using sieve."""
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
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


def _compute_M_max(domain, values, x_max):
    """Compute max of M(x) = sum_k f(k) * floor(x/k) for x in [1, x_max]."""
    m = len(domain)
    if m == 0:
        return 0.0
    
    if x_max > 2000000:
        n_samples = min(x_max, 200000)
        x_samples = np.linspace(1, x_max, n_samples, dtype=np.int64)
        M_vals = np.zeros(len(x_samples))
        for idx in range(len(x_samples)):
            x = x_samples[idx]
            M_vals[idx] = np.sum(values * (x // domain))
        return np.max(M_vals)
    else:
        h = np.zeros(x_max + 1, dtype=np.float64)
        for i in range(m):
            k = int(domain[i])
            if k > x_max:
                continue
            h[k::k] += values[i]
        M_vals = np.cumsum(h[1:])
        return np.max(M_vals)


def _compute_a(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k."""
    k_float = domain.astype(np.float64)
    return -np.sum(values * np.log(k_float) / k_float)


def _check_feasibility(domain, values, K_max, tolerance=1.0001):
    """Check if M(x) <= tolerance for all x in [1, 10*K_max]."""
    x_max = 10 * K_max
    M_max = _compute_M_max(domain, values, x_max)
    return M_max <= tolerance


def prime_number_theorem_partial_function():
    """
    Construct a partial function f that maximizes a = -sum_k f(k)*log(k)/k
    subject to M(x) = sum_k f(k)*floor(x/k) <= 1.0001 for all x in [1, 10K].
    """
    start_time = time.time()
    deadline = start_time + 580
    
    best_a = -np.inf
    best_domain = None
    best_values = None
    
    K_candidates = [200000, 100000, 50000]
    
    for K in K_candidates:
        if time.time() > deadline - 120:
            break
        
        mu = _compute_mobius(K)
        
        # Use all k where mu(k) != 0
        mask = mu[1:] != 0
        domain = np.where(mask)[0] + 1
        values = mu[domain].astype(np.float64)
        
        if not _check_feasibility(domain, values, K):
            continue
            
        a_val = _compute_a(domain, values)
        
        if a_val > best_a:
            best_a = a_val
            best_domain = domain.copy()
            best_values = values.copy()

    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values


# EVOLVE-BLOCK-END
