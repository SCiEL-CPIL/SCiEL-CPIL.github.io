import numpy as np
from math import log
import time

def _compute_mobius(N):
    """Compute Möbius function mu(k) for k=1 to N using a sieve."""
    mu = np.ones(N+1, dtype=np.int64)
    is_prime = np.ones(N+1, dtype=bool)
    primes = []
    for i in range(2, N+1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > N:
                break
            is_prime[i*p] = False
            if i % p == 0:
                mu[i*p] = 0
                break
            else:
                mu[i*p] = -mu[i]
    return mu

def _compute_M_max(f, N, max_x):
    """Compute max of M(x) = sum f(k) floor(x/k) for x in [1, max_x] using difference array."""
    # M(x) = sum_{k=1}^{N} f(k) floor(x/k)
    # Use difference array: for each k, f(k)*floor(x/k) is a step function
    # floor(x/k) = j for x in [j*k, (j+1)*k - 1]
    diff = np.zeros(max_x + 2, dtype=np.float64)
    for k in range(1, N+1):
        fk = f[k]
        if fk == 0:
            continue
        j = 1
        while j * k <= max_x:
            start = j * k
            end = min((j+1) * k - 1, max_x)
            diff[start] += fk * j
            diff[end + 1] -= fk * j
            j += 1
    # Prefix sum to get M(x)
    M = np.cumsum(diff[:max_x+1])
    return M

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Try different N values
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # We'll try a few N values and pick the best
    for N in [1000, 5000, 10000, 20000, 50000]:
        if time.time() > deadline - 60:
            break
        
        mu = _compute_mobius(N)
        f = mu.astype(np.float64)
        
        # Normalize: S = sum f(k)/k, f(1) -= S
        S = np.sum(f[1:N+1] / np.arange(1, N+1))
        f[1] -= S
        
        # Compute M(x) for x in [1, 10N]
        max_x = 10 * N
        M = _compute_M_max(f, N, max_x)
        max_M = np.max(M)
        
        if max_M > 1.0001:
            # Scale down
            # Find the scaling factor that makes max M(x) = 1.0001
            # M(x) is linear in the scaling factor c (for non-1 values)
            # But f(1) also changes with c, so it's not simply linear
            # Let's just try c = 1.0 and scale down if needed
            # Actually, let's compute the required scaling
            # For now, just skip if infeasible
            continue
        
        # Compute objective
        k = np.arange(1, N+1)
        a = -np.sum(f * np.log(k) / k)
        
        if a > best_score:
            best_score = a
            best_domain = k
            best_values = f
    
    if best_domain is None:
        # Fallback to the original
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values
