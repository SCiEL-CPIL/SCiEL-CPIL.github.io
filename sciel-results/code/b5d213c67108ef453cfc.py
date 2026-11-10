import numpy as np
from math import log
import time

def _compute_mobius(N):
    """Compute Möbius function mu(k) for k=1 to N using a linear sieve."""
    mu = np.ones(N + 1, dtype=np.int8)
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > N:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    mu[1] = 1
    return mu

def _compute_M_max(f, N, max_x):
    """Compute max of M(x) = sum f(k) floor(x/k) for x in [1, max_x].
    Returns the maximum value of M(x).
    Uses the fact that M(x) is piecewise linear in x with breakpoints at multiples of k.
    For efficiency, we compute M at all x using a difference array approach.
    """
    # M(x) = sum_{k=1}^{N} f(k) * floor(x/k)
    # For a fixed k, floor(x/k) is constant on intervals [j*k, (j+1)*k - 1]
    # We can build M(x) using a difference array
    # M(x) = sum over k of f(k) * floor(x/k)
    # Let's compute M at all x from 1 to max_x
    # M(x) = sum_{k=1}^{min(N,x)} f(k) * floor(x/k)
    # This is O(N * max_x / k) per x which is too slow.
    # Better: use the fact that M(x) = sum_{k} f(k) * floor(x/k)
    # We can compute this as: for each k, f(k) contributes to M(x) for x in [k, max_x]
    # as f(k) * floor(x/k). floor(x/k) = j for x in [j*k, (j+1)*k - 1].
    # So we can use a difference array:
    # diff[j*k] += f(k) * j
    # diff[(j+1)*k] -= f(k) * j
    # Then M = cumsum(diff)
    
    diff = np.zeros(max_x + 2, dtype=np.float64)
    for k in range(1, N + 1):
        fk = f[k]
        if fk == 0.0:
            continue
        j = 1
        while j * k <= max_x:
            start = j * k
            end = min((j + 1) * k - 1, max_x)
            diff[start] += fk * j
            diff[end + 1] -= fk * j
            j += 1
    M = np.cumsum(diff[:max_x + 1])
    return np.max(M[1:])  # x from 1 to max_x

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580.0
    
    fallback_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    fallback_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # Strategy: use f(k) = c * mu(k) for k >= 2, and determine f(1) via normalization.
    # Normalization: sum_k f(k)/k = 0 after clipping (but clipping is applied by evaluator,
    # so we need to account for it). Actually, the evaluator clips values to [-10, 10],
    # then computes S = sum f(k)/k over clipped values, then f(1) -= S.
    # Since mu(k) is in {-1, 0, 1} and c will be small, clipping likely won't affect
    # the mu values. But f(1) could be large. Let's check: f(1) = -c/2 * sum_{k=2}^N mu(k)/k.
    # sum_{k=2}^N mu(k)/k converges to -1 (since sum_{k=1}^inf mu(k)/k = 0), so
    # sum_{k=2}^N mu(k)/k ≈ -1, thus f(1) ≈ c/2. If c is small, f(1) is small.
    
    # Let's try different N values and find optimal c for each.
    for N in [200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000]:
        if time.time() > deadline - 30:
            break
        
        mu = _compute_mobius(N)
        k_arr = np.arange(1, N + 1)
        
        # For a given c, f(k) = c * mu(k) for k >= 2, f(1) = ?
        # After clipping to [-10, 10]: if |c * mu(k)| <= 10 for all k (which is true for small c),
        # then clipped f(k) = c * mu(k) for k >= 2, and clipped f(1) = f(1) (if |f(1)| <= 10).
        # S = sum_k clipped_f(k)/k = f(1)/1 + c * sum_{k=2}^N mu(k)/k
        # f(1) = -S => f(1) = -f(1) - c * sum_{k=2}^N mu(k)/k => 2*f(1) = -c * sum_{k=2}^N mu(k)/k
        # f(1) = -c/2 * sum_{k=2}^N mu(k)/k
        
        sum_mu_over_k = np.sum(mu[2:] / k_arr[1:])
        # f(1) = -c/2 * sum_mu_over_k
        
        # M(x) = sum_k f(k) * floor(x/k)
        # = f(1) * x + c * sum_{k=2}^N mu(k) * floor(x/k)
        # = -c/2 * sum_mu_over_k * x + c * sum_{k=2}^N mu(k) * floor(x/k)
        # = c * [sum_{k=2}^N mu(k) * floor(x/k) - sum_mu_over_k * x / 2]
        
        # We need M(x) <= 1.0001 for all x in [1, 10*N]
        # M(x) = c * g(x) where g(x) = sum_{k=2}^N mu(k) * floor(x/k) - sum_mu_over_k * x / 2
        # If g(x) can be positive, we need c * max(g(x)) <= 1.0001
        # If g(x) is always negative, any c works (but we want to maximize objective)
        
        # Objective: a = -sum_k f(k) * log(k) / k
        # = -f(1) * log(1) / 1 - sum_{k=2}^N c * mu(k) * log(k) / k
        # = -c * sum_{k=2}^N mu(k) * log(k) / k  (since log(1) = 0)
        
        obj_coeff = -np.sum(mu[2:] * np.log(k_arr[1:]) / k_arr[1:])
        # a = c * obj_coeff
        
        # We want to maximize c * obj_coeff subject to c * max_x g(x) <= 1.0001
        # and c >= 0 (if obj_coeff > 0, we want c as large as possible)
        
        # Compute max of g(x) for x in [1, 10*N]
        # g(x) = sum_{k=2}^N mu(k) * floor(x/k) - sum_mu_over_k * x / 2
        
        # Let's compute g(x) for all x using the same difference array technique
        max_x = 10 * N
        
        # Compute h(x) = sum_{k=2}^N mu(k) * floor(x/k)
        diff = np.zeros(max_x + 2, dtype=np.float64)
        for k in range(2, N + 1):
            mk = mu[k]
            if mk == 0:
                continue
            j = 1
            while j * k <= max_x:
                start = j * k
                end = min((j + 1) * k - 1, max_x)
                diff[start] += mk * j
                diff[end + 1] -= mk * j
                j += 1
        h = np.cumsum(diff[:max_x + 1])
        
        # g(x) = h(x) - sum_mu_over_k * x / 2
        x_arr = np.arange(1, max_x + 1, dtype=np.float64)
        g = h - sum_mu_over_k * x_arr / 2.0
        max_g = np.max(g)
        
        if max_g <= 0:
            # g(x) is always non-positive, so M(x) = c * g(x) <= 0 for c >= 0
            # We can use any c, but we're limited by clipping of f(1)
            # f(1) = -c/2 * sum_mu_over_k, and |f(1)| <= 10 (clipping)
            # So c <= 20 / |sum_mu_over_k| if sum_mu_over_k != 0
            if abs(sum_mu_over_k) > 1e-15:
                c_max = 20.0 / abs(sum_mu_over_k)
            else:
                c_max = 10.0  # arbitrary cap
            c = c_max
        else:
            c_max = 1.0001 / max_g
            c = c_max
        
        # Compute the actual values
        f = np.zeros(N + 1, dtype=np.float64)
        f[1:] = c * mu[1:]
        f[1] = -c / 2.0 * sum_mu_over_k
        
        # Check clipping
        f_clipped = np.clip(f[1:], -10.0, 10.0)
        
        # Compute S over clipped values
        S = np.sum(f_clipped / k_arr)
        f_clipped[0] -= S  # f(1) adjusted
        
        # Verify M(x) <= 1.0001
        # Recompute M with final f_clipped
        M_max = _compute_M_max(f_clipped, N, max_x)
        
        if M_max <= 1.0001:
            a = -np.sum(f_clipped * np.log(k_arr) / k_arr)
            if a > best_score:
                best_score = a
                best_domain = k_arr.copy()
                best_values = f_clipped.copy()
    
    if best_domain is None:
        return fallback_domain, fallback_values
    
    return best_domain, best_values
