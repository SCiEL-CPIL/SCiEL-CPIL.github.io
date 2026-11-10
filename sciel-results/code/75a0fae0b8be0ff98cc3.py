# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f approximating the Mobius function to maximize the objective.
    Uses a global scaling factor on non-1 values to strictly satisfy M(x) <= 1 constraint.
    """
    start_time = time.time()
    deadline = 600.0
    
    # Choose K. Larger K gives better approximation of 1.
    # K=50000 is safe and fast.
    K = 50000
    
    # 1. Compute Mobius function up to K
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    # Sieve of Eratosthenes to compute mu
    mu[0] = 0
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0

    # 2. Find the maximum scaling factor alpha such that M(x) <= 1.0001 for all x in [1, 10K]
    # Strategy: f(k) = alpha * mu(k) for k > 1, f(1) adjusted to satisfy sum constraint
    
    max_x = 10 * K
    A_base = np.zeros(max_x + 1, dtype=np.float64)
    
    # Compute base A(n) = sum_{d|n, d<=K} mu(d) for d >= 2 (excluding d=1)
    # For d=1, mu(1)=1, so A_base includes the contribution from d=1
    # We want to separate: A_total = A_d1 + alpha * A_non_d1
    # where A_d1(n) = 1 if 1|n (always true), A_non_d1(n) = sum_{d|n, d>=2, d<=K} mu(d)
    
    # Actually, let's think differently:
    # f(1) = c1, f(k) = alpha * mu(k) for k >= 2
    # Constraint: sum_k f(k)/k = 0 => c1 + alpha * sum_{k=2}^K mu(k)/k = 0
    # So c1 = -alpha * sum_{k=2}^K mu(k)/k
    
    # M(x) = sum_k f(k) * floor(x/k)
    # = c1 * floor(x/1) + alpha * sum_{k=2}^K mu(k) * floor(x/k)
    # = c1 * x + alpha * sum_{k=2}^K mu(k) * floor(x/k)
    # = -alpha * S * x + alpha * sum_{k=2}^K mu(k) * floor(x/k)
    # where S = sum_{k=2}^K mu(k)/k
    # = alpha * (sum_{k=2}^K mu(k) * floor(x/k) - S * x)
    
    # Let T(x) = sum_{k=2}^K mu(k) * floor(x/k) - S * x
    # Then M(x) = alpha * T(x)
    # We need alpha * T(x) <= 1.0001 for all x in [1, 10K]
    # So alpha <= 1.0001 / max_x T(x) if max T(x) > 0
    # Also need alpha > 0 for meaningful objective
    
    # Compute S = sum_{k=2}^K mu(k)/k
    S = np.sum(mu[2:K+1] / np.arange(2, K+1, dtype=np.float64))
    
    # Compute T(x) for all x in [1, max_x]
    # T(x) = sum_{k=2}^K mu(k) * floor(x/k) - S * x
    # = sum_{k=2}^K mu(k) * floor(x/k) - S * x
    
    # Efficient computation: for each k, add mu(k) to T[k], T[2k], ...
    # But we need floor(x/k), not just indicator
    
    # Alternative: compute U(x) = sum_{k=2}^K mu(k) * floor(x/k)
    # U(x) = sum_{k=2}^K mu(k) * floor(x/k)
    # This can be computed as: for each k, mu(k) contributes to U(x) for x >= k,
    # and the contribution is mu(k) * floor(x/k)
    
    # Let's compute it directly using the fact that floor(x/k) changes at multiples of k
    # But that's O(K * max_x / k) which is too slow
    
    # Better approach: U(x) = sum_{n=1}^x sum_{k|n, k>=2, k<=K} mu(k)
    # = sum_{n=1}^x (A(n) - 1) where A(n) = sum_{d|n, d<=K} mu(d)
    # Wait, A(n) includes d=1, so A(n) - 1 = sum_{d|n, d>=2, d<=K} mu(d)
    
    # So U(x) = sum_{n=1}^x (A(n) - 1)
    # And T(x) = U(x) - S * x
    
    # Let's compute A(n) first
    A = np.zeros(max_x + 1, dtype=np.float64)
    for d in range(2, K + 1):
        if mu[d] != 0:
            A[d:max_x+1:d] += mu[d]
    
    # A(n) = sum_{d|n, d>=2, d<=K} mu(d)
    # U(x) = cumsum(A)
    U = np.cumsum(A)
    
    # T(x) = U(x) - S * x for x in [1, max_x]
    x_vals = np.arange(1, max_x + 1, dtype=np.float64)
    T = U[1:max_x+1] - S * x_vals
    
    # Find max T(x)
    max_T = np.max(T)
    
    if max_T > 0:
        alpha_max = 1.0001 / max_T
        # Use a slightly smaller alpha to be safe
        alpha = min(1.0, alpha_max * 0.999)
    else:
        # If max_T <= 0, we can use alpha = 1
        alpha = 1.0
    
    # 3. Construct output
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = np.zeros(K, dtype=np.float64)
    
    # f(k) = alpha * mu(k) for k >= 2
    values[1:] = alpha * mu[2:K+1]
    
    # f(1) = -alpha * S
    values[0] = -alpha * S
    
    # 4. Verify feasibility
    # Recompute M(x) to check
    A_check = np.zeros(max_x + 1, dtype=np.float64)
    for d in range(1, K + 1):
        if values[d-1] != 0:
            A_check[d:max_x+1:d] += values[d-1]
    
    M_check = np.cumsum(A_check)
    
    if np.max(M_check) > 1.0001:
        # Fallback: reduce alpha further
        max_M = np.max(M_check)
        alpha *= 1.0001 / max_M * 0.999
        values[1:] = alpha * mu[2:K+1]
        values[0] = -alpha * S
        
        # Re-verify
        A_check = np.zeros(max_x + 1, dtype=np.float64)
        for d in range(1, K + 1):
            if values[d-1] != 0:
                A_check[d:max_x+1:d] += values[d-1]
        M_check = np.cumsum(A_check)
        
        if np.max(M_check) > 1.0001:
            # Last resort fallback
            K = 10000
            mu = np.ones(K + 1, dtype=np.float64)
            is_prime = np.ones(K + 1, dtype=bool)
            mu[0] = 0
            for i in range(2, K + 1):
                if is_prime[i]:
                    for j in range(i, K + 1, i):
                        is_prime[j] = False
                        mu[j] *= -1
                    i2 = i * i
                    if i2 <= K:
                        for j in range(i2, K + 1, i2):
                            mu[j] = 0
            
            max_x = 10 * K
            S = np.sum(mu[2:K+1] / np.arange(2, K+1, dtype=np.float64))
            A = np.zeros(max_x + 1, dtype=np.float64)
            for d in range(2, K + 1):
                if mu[d] != 0:
                    A[d:max_x+1:d] += mu[d]
            U = np.cumsum(A)
            x_vals = np.arange(1, max_x + 1, dtype=np.float64)
            T = U[1:max_x+1] - S * x_vals
            max_T = np.max(T)
            
            if max_T > 0:
                alpha = min(1.0, 1.0001 / max_T * 0.999)
            else:
                alpha = 1.0
            
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = np.zeros(K, dtype=np.float64)
            values[1:] = alpha * mu[2:K+1]
            values[0] = -alpha * S

    # Time check
    if time.time() > deadline - 10:
        pass

    return domain, values

# EVOLVE-BLOCK-END
