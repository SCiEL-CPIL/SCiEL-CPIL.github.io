# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    # Use 500s to be safe
    deadline = start_time + 500
    
    K_search_max = 5000
    
    # Compute Mobius
    mu = np.zeros(K_search_max + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(K_search_max + 1, dtype=bool)
    primes = []
    for i in range(2, K_search_max + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
            for j in primes:
                if i * j > K_search_max:
                    break
                is_prime[i * j] = False
                if i % j == 0:
                    mu[i * j] = 0
                    break
                else:
                    mu[i * j] = -mu[i]
    
    k_arr = np.arange(1, K_search_max + 1)
    log_k = np.log(k_arr)
    
    max_x_global = 10 * K_search_max
    G = np.zeros(max_x_global + 1, dtype=np.float64)
    
    H = 0.0
    S = 0.0
    
    best_K = 1
    best_score = -np.inf
    best_c = 0.0
    
    # Precompute mu array for fast access
    mu_arr = mu[1:]
    
    for K in range(1, K_search_max + 1):
        if time.time() > deadline:
            break
            
        # 1. Fill G for new x range if K > 1
        if K > 1:
            start_x = 10 * (K - 1) + 1
            end_x = 10 * K
            # x values
            new_xs = np.arange(start_x, end_x + 1)
            # Calculate G_K-1(x) for these x
            # G_K-1(x) = sum_{k=1}^{K-1} mu[k] floor(x/k)
            # Vectorized: 
            # We can sum over k
            # But K is up to 5000. 10 points. 5000*10 = 50000 ops per K.
            # Total 5000 * 50000 = 2.5e8. Might be slow in Python loop.
            # Let's try to optimize.
            # Actually, we can just compute G for all x in [1, 10K] from scratch? No.
            # Let's stick to the incremental update but be careful with the new points.
            
            # For the 10 new points, we need sum_{k=1}^{K-1} mu[k] floor(x/k)
            # We can do this by looping k from 1 to K-1 and adding mu[k] * floor(x/k)
            # This is O(K) per new point. 10 points -> 10K.
            # Total O(K^2).
            
            # To speed up, we can just compute G for the new points using the existing G array?
            # No, G array only has values up to 10(K-1).
            
            # Let's just compute it.
            for idx, x in enumerate(new_xs):
                val = 0.0
                # Sum over k=1 to K-1
                # floor(x/k) is constant for ranges of k
                # But K is small enough?
                # For K=5000, 5000 iterations per point.
                # 10 points * 5000 = 50000 ops.
                # Python loop is slow.
                # Use numpy?
                # ks = np.arange(1, K)
                # val = np.sum(mu_arr[:K-1] * (x // ks))
                # This creates arrays of size K.
                # 10 * 5000 = 50000 elements.
                # This is fast in numpy.
                ks = np.arange(1, K)
                val = np.dot(mu_arr[:K-1], x // ks)
                G[x] = val
        
        # 2. Update G for all x in [1, 10K] with term for k=K
        mu_k = mu[K]
        if mu_k != 0:
            x_range = np.arange(1, 10*K + 1)
            G[1:10*K+1] += mu_k * (x_range // K)
            
        # 3. Update H and S
        H += mu_k / K
        S += mu_k * log_k[K-1] / K
        
        # 4. Compute V_max
        x_range = np.arange(1, 10*K + 1)
        V_vals = G[1:10*K+1] - x_range * H
        V_max = np.max(V_vals)
        
        # 5. Determine c and score
        if V_max <= 0:
            c = 10.0
        else:
            c = 1.0001 / V_max
            if c > 10.0:
                c = 10.0
        
        score = -c * S
        
        if score > best_score:
            best_score = score
            best_K = K
            best_c = c

    # Construct result
    domain = np.array([k for k in range(1, best_K + 1) if mu[k] != 0], dtype=np.int64)
    values = best_c * mu[1:best_K+1][domain-1].astype(np.float64)
    
    return domain, values
# EVOLVE-BLOCK-END
