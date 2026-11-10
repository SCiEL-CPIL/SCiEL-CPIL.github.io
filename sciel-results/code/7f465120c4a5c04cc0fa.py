import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    # Initial best from seed
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    # We'll try to use the Möbius function approach with scaling
    # Compute mu up to a reasonable max K
    max_K = 200000
    
    # Sieve for Möbius function
    mu = np.ones(max_K + 1, dtype=np.float64)
    mu[0] = 0
    primes = []
    is_prime = np.ones(max_K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, max_K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, max_K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= max_K:
                for j in range(i2, max_K + 1, i2):
                    mu[j] = 0
    
    # Precompute log(k) / k for efficiency
    k_vals = np.arange(1, max_K + 1, dtype=np.float64)
    log_k_over_k = np.log(k_vals) / k_vals
    log_k_over_k[0] = 0  # undefined at 0, but we won't use it
    
    # Try different K values, from small to large
    # For each K, we use mu[1:K+1] as the base values, then adjust f(1) for normalization
    # and scale if needed to satisfy M(x) <= 1.0001
    
    K_candidates = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    
    best_a = None
    
    for K in K_candidates:
        if time.time() > deadline - 120:
            break
        
        if K > max_K:
            continue
        
        # Base values: mu(1..K)
        base_values = mu[1:K+1].copy()
        
        # Compute S_tail = sum_{k=2}^K mu(k)/k
        # f(1) should be -S_tail so that sum f(k)/k = 0
        # But first, let's check M(x) for the unscaled values
        # Actually, we need to be careful: the normalization happens AFTER clipping
        # But let's first work with the raw mu values and see what M(x) looks like
        
        # For now, let's just use mu values and see if scaling is needed
        # The key insight: M(x) = sum_{k=1}^K mu(k) * floor(x/k)
        # For x <= K: M(x) = 1 (since sum_{d|x} mu(d) = 1 if x=1 else 0, 
        #   but actually M(x) = sum_{k=1}^x mu(k)*floor(x/k) = sum_{n=1}^x sum_{k|n} mu(k) = 1)
        # Wait, that's not quite right. Let me reconsider.
        # M(x) = sum_{k=1}^K mu(k) * floor(x/k)
        # = sum_{k=1}^K mu(k) * (number of multiples of k up to x)
        # = sum_{k=1}^K mu(k) * sum_{j: k*j <= x} 1
        # = sum_{k=1}^K mu(k) * sum_{j=1}^{floor(x/k)} 1
        # = sum_{n=1}^x sum_{k|n, k<=K} mu(k)
        # For x <= K: = sum_{n=1}^x sum_{k|n} mu(k) = 1 (since sum_{k|n} mu(k) = 1 if n=1 else 0)
        # Actually sum_{k|n} mu(k) = 1 if n=1, 0 otherwise. So M(x) = 1 for all x >= 1 when K >= x.
        # For x > K: M(x) = sum_{n=1}^x sum_{k|n, k<=K} mu(k)
        # This is not necessarily 1 anymore.
        
        # Let's compute M(x) for x in [1, 10K]
        # This is expensive. Let's use the fact that for x > K, 
        # M(x) = sum_{k=1}^K mu(k) * floor(x/k)
        
        # Efficient computation: for each k, mu[k] * floor(x/k) contributes
        # We can vectorize over x for each k, but that's O(K * 10K) operations total
        
        # Let's try a smarter approach: compute M(x) incrementally
        # M(x) = M(x-1) + sum_{k|... } 
        # Actually, M(x) - M(x-1) = sum_{k=1}^K mu(k) * (floor(x/k) - floor((x-1)/k))
        # floor(x/k) - floor((x-1)/k) = 1 if k|x, 0 otherwise
        # So M(x) - M(x-1) = sum_{k|x, k<=K} mu(k)
        
        # This is much more efficient! We can precompute the divisor sums
        # div_sum[x] = sum_{k|x, k<=K} mu(k) for x = 1..10K
        
        # But 10K can be up to 2,000,000, and we need div_sum for all x in [1, 10K]
        # We can compute this by iterating over k and adding mu[k] to all multiples of k
        
        if K > 50000:
            # Too large, skip for now to save time
            continue
        
        # Compute div_sum for x = 1 to 10K
        ten_K = 10 * K
        div_sum = np.zeros(ten_K + 1, dtype=np.float64)
        
        for k in range(1, K + 1):
            if mu[k] != 0:
                # Add mu[k] to all multiples of k up to ten_K
                for j in range(k, ten_K + 1, k):
                    div_sum[j] += mu[k]
        
        # M(x) = cumsum of div_sum
        # M(1) = div_sum[1]
        # M(x) = M(x-1) + div_sum[x]
        M_vals = np.cumsum(div_sum)
        
        max_M = np.max(M_vals)
        
        # Now we need to handle normalization and scaling
        # The evaluator clips values to [-10, 10] first
        # mu(k) is in {-1, 0, 1}, so no clipping needed
        # Then f(1) is replaced by f(1) - S where S = sum f(k)/k
        # We want sum f(k)/k = 0 after this replacement
        
        # If we set f(k) = c * mu(k) for all k, then:
        # S = c * sum_{k=1}^K mu(k)/k
        # f(1) becomes c*mu(1) - S = c*1 - c*sum_{k=1}^K mu(k)/k = c*(1 - sum_{k=1}^K mu(k)/k)
        # sum_{k=1}^K f(k)/k after replacement = S + (f(1) - S)/1 = S + f(1) - S = f(1)
        # Wait, that's not right. Let me re-read.
        # "f(1) is replaced by f(1) - S with S = sum_k f(k)/k"
        # So new_f(1) = old_f(1) - S
        # sum_k new_f(k)/k = sum_{k != 1} new_f(k)/k + new_f(1)/1
        # = sum_{k != 1} f(k)/k + (f(1) - S)
        # = (sum_{k != 1} f(k)/k + f(1)) - S
        # = S - S = 0
        # Yes, so after the replacement, the sum is always 0 regardless of f(1).
        
        # So the key is: what is the resulting f vector?
        # f(1) = c * mu(1) - S = c - c * sum_{k=1}^K mu(k)/k = c * (1 - S_mu)
        # where S_mu = sum_{k=1}^K mu(k)/k
        # f(k) = c * mu(k) for k >= 2
        
        # The M(x) constraint applies to the CLIPPED values BEFORE the f(1) replacement.
        # Wait, re-reading: "values are clipped to [-10, 10]. Then f(1) is replaced..."
        # So M(x) is computed with the clipped values, which for mu are just mu (since |mu| <= 1).
        # But f(1) is replaced AFTER M(x) is computed? No...
        # "M(x) = sum_k f(k) * floor(x/k) is computed for every integer x in [1, 10K]"
        # This is after the clipping and f(1) replacement? Or before?
        # The order is: clip -> replace f(1) -> compute M(x) -> check constraint -> compute metric
        # So M(x) uses the final f values (after clipping and f(1) replacement).
        
        # Hmm, this changes things. Let me re-read more carefully.
        # "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S...
        # so that afterwards sum_k f(k)/k = 0. M(x) = sum_k f(k) * floor(x/k) is computed..."
        # So M(x) is computed with the f values AFTER the f(1) replacement.
        
        # So if f(k) = c * mu(k) for all k, then after replacement:
        # f(1) = c * (1 - S_mu)
        # f(k) = c * mu(k) for k >= 2
        # M(x) = c * [ (1 - S_mu) * 1 + sum_{k=2}^K mu(k) * floor(x/k) ]
        #      = c * [ 1 - S_mu + sum_{k=2}^K mu(k) * floor(x/k) ]
        #      = c * [ 1 - S_mu + (M_unscaled(x) - mu(1)*1) ]
        #      = c * [ 1 - S_mu + M_unscaled(x) - 1 ]
        #      = c * [ M_unscaled(x) - S_mu ]
        
        # Where M_unscaled(x) = sum_{k=1}^K mu(k) * floor(x/k)
        
        # So max M(x) = c * max_x (M_unscaled(x) - S_mu)
        # We need this <= 1.0001
        
        # S_mu = sum_{k=1}^K mu(k)/k
        S_mu = np.sum(mu[1:K+1] / k_vals[1:K+1])
        
        # M_unscaled(x) for x in [1, 10K]
        # We already have M_vals which is M_unscaled(x) for x in [1, 10K]
        M_adjusted = M_vals - S_mu
        max_M_adjusted = np.max(M_adjusted)
        
        # We need c * max_M_adjusted <= 1.0001
        # Also, f(1) = c * (1 - S_mu) should be in [-10, 10] after clipping
        # And all f(k) = c * mu(k) should be in [-10, 10]
        # Since mu(k) in {-1, 0, 1}, we need c <= 10
        # And c * (1 - S_mu) should be in [-10, 10]
        
        c_max = 10.0
        if max_M_adjusted > 0:
            c_max = min(c_max, 1.0001 / max_M_adjusted)
        
        # Also check f(1) constraint
        f1_candidate = c_max * (1 - S_mu)
        if abs(f1_candidate) > 10:
            # Need to scale down more
            c_max = min(c_max, 10.0 / abs(1 - S_mu)) if 1 - S_mu != 0 else c_max
        
        if c_max <= 0:
            c_max = 0.001
        
        c = c_max
        
        # Compute the resulting f values
        f_values = c * mu[1:K+1].copy()
        f_values[0] = c * (1 - S_mu)  # f(1) after replacement
        
        # Compute the metric: a = -sum f(k) * log(k) / k
        a = -np.sum(f_values * log_k_over_k[1:K+1])
        
        if best_a is None or a > best_a:
            best_a = a
            best_domain = np.arange(1, K + 1, dtype=np.int64)
            best_values = f_values.copy()
    
    return best_domain, best_values

