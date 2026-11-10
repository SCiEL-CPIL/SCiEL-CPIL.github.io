import numpy as np
import math
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600  # seconds

    # Use a heuristic based on the Möbius function.
    # f(k) = mu(k) for k >= 2.
    # Adjust f(1) such that sum f(k)/k = 0.
    # Scale if M(x) > 1.0001.

    K = 200000  # Max allowed K

    # 1. Compute Möbius function up to K
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
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
                
    # 2. Construct initial values
    values = np.zeros(K, dtype=np.float64)
    # values[0] corresponds to k=1
    # values[i-1] corresponds to k=i
    values[1:] = mu[2:K+1].astype(np.float64)
    
    # 3. Adjust f(1)
    # S_rest = sum_{k=2}^K f(k)/k
    k_indices = np.arange(2, K + 1, dtype=np.float64)
    S_rest = np.sum(values[1:] / k_indices)
    values[0] = -S_rest
    
    # 4. Compute M(x) for x in [1, 10K] to check feasibility
    # M(x) = sum_{k=1}^K f(k) floor(x/k)
    # We use a difference array technique.
    # diff[x] accumulates contributions.
    # For each k, f(k) adds f(k) to M(x) for x in [j*k, (j+1)*k - 1].
    # This is equivalent to:
    # For j=1 to floor(10K/k):
    #   diff[j*k] += f(k)
    #   diff[(j+1)*k] -= f(k)  (if (j+1)*k <= 10K)
    
    max_x = 10 * K
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    # Iterate k from 1 to K
    # To optimize, we can vectorize the inner loop for each k?
    # No, the range of j depends on k.
    # But we can just loop k. 200k iterations is fine.
    for k in range(1, K + 1):
        f_k = values[k-1]
        if f_k == 0:
            continue
        # j goes from 1 to max_x // k
        max_j = max_x // k
        # start indices: k, 2k, ..., max_j*k
        # end indices: 2k, 3k, ..., (max_j+1)*k
        # We add f_k at starts and subtract at ends.
        starts = np.arange(k, max_j * k + 1, k)
        ends = starts + k
        
        # Update diff
        # We need to be careful with indices.
        # diff[starts] += f_k
        # diff[ends] -= f_k (where ends <= max_x)
        
        # Vectorized update
        diff[starts] += f_k
        # Only subtract if end index is within bounds
        valid_ends = ends[ends <= max_x]
        if len(valid_ends) > 0:
            diff[valid_ends] -= f_k
            
    # Prefix sum to get M(x)
    M_vals = np.cumsum(diff)
    # M_vals[x-1] is M(x) for x in 1..max_x?
    # diff[1] gets f(1) for x=1?
    # Let's trace:
    # k=1: starts=[1, 2, ..., max_x]. diff[1]+=f(1), diff[2]-=f(1).
    # So M(1) = diff[1] = f(1). Correct (floor(1/1)=1).
    # M(2) = diff[1] + diff[2] = f(1) - f(1) = 0?
    # Wait. M(2) = f(1)*2 + f(2)*1.
    # My difference array logic:
    # Contribution of f(k) to M(x) is f(k) * floor(x/k).
    # floor(x/k) is constant on [j*k, (j+1)*k - 1] with value j.
    # So for a specific k, the term is f(k)*j on that interval.
    # My diff update:
    # Add f(k) at start j*k? No.
    # If I add f(k) at j*k and subtract at (j+1)*k, the prefix sum will have value f(k) on [j*k, (j+1)*k - 1].
    # But the value should be j * f(k).
    # So I should add j * f(k) at j*k and subtract j * f(k) at (j+1)*k?
    # No, that's not how difference arrays work for constant values.
    # If I want value V on [A, B], I do diff[A] += V, diff[B+1] -= V.
    # Here V = j * f(k).
    # So for each k and each j:
    #   A = j * k
    #   B = (j+1) * k - 1
    #   V = j * f(k)
    #   diff[A] += V
    #   diff[B+1] -= V  (which is diff[(j+1)*k] -= V)
    
    # Let's re-implement this correctly.
    diff = np.zeros(max_x + 2, dtype=np.float64)
    for k in range(1, K + 1):
        f_k = values[k-1]
        if f_k == 0:
            continue
        max_j = max_x // k
        # j from 1 to max_j
        # V_j = j * f_k
        # Starts at j*k
        # Ends at (j+1)*k - 1 -> subtract at (j+1)*k
        
        # Vectorize over j?
        # j array: 1, ..., max_j
        js = np.arange(1, max_j + 1)
        Vs = js * f_k
        starts = js * k
        ends = (js + 1) * k
        
        diff[starts] += Vs
        # Subtract at ends if within bounds
        valid_ends = ends[ends <= max_x]
        if len(valid_ends) > 0:
            # We need to subtract the corresponding Vs
            # valid_ends corresponds to a subset of ends.
            # We need the Vs corresponding to these ends.
            # ends = (js+1)*k.
            # If end <= max_x, then js+1 <= max_x/k -> js <= max_x/k - 1.
            # So we can just take the first `len(valid_ends)` elements of Vs?
            # Yes, because ends is monotonically increasing.
            diff[valid_ends] -= Vs[:len(valid_ends)]
            
    M_vals = np.cumsum(diff)
    # M_vals[i] corresponds to M(i+1)?
    # diff[1] is updated by k=1, j=1: start=1, V=1*f(1).
    # diff[1] += f(1).
    # diff[2] -= f(1).
    # M(1) = diff[1] = f(1). Correct.
    # M(2) = diff[1] + diff[2] = f(1) - f(1) + ... 
    # Wait, for k=2, j=1: start=2, V=1*f(2).
    # diff[2] += f(2).
    # So M(2) = f(1) - f(1) + f(2) = f(2).
    # But M(2) should be f(1)*2 + f(2)*1 = 2f(1) + f(2).
    # My logic is flawed.
    
    # Let's stick to the simpler O(K log K) check or just trust the Möbius property for small K and scale down.
    # Actually, let's just compute M(x) for a subset of x to estimate the max.
    # Or use the fact that M(x) is close to 1.
    # If we just return the unscaled Möbius, it might be infeasible.
    # Let's compute M(x) for x = 1, 2, ..., 1000 and x = K, 2K, ..., 10K to get a sense.
    
    # To be safe and fast, let's just compute M(x) for all x using the correct difference array method.
    # The correct method:
    # M(x) = sum_k f(k) floor(x/k)
    # Let's use the property that floor(x/k) = sum_{j=1}^{floor(x/k)} 1.
    # M(x) = sum_k f(k) sum_{j=1}^{floor(x/k)} 1 = sum_{j,k} f(k) [k <= x/j]
    # This is not easier.
    
    # Let's just do the direct calculation for the first 1000 x and last 1000 x, and assume the max is there.
    # Or better, compute it for all x if K is small enough.
    # For K=2000, 10K=20000. 2000 * 20000 / 2 = 20M ops. Fast in NumPy.
    # For K=200000, 10K=2M. 200k * 2M = 400B ops. Too slow.
    
    # So for K=200000, we must rely on the theoretical bound or a smaller K.
    # Let's use K=2000 for the LP-free heuristic as well?
    # No, larger K gives better score (more terms).
    # But we can't verify feasibility for K=200000 easily.
    # However, it is known that sum_{k=1}^K mu(k) floor(x/k) = 1 + O(K^{1/2+epsilon}).
    # The error is small.
    # Let's use K=2000 to be safe and fast, and ensure feasibility.
    
    K = 2000
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
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
                
    values = np.zeros(K, dtype=np.float64)
    values[1:] = mu[2:K+1].astype(np.float64)
    k_indices = np.arange(2, K + 1, dtype=np.float64)
    S_rest = np.sum(values[1:] / k_indices)
    values[0] = -S_rest
    
    # Verify feasibility for K=2000
    # M(x) = sum_{k=1}^K f(k) floor(x/k)
    # We can compute this for all x in [1, 10K] using matrix multiplication?
    # M = F @ FloorMatrix
    # FloorMatrix[x-1, k-1] = floor(x/k)
    # Size: 20000 x 2000. 40M floats. 320MB.
    # This might be slow to allocate and multiply.
    # Let's do it in chunks or just check a few x.
    # Given the time, let's just scale by 0.5 to be safe.
    # The metric is linear, so scaling by 0.5 halves the score, but ensures feasibility.
    # We can try to find the exact scaling factor by checking max M(x).
    
    # Compute M(x) for x in [1, 10K] efficiently for K=2000.
    x_vals = np.arange(1, 10 * K + 1)
    k_vals = np.arange(1, K + 1)
    # Floor matrix
    # To avoid 320MB, we can compute M(x) in a loop over k?
    # M = np.zeros(len(x_vals))
    # for k in range(1, K+1):
    #     M += values[k-1] * np.floor(x_vals / k)
    # This is 2000 iterations of 20000 ops. 40M ops. Very fast.
    
    M = np.zeros(len(x_vals), dtype=np.float64)
    for k in range(1, K + 1):
        M += values[k-1] * np.floor(x_vals / k)
        
    M_max = np.max(M)
    
    if M_max > 1.0001:
        scale = 1.0001 / M_max
        values *= scale
    elif M_max < 0.9999:
        # If it's too low, we can scale up?
        # But M(x) must be <= 1.0001. It can be lower.
        # The metric is -sum f(k) log(k)/k.
        # If we scale up, we increase the metric (make it less negative? No, f(k) are mixed signs).
        # Actually, for Möbius, the sum is positive.
        # So we want to maximize the scale.
        # But we are constrained by M(x) <= 1.0001.
        # So we should scale up to the limit 1.0001.
        scale = 1.0001 / M_max
        values *= scale
        
    domain = np.arange(1, K + 1, dtype=np.int64)
    return domain, values
