import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    MAX_N = 10_000_000
    
    # Build SPF
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:
            step = i
            start = i * i
            for j in range(start, MAX_N + 1, step):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute sigma for all numbers up to MAX_N
    # Using the multiplicative property
    sigma_arr = [0] * (MAX_N + 1)
    sigma_arr[1] = 1
    
    for i in range(2, MAX_N + 1):
        p = spf[i]
        m = i // p
        if m % p == 0:
            # p^2 divides i
            # i = p^k * r, p∤r, k>=2
            # sigma[i] = sigma[i/p] + p^k * sigma[r]
            # But we need to track p^k * sigma[r]
            # Let's use a different approach
            pass
        else:
            # p^1 divides i, p^2 doesn't
            sigma_arr[i] = sigma_arr[m] * (1 + p)
    
    # This doesn't work for all cases. Let me use a different method.
    
    # Actually, let me just compute sigma on the fly with caching.
    
    sigma_cache = {1: 1}
    
    def sigma(n):
        if n in sigma_cache:
            return sigma_cache[n]
        orig = n
        result = 1
        while n > 1:
            p = spf[n]
            power_sum = 1
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                power_sum += power
            result *= power_sum
        sigma_cache[orig] = result
        return result
    
    best_diff = float('inf')
    best_pair = None
    
    # Local variables for speed
    spf_local = spf
    sigma_func = sigma
    MAX = MAX_N
    
    for b in range(1, MAX + 1):
        if time.time() > deadline:
            break
        # a ≈ 3b/2
        three_b = 3 * b
        a = three_b // 2
        if three_b % 2 == 0:
            # a is exact 3b/2, skip
            continue
        # a = floor(3b/2), also check a+1
        if a >= 1 and a <= MAX:
            sa = sigma_func(a)
            sb = sigma_func(b)
            if sa == sb:
                diff_num = abs(2 * a - three_b)  # = 1
                diff = diff_num / (2 * b)
                if diff < best_diff:
                    best_diff = diff
                    best_pair = (a, b)
        
        a2 = a + 1
        if a2 >= 1 and a2 <= MAX:
            if 2 * a2 == three_b:
                continue
            sa2 = sigma_func(a2)
            sb = sigma_func(b)  # already computed
            if sa2 == sb:
                diff_num = abs(2 * a2 - three_b)
                diff = diff_num / (2 * b)
                if diff < best_diff:
                    best_diff = diff
                    best_pair = (a2, b)
    
    if best_pair is not None:
        return [best_pair[0], best_pair[1]]
    else:
        return [11, 6]
