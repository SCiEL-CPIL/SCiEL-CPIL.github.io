def construct():
    import math
    
    MAX_N = 10_000_000
    target_num, target_den = 3, 2
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_N + 1))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def sigma(n):
        if n <= 0:
            return 0
        if n == 1:
            return 1
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
        return result
    
    best_error = float('inf')
    best_a, best_b = None, None
    
    # Search over b values
    # For each b, a should be near 3b/2
    # a must be integer, so 3b/2 might not be integer
    # Let's iterate b and check a in a small window around 3b/2
    
    # To keep within time limits, let's be strategic about which b to check
    # We can check b up to some limit, and for each b check a few candidates
    
    # Let's check b from 1 to MAX_N, but that's 10^7 iterations
    # For each, we check maybe 5-10 a candidates
    # sigma computation is O(log n) with SPF
    
    # This might be feasible. Let's try.
    
    for b in range(1, MAX_N + 1):
        sigma_b = sigma(b)
        if sigma_b == 0:
            continue
        
        # a should be near 3b/2
        a_center = (3 * b) // 2
        # Check a values in a window around a_center
        # The window size: we want |a/b - 3/2| to be small
        # |a - 3b/2| / b < epsilon
        # |a - 3b/2| < epsilon * b
        # For epsilon = 0.01, window is 0.01 * b
        # But we want the best, so let's check a reasonable window
        
        # Let's check a from max(1, a_center - b//10) to min(MAX_N, a_center + b//10)
        # But that could be large. Let's use a smaller fixed window for now
        # Actually, let's check a few specific offsets
        
        # a = floor(3b/2), ceil(3b/2), and nearby
        a_low = max(1, (3 * b) // 2 - 10)
        a_high = min(MAX_N, (3 * b) // 2 + 10)
        
        for a in range(a_low, a_high + 1):
            if a == b:  # sigma would be same but ratio is 1, not close to 3/2
                continue
            if a > MAX_N or a < 1:
                continue
            
            # Compute error
            # |a/b - 3/2| = |2a - 3b| / (2b)
            num = abs(2 * a - 3 * b)
            if num == 0:
                continue  # invalid, exact 3/2
            
            err_num = num
            err_den = 2 * b
            
            # Compare with best
            if best_a is None or err_num * best_error_den > best_error_num * err_den:
                # Need to check sigma
                sigma_a = sigma(a)
                if sigma_a == sigma_b:
                    best_a = a
                    best_b = b
                    best_error_num = err_num
                    best_error_den = err_den
    
    if best_a is None:
        return [11, 6]
    
    return [best_a, best_b]
