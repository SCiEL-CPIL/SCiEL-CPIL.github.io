import math

def construct():
    N = 10000000
    
    # Compute smallest prime factor (SPF) sieve
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Memoize sigma computation
    sigma_cache = {}
    
    def sigma(n):
        if n in sigma_cache:
            return sigma_cache[n]
        if n == 1:
            sigma_cache[n] = 1
            return 1
        temp = n
        result = 1
        while temp > 1:
            p = spf[temp]
            power = 1
            while temp % p == 0:
                temp //= p
                power *= p
            # sigma(p^k) = (p^(k+1) - 1) / (p - 1)
            result *= (power * p - 1) // (p - 1)
        sigma_cache[n] = result
        return result
    
    best_error = float('inf')
    best_pair = None
    
    # Search over k = |2a - 3b| starting from 1
    # For a given k, we want to maximize b to minimize error = k / (2b)
    # Case 1: 2a - 3b = k => a = (3b + k) / 2
    #   Constraints: a >= 1, a <= N, b >= 1, b <= N
    #   3b + k must be even
    #   b_min = max(1, (2 - k) // 3) but since k >= 1, b_min = 1
    #   b_max = min(N, (2*N - k) // 3)
    # Case 2: 2a - 3b = -k => a = (3b - k) / 2
    #   Constraints: a >= 1, a <= N, b >= 1, b <= N
    #   3b - k must be even
    #   b_min = max(1, (k + 2) // 3)
    #   b_max = min(N, (2*N + k) // 3)
    
    # We'll iterate k from 1 upward. For each k, we check both cases.
    # For each case, we iterate b from b_max downward (to find largest b first).
    # Once we find a valid pair with sigma(a) == sigma(b), we break (largest b for this k).
    # Early termination: if k / (2 * b_max_possible) >= best_error, stop.
    
    k = 1
    while True:
        # Early termination check
        # Maximum possible b is roughly 2N/3
        if k / (2.0 * (2 * N // 3)) >= best_error:
            break
        
        found_for_k = False
        
        # Case 1: 2a - 3b = k, a = (3b + k) // 2
        # b must satisfy: (3b + k) is even
        # b_max1 = min(N, (2*N - k) // 3)
        b_max1 = min(N, (2 * N - k) // 3)
        b_min1 = 1
        if b_max1 >= b_min1:
            # Determine parity requirement for b
            # 3b + k must be even
            # If k is even: 3b must be even => b must be even
            # If k is odd: 3b must be odd => b must be odd
            if k % 2 == 0:
                start_b = b_max1 if b_max1 % 2 == 0 else b_max1 - 1
            else:
                start_b = b_max1 if b_max1 % 2 == 1 else b_max1 - 1
            
            for b in range(start_b, b_min1 - 1, -2):
                a = (3 * b + k) // 2
                if a < 1 or a > N:
                    continue
                # Check sigma
                sa = sigma(a)
                sb = sigma(b)
                if sa == sb:
                    error = k / (2.0 * b)
                    if error < best_error:
                        best_error = error
                        best_pair = (a, b)
                    found_for_k = True
                    break
        
        # Case 2: 2a - 3b = -k, a = (3b - k) // 2
        # b must satisfy: (3b - k) is even
        # b_max2 = min(N, (2*N + k) // 3)
        # b_min2 = max(1, (k + 2) // 3)
        b_max2 = min(N, (2 * N + k) // 3)
        b_min2 = max(1, (k + 2) // 3)
        if b_max2 >= b_min2:
            # Determine parity requirement for b
            # 3b - k must be even
            # If k is even: 3b must be even => b must be even
            # If k is odd: 3b must be odd => b must be odd
            if k % 2 == 0:
                start_b = b_max2 if b_max2 % 2 == 0 else b_max2 - 1
            else:
                start_b = b_max2 if b_max2 % 2 == 1 else b_max2 - 1
            
            for b in range(start_b, b_min2 - 1, -2):
                a = (3 * b - k) // 2
                if a < 1 or a > N:
                    continue
                # Check sigma
                sa = sigma(a)
                sb = sigma(b)
                if sa == sb:
                    error = k / (2.0 * b)
                    if error < best_error:
                        best_error = error
                        best_pair = (a, b)
                    found_for_k = True
                    break
        
        k += 1
    
    if best_pair is None:
        # Fallback: return a valid pair with sigma(a) == sigma(b)
        # Simplest: a=1, b=1 but that gives error 0.5, and a/b=1, |1-1.5|=0.5
        # But we need a != b ideally? No, just sigma(a)=sigma(b) and a/b != 3/2
        # Let's find a simple one
        # sigma(1)=1, sigma(2)=3, sigma(3)=4, sigma(4)=7, sigma(5)=6, sigma(6)=12
        # sigma(1)=1, no other n has sigma=1 except n=1
        # sigma(2)=3, no other
        # sigma(3)=4, no other
        # sigma(4)=7, no other
        # sigma(5)=6, sigma(6)=12
        # Let's just use a=1, b=1? No, |1/1 - 3/2| = 0.5
        # Actually the problem says a, b positive integers. a=b=1 is valid if sigma(1)=sigma(1).
        # But let's find something better.
        # For now, return [1, 1] as fallback
        return [1, 1]
    
    return [int(best_pair[0]), int(best_pair[1])]
