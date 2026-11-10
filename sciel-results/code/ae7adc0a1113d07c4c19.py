import numpy as np

def construct():
    N = 10000000
    
    # Compute smallest prime factor (SPF) sieve
    spf = np.arange(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_sigma(n):
        if n == 1:
            return 1
        result = 1
        temp = n
        while temp > 1:
            p = spf[temp]
            # Count power of p
            power_sum = 1
            power_val = 1
            while temp % p == 0:
                temp //= p
                power_val *= p
                power_sum += power_val
            result *= power_sum
        return result

    # Search for k starting from 1
    # We want to minimize k/(2b).
    # For each k, we find the b that maximizes b (minimizes error) for |2a-3b|=k.
    
    for k in range(1, 1000000):  # Limit k search to reasonable range
        candidates = []
        
        # Case 1: 2a - 3b = k
        # a = (3b + k) / 2
        # Constraints: 1 <= a <= N, 1 <= b <= N
        # 3b + k <= 2N => b <= (2N - k) / 3
        # 3b + k >= 2 => b >= (2 - k) / 3 (always true for k>=1, b>=1)
        # Parity: 3b + k must be even => b and k must have same parity?
        # 3b + k even => b + k even (since 3b has same parity as b) => b % 2 == k % 2
        
        max_b_case1 = min(N, (2 * N - k) // 3)
        if max_b_case1 < 1:
            pass
        else:
            # Adjust parity
            if max_b_case1 % 2 != k % 2:
                max_b_case1 -= 1
            if max_b_case1 >= 1:
                b = max_b_case1
                a = (3 * b + k) // 2
                if 1 <= a <= N and a != b:
                    candidates.append((a, b))
        
        # Case 2: 3b - 2a = k
        # 2a = 3b - k => a = (3b - k) / 2
        # Constraints: 1 <= a <= N, 1 <= b <= N
        # 3b - k <= 2N => b <= (2N + k) / 3
        # 3b - k >= 2 => b >= (k + 2) / 3
        
        max_b_case2 = min(N, (2 * N + k) // 3)
        min_b_case2 = (k + 2) // 3
        if min_b_case2 > max_b_case2:
            pass
        else:
            # Adjust parity: 3b - k even => b - k even => b % 2 == k % 2
            if max_b_case2 % 2 != k % 2:
                max_b_case2 -= 1
            if max_b_case2 >= min_b_case2:
                b = max_b_case2
                a = (3 * b - k) // 2
                if 1 <= a <= N and a != b:
                    candidates.append((a, b))
        
        # Check candidates
        for a, b in candidates:
            if get_sigma(a) == get_sigma(b):
                return [a, b]

    # Fallback if no small k found (unlikely)
    # Return a known valid pair or search more
    # For now, return a pair with a=b (error 0.5) if we must
    return [1, 1]
