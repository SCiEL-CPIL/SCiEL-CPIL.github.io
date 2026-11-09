import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    # Generate smallest prime factor (SPF) sieve up to 1000001
    MAX_N = 1000001
    spf = list(range(MAX_N + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        lpf = 1
        while n > 1:
            p = spf[n]
            lpf = p
            while n % p == 0:
                n //= p
        return lpf
    
    # Precompute largest prime factor for all numbers in [10000, 1000001]
    lpf_arr = [0] * (MAX_N + 1)
    for i in range(10000, MAX_N + 1):
        lpf_arr[i] = largest_prime_factor(i)
    
    # Scan for minimum of max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf_arr[10000], lpf_arr[10001])
    
    for n in range(10001, 1000001):
        if time.time() > deadline:
            break
        val = max(lpf_arr[n], lpf_arr[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
