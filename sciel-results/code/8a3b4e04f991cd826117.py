import time
from math import isqrt
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    def time_left():
        return deadline - time.time()

    # Generate powerful numbers up to a limit M.
    # A powerful number n has all prime factors with exponent >= 2.
    # So n = a^2 * b^3 where b is square-free.
    # Count of powerfuls up to X is ~ c * X^(2/5).
    
    # We need to balance M (larger M -> more powerfuls -> more pairs) 
    # against generation time and O(K^2) counting time.
    # For M=50M, K ~ 2000, O(K^2) ~ 4M ops, very fast.
    # For M=100M, K ~ 3000, O(K^2) ~ 9M ops, still fast.
    # For M=200M, K ~ 5000, O(K^2) ~ 25M ops, maybe 1-2s in Python.
    # Let's try M=100_000_000 first. If time allows, we can try higher?
    # But generation also takes time.
    
    # Let's stick to a robust M that ensures K is manageable for O(K^2) in Python.
    # K ~ 5000 is the limit for comfortable O(K^2) in pure Python within seconds.
    # M=100M -> K~3000. Good.
    # M=200M -> K~5000. Risky if Python loop is slow.
    # Let's use M=150_000_000 as a sweet spot?
    # Actually, the inspiration 1 used M=50M and got 5.8.
    # The current program used FFT with M=30M and got 4.7.
    # The FFT approach is O(M log M) which is fast for large M, but limited by memory.
    # The O(K^2) approach is O(K^2) where K ~ M^(2/5).
    # Let's compare:
    # M=10^8: FFT size 2^27 ~ 134M floats ~ 1GB. OK. Time ~ 1s.
    # M=10^8: K ~ 3000. O(K^2) ~ 9M ops. Time ~ 0.5s in Python? Maybe 1s.
    # M=10^9: FFT size 2^30 ~ 1B floats ~ 8GB. TOO MUCH MEMORY.
    # M=10^9: K ~ 10,000. O(K^2) ~ 100M ops. Time ~ 10s in Python.
    
    # So for M > 10^8, FFT runs out of memory.
    # O(K^2) can go to M=10^9 if we optimize the inner loop?
    # 100M iterations in Python is ~10-20s.
    # Plus generation time for M=10^9.
    # max_b for M=10^9 is ~1000.
    # Generation is fast.
    
    # Let's try M = 500_000_000 (500M).
    # K ~ (500M)^(0.4) ~ 25000? No.
    # (5*10^8)^0.4 = 5^0.4 * 10^3.2 = 1.9 * 1584 ~ 3000?
    # Let's check: 100^2.5 = 10^5. 1000^2.5 = 10^7.5.
    # Formula: Count ~ c * X^(2/5).
    # X=10^8 -> X^0.4 = 10^3.2 = 1584.
    # X=5*10^8 -> 5^0.4 * 1584 = 1.9 * 1584 = 3000.
    # So K ~ 3000 for M=500M?
    # Wait, 100^2 = 10000. 100^3 = 1000000.
    # Powerful numbers: 1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81, 100, 121, 125, 128, 144, 169, 196, 216, 225, 243, 256, 289, 324, 343, 361, 400, ...
    # Up to 100: 1,4,8,9,16,25,27,32,36,49,64,81,100. Count=13.
    # 100^(2/5) = 10^0.8 = 6.3. c ~ 2?
    # Up to 10^6: 10^6^(0.4) = 10^2.4 = 251. c=2 -> 500.
    # Up to 10^8: 10^8^(0.4) = 10^3.2 = 1584. c=2 -> 3168.
    # So for M=500M, K ~ 5000.
    # O(K^2) = 25M iterations.
    # In Python, 25M iterations of a simple loop (addition, dict update) takes ~2-3 seconds.
    # This is feasible!
    
    # Strategy: Use O(K^2) counting with a large M.
    # This avoids FFT memory limits and allows searching up to 2*M = 10^9.
    
    M = 500_000_000
    
    # Precompute square-free numbers up to M^(1/3) + 2
    max_b = int(M ** (1.0 / 3.0)) + 2  # ~800 for M=500M
    
    # Sieve for square-free check
    # We need to check if b is square-free for b up to 800.
    # Simple trial division is enough.
    
    def is_square_free(n):
        if n < 2:
            return n == 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % (p * p) == 0:
                return False
            while temp % p == 0:
                temp //= p
            p += 1
        return True

    powerful = set()
    
    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        b_cubed = b ** 3
        if b_cubed > M:
            break
        max_a = isqrt(M // b_cubed)
        for a in range(1, max_a + 1):
            powerful.add(a * a * b_cubed)
            
    if time.time() > deadline - 20:
        # Fallback if generation was too slow (unlikely)
        M = 50_000_000
        max_b = int(M ** (1.0 / 3.0)) + 2
        powerful = set()
        for b in range(1, max_b + 1):
            if not is_square_free(b):
                continue
            b_cubed = b ** 3
            if b_cubed > M:
                break
            max_a = isqrt(M // b_cubed)
            for a in range(1, max_a + 1):
                powerful.add(a * a * b_cubed)

    powerful_list = sorted(powerful)
    powerful_set = powerful
    K = len(powerful_list)
    
    # O(K^2) counting
    # We want to find N with max representations.
    # counts[N] = number of pairs (a,b) with a<=b, a+b=N, a,b in powerful_set.
    
    counts = defaultdict(int)
    best_N = 0
    best_count = 0
    
    # Optimization: 
    # For each a in powerful_list:
    #   for each b in powerful_list where b >= a:
    #       N = a + b
    #       counts[N] += 1
    #       if counts[N] > best_count: update best
    
    # This is O(K^2).
    # To speed up, we can break early if a + b > 2*M? 
    # No, N can be up to 2*M.
    # But we only care about N up to 2*M.
    # Since b <= M, a <= M, N <= 2*M.
    # All sums are valid.
    
    # Let's just run the loop.
    # 5000^2 / 2 = 12.5M iterations.
    # Python loop overhead is ~100ns per iteration? 12.5M * 100ns = 1.25s.
    # Dict update is slower, maybe 200-300ns.
    # Total ~ 3-4s.
    # Acceptable.
    
    for i in range(K):
        a = powerful_list[i]
        # b must be >= a
        for j in range(i, K):
            b = powerful_list[j]
            N = a + b
            c = counts[N] + 1
            counts[N] = c
            if c > best_count:
                best_count = c
                best_N = N
        
        # Time check
        if time.time() > deadline - 10:
            break

    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}

    # Reconstruct pairs for best_N
    pairs = []
    limit = best_N // 2
    # Binary search for end of a <= limit
    # powerful_list is sorted
    end_idx = np.searchsorted(np.array(powerful_list), limit)
    
    for i in range(end_idx):
        a = powerful_list[i]
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': int(best_N), 'pairs': pairs}
