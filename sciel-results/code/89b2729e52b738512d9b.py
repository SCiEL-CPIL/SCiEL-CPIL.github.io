import math
import time
import bisect
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    MAX_VAL = 10210
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL ** 0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def compute_sf(n):
        sf_val = 1
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            if cnt % 2 == 1:
                sf_val *= p
        return sf_val
    
    # Precompute square-free parts for all numbers 211..10210
    sf_cache = {}
    for i in range(211, MAX_VAL + 1):
        sf_cache[i] = compute_sf(i)
    
    # Build reverse map: sf_value -> sorted list of numbers with that sf
    sf_to_nums = defaultdict(list)
    for i in range(211, MAX_VAL + 1):
        sf_to_nums[sf_cache[i]].append(i)
    for v in sf_to_nums:
        sf_to_nums[v].sort()
    
    def sf_xor(sf_a, target):
        """Return sf_c such that sf(sf_a * sf_c) = target."""
        def prime_set(n):
            ps = set()
            x = n
            for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
                if p * p > x:
                    break
                if x % p == 0:
                    ps.add(p)
                    while x % p == 0:
                        x //= p
            if x > 1:
                ps.add(x)
            return ps
        
        pa = prime_set(sf_a)
        pt = prime_set(target)
        pc = pa ^ pt  # symmetric difference
        result = 1
        for p in pc:
            result *= p
        return result
    
    # 1. Single element: [840], raw = 630
    best_sol = [840]
    best_raw = 630
    
    # 2. Pair search
    for a in range(211, MAX_VAL + 1):
        if a >= best_raw + 210:
            break
        if time.time() > deadline:
            break
        sf_a = sf_cache[a]
        sf_b_needed = sf_xor(sf_a, 210)
        if sf_b_needed in sf_to_nums:
            nums = sf_to_nums[sf_b_needed]
            idx = bisect.bisect_right(nums, a)
            if idx < len(nums):
                b = nums[idx]
                if b > a and b <= MAX_VAL:
                    raw = b - 210
                    if raw < best_raw:
                        best_raw = raw
                        best_sol = [a, b]
    
    # 3. Triplet search: iterate M (max element) upward
    for M in range(211, min(best_raw + 210, MAX_VAL + 1)):
        if time.time() > deadline:
            break
        sf_M = sf_cache[M]
        T = sf_xor(sf_M, 210)
        # Need a, c in (210, M), a < c, sf(a*c) = T
        for a in range(211, M):
            if time.time() > deadline:
                break
            sf_a = sf_cache[a]
            sf_c_needed = sf_xor(sf_a, T)
            if sf_c_needed in sf_to_nums:
                nums_c = sf_to_nums[sf_c_needed]
                idx = bisect.bisect_right(nums_c, a)
                if idx < len(nums_c) and nums_c[idx] < M:
                    c = nums_c[idx]
                    if a != c and a != M and c != M:
                        # Verify
                        prod = a * c * M * 210
                        k = math.isqrt(prod)
                        if k * k == prod:
                            best_raw = M - 210
                            best_sol = [a, c, M]
                            return [int(a), int(c), int(M)]
    
    # 4. Four-element search: iterate M (max element) upward
    if best_raw > 50:
        # For 4 elements a < b < c < M, need sf(a*b*c) = complement(sf_M) in target 210
        # T = sf_xor(M, 210), need sf(a*b*c) = T
        for M in range(211, min(best_raw + 210, MAX_VAL + 1)):
            if time.time() > deadline:
                break
            sf_M = sf_cache[M]
            T = sf_xor(sf_M, 210)
            # Need to find a < b < c < M such that sf(a*b*c) = T
            # Iterate c from M-1 down, then for each c, find a, b < c with sf(a*b) = sf_xor(c, T)
            for c in range(M - 1, 210, -1):
                if time.time() > deadline:
                    break
                sf_c = sf_cache[c]
                T_ab = sf_xor(sf_c, T)
                # Need a < b < c with sf(a*b) = T_ab
                # Iterate a from 211 to c-1
                for a in range(211, c):
                    if time.time() > deadline:
                        break
                    sf_a = sf_cache[a]
                    sf_b_needed = sf_xor(sf_a, T_ab)
                    if sf_b_needed in sf_to_nums:
                        nums_b = sf_to_nums[sf_b_needed]
                        idx = bisect.bisect_right(nums_b, a)
                        if idx < len(nums_b) and nums_b[idx] < c:
                            b = nums_b[idx]
                            if a != b and b != c and a != M and b != M and c != M:
                                # Verify
                                prod = a * b * c * M * 210
                                k = math.isqrt(prod)
                                if k * k == prod:
                                    best_raw = M - 210
                                    best_sol = [a, b, c, M]
                                    return [int(a), int(b), int(c), int(M)]
    
    result = [int(x) for x in best_sol]
    return result
