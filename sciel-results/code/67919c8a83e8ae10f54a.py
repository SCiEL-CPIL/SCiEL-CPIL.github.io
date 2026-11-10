# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def are_pairwise_coprime(nums):
    unique = list(set(nums))
    n = len(unique)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(unique[i], unique[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    def check_pair(A, B):
        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
        distinct_sums = set(sums)
        if len(distinct_sums) < 9:
            return False, 0
        s_list = list(distinct_sums)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                if gcd(s_list[i], s_list[j]) != 1:
                    return False, len(distinct_sums)
        return True, len(distinct_sums)

    # 1. Exhaustive search in small range [1, 20]
    # C(20,3) = 1140. 1140^2 = 1,299,600 pairs. ~2-3s in Python.
    candidates_tiny = []
    for x in range(1, 21):
        for y in range(x + 1, 21):
            for z in range(y + 1, 21):
                candidates_tiny.append([x, y, z])

    for A in candidates_tiny:
        if time.time() > deadline:
            break
        for B in candidates_tiny:
            if time.time() > deadline:
                break
            valid, count = check_pair(A, B)
            if valid and count == 9:
                return [A, B]
                
    # 2. Extended exhaustive search in range [1, 30] if time permits
    # C(30,3) = 4060. 4060^2 = 16.5M. Too slow for full, but we can be smarter.
    # Instead, use prime-targeted construction:
    # If all 9 sums are distinct primes, they are automatically pairwise coprime.
    # So we need A+B to be a 3x3 grid of distinct primes.
    # Strategy: pick A = [a1, a2, a3], then for each aj, B[j] + ai must be prime
    # for all i, and all 9 sums distinct.
    
    # Precompute primes up to 256 (max sum = 128+128)
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(256)
    
    # 3. Backtracking: choose A first, then find B such that all sums are distinct primes
    # For a given A, B must satisfy: for each j in B, {a_i + b_j for i in 1..3} are all prime
    # and the 9 values are all distinct.
    
    # Precompute for each candidate b, which a-values make a+b prime
    # For a given A and candidate b, check if a1+b, a2+b, a3+b are all prime
    
    # Search A in [1, 50] range, B in [1, 128]
    # But be smart: iterate A, then for each A, find valid B values
    
    best_result = None
    
    for a1 in range(1, 51):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 51):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 51):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                # Find all b in [1, 128] such that a1+b, a2+b, a3+b are all prime
                valid_bs = []
                for b in range(1, 129):
                    if is_prime[a1+b] and is_prime[a2+b] and is_prime[a3+b]:
                        valid_bs.append(b)
                
                # Need 3 distinct b's from valid_bs such that all 9 sums are distinct
                # Try all combinations of 3 from valid_bs
                if len(valid_bs) >= 3:
                    found = False
                    for b1_idx in range(len(valid_bs)):
                        if time.time() > deadline:
                            break
                        b1 = valid_bs[b1_idx]
                        s1 = {a1+b1, a2+b1, a3+b1}
                        for b2_idx in range(b1_idx+1, len(valid_bs)):
                            if time.time() > deadline:
                                break
                            b2 = valid_bs[b2_idx]
                            s2 = {a1+b2, a2+b2, a3+b2}
                            if s1 & s2:
                                continue
                            for b3_idx in range(b2_idx+1, len(valid_bs)):
                                if time.time() > deadline:
                                    break
                                b3 = valid_bs[b3_idx]
                                s3 = {a1+b3, a2+b3, a3+b3}
                                if s1 & s3 or s2 & s3:
                                    continue
                                B = [b1, b2, b3]
                                return [A, B]
    
    # 4. Random search as fallback
    import random
    random.seed(42)
    
    best_valid_A = None
    best_valid_B = None
    best_count = 0
    
    while time.time() < deadline:
        A = sorted(random.sample(range(1, 129), 3))
        B = sorted(random.sample(range(1, 129), 3))
        valid, count = check_pair(A, B)
        if valid and count > best_count:
            best_count = count
            best_valid_A = A[:]
            best_valid_B = B[:]
            if best_count == 9:
                return [best_valid_A, best_valid_B]
        
    # 5. Fallback: Return best valid found or a known valid set
    if best_valid_A is not None:
        return [best_valid_A, best_valid_B]
        
    # Known valid set with 5 distinct sums
    return [[1, 3, 5], [4, 6, 8]]
# EVOLVE-BLOCK-END
