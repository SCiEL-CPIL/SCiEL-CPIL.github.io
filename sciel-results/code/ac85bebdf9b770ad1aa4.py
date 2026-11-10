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

    # Strategy 1: Exhaustive search in small range [1, 20]
    # This is fast and likely to find the optimal solution if it exists with small numbers.
    # C(20,3) = 1140. 1140*1140 = 1.3M checks.
    # Each check is ~9 gcds. ~12M gcds. Python can do ~1-5M ops/sec.
    # This might take 2-10 seconds.
    
    candidates = []
    for x in range(1, 21):
        for y in range(x + 1, 21):
            for z in range(y + 1, 21):
                candidates.append([x, y, z])
                
    # Optimization: Pre-filter candidates?
    # Not strictly necessary if we break early.
    
    for A in candidates:
        if time.time() > deadline - 30: # Leave 30s for fallback/random
            break
        for B in candidates:
            if time.time() > deadline - 30:
                break
            valid, count = check_pair(A, B)
            if valid and count == 9:
                return [A, B]
                
    # Strategy 2: Random search in larger range [1, 128]
    import random
    random.seed(42)
    
    best_valid_A = None
    best_valid_B = None
    best_count = 0
    
    # We want to find 9. If we find 9, return immediately.
    # If not, we keep the best valid found.
    
    attempts = 0
    while time.time() < start_time + 50:
        A = sorted(random.sample(range(1, 129), 3))
        B = sorted(random.sample(range(1, 129), 3))
        valid, count = check_pair(A, B)
        if valid and count > best_count:
            best_count = count
            best_valid_A = A[:]
            best_valid_B = B[:]
            if best_count == 9:
                return [best_valid_A, best_valid_B]
        attempts += 1
        
    # Strategy 3: Fallback
    if best_valid_A is not None:
        return [best_valid_A, best_valid_B]
        
    # Known valid set with 9 distinct sums?
    # Let's construct one manually to be safe.
    # A = [1, 2, 4], B = [1, 5, 11] failed.
    # Let's try A = [1, 3, 8], B = [1, 4, 10]
    # Sums:
    # 1+1=2, 1+4=5, 1+10=11
    # 3+1=4, 3+4=7, 3+10=13
    # 8+1=9, 8+4=12, 8+10=18
    # Distinct: {2,5,11,4,7,13,9,12,18}
    # Pairs:
    # 2,5,11,4,7,13,9,12,18
    # 2 and 4 share 2. Fail.
    
    # Let's try A = [1, 5, 11], B = [1, 2, 6]
    # Sums:
    # 2, 3, 7
    # 6, 7, 11 -> 7 duplicate.
    
    # Let's rely on the search. If it fails, return a valid 5-sum set.
    # A = [1, 3, 5], B = [4, 6, 8]
    # Sums:
    # 5, 7, 9
    # 7, 9, 11
    # 9, 11, 13
    # Distinct: {5, 7, 9, 11, 13}
    # 5,7,9,11,13 are pairwise coprime?
    # 9 and 3? No 3. 9 and 6? No 6.
    # gcd(5,7)=1, gcd(5,9)=1, gcd(5,11)=1, gcd(5,13)=1
    # gcd(7,9)=1, gcd(7,11)=1, gcd(7,13)=1
    # gcd(9,11)=1, gcd(9,13)=1
    # gcd(11,13)=1
    # Yes. Count 5.
    
    return [[1, 3, 5], [4, 6, 8]]
# EVOLVE-BLOCK-END
