"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def _is_pairwise_coprime(nums):
    distinct = list(set(nums))
    n = len(distinct)
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best_count = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    sums0 = set(a + b for a in best_A for b in best_B)
    if _is_pairwise_coprime(sums0):
        best_count = len(sums0)
    
    random.seed(42)
    
    # Structured search: arithmetic progressions
    for d_a in range(1, 20):
        if time.time() > deadline:
            break
        for d_b in range(1, 20):
            if time.time() > deadline:
                break
            for x in range(1, 60):
                if time.time() > deadline:
                    break
                a_vals = [x, x + d_a, x + 2*d_a]
                if max(a_vals) > 128:
                    continue
                for y in range(1, 60):
                    if time.time() > deadline:
                        break
                    b_vals = [y, y + d_b, y + 2*d_b]
                    if max(b_vals) > 128:
                        continue
                    sums = set(a + b for a in a_vals for b in b_vals)
                    cnt = len(sums)
                    if cnt <= best_count:
                        continue
                    if _is_pairwise_coprime(sums):
                        best_count = cnt
                        best_A = a_vals[:]
                        best_B = b_vals[:]
                        if best_count >= 9:
                            return best_A, best_B
    
    # Random search
    while time.time() < deadline:
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        sums = set(a + b for a in A for b in B)
        cnt = len(sums)
        if cnt <= best_count:
            continue
        if _is_pairwise_coprime(sums):
            best_count = cnt
            best_A = A[:]
            best_B = B[:]
            if best_count >= 9:
                break
    
    return best_A, best_B
# EVOLVE-BLOCK-END
