"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _pairwise_coprime(values):
    vals = list(set(values))
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(vals[i], vals[j]) > 1:
                return False
    return True

def _score_pair(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if not _pairwise_coprime(sums):
        return -1
    return len(sums)

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Precompute gcd table for 2..256
    max_sum = 256
    gcd_table = [[0] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(2, max_sum + 1):
        for j in range(i, max_sum + 1):
            gcd_table[i][j] = math.gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    # Strategy: For each A, we want to find B such that A+B has max distinct pairwise-coprime values.
    # Key: sums range from 2 to 256. We want to pick B so that the 9 sums (3x3) are distinct and coprime.
    # 
    # For a given A, let's see which sums are "available" and coprime to each other.
    # We'll iterate over possible B values (not triplets, just individual values) and build B greedily.
    #
    # Actually, let's do: for each A, try all combinations of B values from [1,128] of size 3.
    # But C(128,3) = 336K per A, and 336K A's = 1e11. Still too slow.
    #
    # Better: Precompute for each sum s whether it can be part of a coprime set.
    # For a given A, the sums are a_i + b_j. We want to choose B = {b1,b2,b3} to maximize distinct coprime sums.
    #
    # Let's do a two-stage approach:
    # Stage 1: For each A, compute the set of possible sums and their coprimality structure.
    # Stage 2: Find B that maximizes distinct coprime sums.
    #
    # Simpler: Just do a smart random search + local optimization with time limit.
    
    import random
    random.seed(42)
    
    # Random sampling phase
    all_vals = list(range(1, 129))
    
    # Try a bunch of random pairs
    for _ in range(200000):
        if time.time() > deadline:
            break
        A = list(random.sample(all_vals, 3))
        A.sort()
        B = list(random.sample(all_vals, 3))
        B.sort()
        
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        
        # Check coprime
        ds = list(sums)
        valid = True
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                if gcd_table[ds[i]][ds[j]] > 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            sc = len(ds)
            if sc > best_score:
                best_score = sc
                best_A = A
                best_B = B
    
    # Local optimization: try to improve best_A and best_B
    if time.time() < deadline - 10:
        for _ in range(50000):
            if time.time() > deadline:
                break
            
            # Try modifying A or B
            if random.random() < 0.5:
                # Modify A
                A = list(best_A)
                idx = random.randint(0, 2)
                old = A[idx]
                A[idx] = random.randint(1, 128)
                while len(set(A)) != 3:
                    A[idx] = random.randint(1, 128)
                A.sort()
                B = best_B
            else:
                # Modify B
                B = list(best_B)
                idx = random.randint(0, 2)
                old = B[idx]
                B[idx] = random.randint(1, 128)
                while len(set(B)) != 3:
                    B[idx] = random.randint(1, 128)
                B.sort()
                A = best_A
            
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            
            ds = list(sums)
            valid = True
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    if gcd_table[ds[i]][ds[j]] > 1:
                        valid = False
                        break
            if not valid:
                continue
            
            sc = len(ds)
            if sc > best_score:
                best_score = sc
                best_A = A
                best_B = B
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
