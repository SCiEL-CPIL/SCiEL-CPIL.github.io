"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin before 90s kill
    
    # Precompute coprime relationships for numbers in [2, 256]
    # s_{ij} = A[i] + B[j], max value = 128+128 = 256
    max_val = 256
    # coprime[a][b] = True if gcd(a,b)==1
    coprime = [[False] * (max_val + 1) for _ in range(max_val + 1)]
    for a in range(2, max_val + 1):
        for b in range(a, max_val + 1):
            if gcd(a, b) == 1:
                coprime[a][b] = True
                coprime[b][a] = True
    
    best_count = 0
    best_A = None
    best_B = None
    
    # Strategy: enumerate A and B, check pairwise coprimality of distinct sums
    # To make this feasible, we'll use a smarter approach:
    # Generate candidate sets and check
    
    # Precompute all possible sums for a given A,B pair
    # We need to be efficient. Let's enumerate A first (smaller space if we fix structure)
    
    # Actually, let's think about it differently.
    # For 9 distinct pairwise-coprime sums, we need 9 pairwise-coprime numbers in [2,256]
    # that form a rank-1 additive grid.
    
    # Let me just do a systematic but bounded search.
    # Enumerate A as sorted tuples of 3 distinct values from [1,128]
    # For each A, enumerate B similarly, but prune early.
    
    # This is C(128,3)^2 which is too large. Need smarter approach.
    
    # Alternative: enumerate the 9 sum values directly, then try to factor them into A x B
    # The 9 sums s_ij = a_i + b_j form a rank-1 structure:
    # s_00 + s_11 = s_01 + s_10 (and similar for other pairs)
    
    # Let me try a different approach: 
    # 1. Pick a target set of k pairwise-coprime numbers in [2,256]
    # 2. Check if they can be arranged as a 3x3 rank-1 additive grid
    # 3. Extract A and B from the grid
    
    # For k=9: all 9 must be distinct and pairwise coprime
    # For k=8: 8 distinct, pairwise coprime (one collision in the grid)
    # etc.
    
    # Let's try to find good A,B by structured search.
    # Key insight: if A = {a1, a2, a3} and B = {b1, b2, b3},
    # the sums are a_i + b_j. For all to be pairwise coprime,
    # we need gcd(a_i + b_j, a_k + b_l) = 1 for all distinct pairs.
    
    # Let me try a randomized + local search approach with time budget.
    import random
    random.seed(42)
    
    # Try random pairs first
    for iteration in range(5000000):
        if time.time() > deadline:
            break
        
        # Random A and B
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        
        # Compute sums
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        
        if len(sums) <= best_count:
            continue
        
        # Check pairwise coprimality
        slist = sorted(sums)
        valid = True
        for i in range(len(slist)):
            for j in range(i+1, len(slist)):
                if not coprime[slist[i]][slist[j]]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            best_count = len(sums)
            best_A = A
            best_B = B
    
    # Now do local search around the best found
    if best_A is not None and time.time() < deadline:
        current_A = best_A[:]
        current_B = best_B[:]
        current_count = best_count
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try changing one element of A or B
            for idx in range(3):
                for new_val in range(1, 129):
                    if new_val == current_A[idx]:
                        continue
                    # Check distinctness
                    if new_val in current_A and current_A.index(new_val) != idx:
                        continue
                    new_A = current_A[:]
                    new_A[idx] = new_val
                    new_A.sort()
                    
                    sums = set()
                    for a in new_A:
                        for b in current_B:
                            sums.add(a + b)
                    
                    if len(sums) <= current_count:
                        continue
                    
                    slist = sorted(sums)
                    valid = True
                    for i in range(len(slist)):
                        for j in range(i+1, len(slist)):
                            if not coprime[slist[i]][slist[j]]:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        current_A = new_A
                        current_count = len(sums)
                        improved = True
                        break
                if improved:
                    break
            
            for idx in range(3):
                if not improved:
                    for new_val in range(1, 129):
                        if new_val == current_B[idx]:
                            continue
                        if new_val in current_B and current_B.index(new_val) != idx:
                            continue
                        new_B = current_B[:]
                        new_B[idx] = new_val
                        new_B.sort()
                        
                        sums = set()
                        for a in current_A:
                            for b in new_B:
                                sums.add(a + b)
                        
                        if len(sums) <= current_count:
                            continue
                        
                        slist = sorted(sums)
                        valid = True
                        for i in range(len(slist)):
                            for j in range(i+1, len(slist)):
                                if not coprime[slist[i]][slist[j]]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            current_B = new_B
                            current_count = len(sums)
                            improved = True
                            break
                    if improved:
                        break
            
            if current_count > best_count:
                best_count = current_count
                best_A = current_A
                best_B = current_B
    
    # Also try the seed solution as fallback
    seed_A = [1, 7, 13]
    seed_B = [4, 10, 16]
    seed_sums = set()
    for a in seed_A:
        for b in seed_B:
            seed_sums.add(a + b)
    seed_count = len(seed_sums)
    
    if best_count < seed_count:
        return [seed_A, seed_B]
    
    if best_A is None:
        return [seed_A, seed_B]
    
    return [sorted(best_A), sorted(best_B)]
# EVOLVE-BLOCK-END
