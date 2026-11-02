import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def calc_score(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(U)
        if len(U) < 2:
            return 0.0
        
        max_u = U[-1]
        if max_u == 0:
            return 0.0
            
        sum_set = set()
        diff_set = set()
        
        n = len(U)
        for i in range(n):
            for j in range(n):
                sum_set.add(U[i] + U[j])
                diff_set.add(U[i] - U[j])
        
        size_sum = len(sum_set)
        size_diff = len(diff_set)
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
            
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Evaluate seed
    best_score = calc_score(best_set)
    
    # Strategy 1: Exhaustive search for small K and M
    # K is size of set, M is max element
    # We know K=4, M=5 works well.
    # Let's try K=3 to 7, M=5 to 40
    
    # Pre-generate some candidates
    candidates = []
    
    # Add known good ones
    candidates.append([0, 1, 3])
    candidates.append([0, 1, 2, 5])
    candidates.append([0, 1, 3, 5])
    
    # Random sampling
    random.seed(42)
    
    max_k = 8
    max_m = 50
    samples_per_config = 500
    
    for k in range(3, max_k + 1):
        if time.time() > deadline:
            break
        for m in range(5, max_m + 1):
            if time.time() > deadline:
                break
            
            # If m is small relative to k, we can be more exhaustive or just sample
            # Number of ways to choose k-1 from m is C(m, k-1)
            # If C(m, k-1) is small, we can iterate all
            
            # Simple heuristic: if m < 20, iterate more
            # Otherwise sample
            
            # We always include 0. So we choose k-1 from 1..m
            
            # To speed up, we can just sample
            for _ in range(samples_per_config):
                if time.time() > deadline:
                    break
                
                # Choose k-1 distinct numbers from 1 to m
                if k - 1 > m:
                    continue
                try:
                    others = random.sample(range(1, m + 1), k - 1)
                except ValueError:
                    continue
                
                U = [0] + sorted(others)
                s = calc_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
                    
                # Local improvement: try swapping one element
                # Pick a random index in U (except 0) and replace with a random new value
                if k > 1:
                    idx = random.randint(1, k - 1)
                    new_val = random.randint(1, m + 10) # Allow slightly larger
                    if new_val not in U:
                        U_new = U[:]
                        U_new[idx] = new_val
                        U_new.sort()
                        s_new = calc_score(U_new)
                        if s_new > best_score:
                            best_score = s_new
                            best_set = U_new[:]
    
    # Final check
    if time.time() < deadline - 5.0:
        # Try a few specific structured sets
        structured = [
            [0, 1, 3],
            [0, 1, 2, 5],
            [0, 1, 3, 5],
            [0, 1, 2, 5, 9],
            [0, 1, 2, 5, 9, 18],
            [0, 1, 3, 9, 27],
            [0, 1, 4, 10, 28],
            [0, 2, 3, 5, 8],
            [0, 1, 2, 4, 8, 16],
        ]
        for U in structured:
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]

    return best_set
# EVOLVE-BLOCK-END
