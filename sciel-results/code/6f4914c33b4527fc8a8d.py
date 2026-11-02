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
    
    # Strategy 1: Evaluate structured candidates first
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
        # Mersenne-like sets: 2^k - 1
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 3, 7, 15, 31, 63],
        # Other structured
        [0, 1, 3, 6, 10],
        [0, 1, 3, 6, 10, 15],
        [0, 1, 4, 10, 22],
        [0, 1, 4, 10, 22, 46],
        [0, 2, 4, 8, 16],
        [0, 2, 4, 8, 16, 32],
        [0, 1, 2, 4, 8, 16, 32],
        [0, 3, 6, 12, 24],
        [0, 1, 3, 9, 27, 81],
        [0, 1, 2, 5, 12, 29],
        [0, 1, 3, 7, 15, 31, 63, 127],
    ]
    for U in structured:
        s = calc_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Strategy 2: Systematic local search from best structured sets
    def local_search(U, max_iterations=200):
        """Try all single-element replacements and additions/deletions."""
        current = sorted(U)
        current_score = calc_score(current)
        improved = True
        while improved and max_iterations > 0:
            improved = False
            max_iterations -= 1
            
            # Try replacing each element (except 0) with a nearby value
            for i in range(1, len(current)):
                old_val = current[i]
                # Try values in a range around old_val
                search_range = max(1, old_val // 2)
                for new_val in range(1, max(1, old_val + search_range)):
                    if new_val in current:
                        continue
                    U_new = current[:]
                    U_new[i] = new_val
                    U_new = sorted(U_new)
                    s = calc_score(U_new)
                    if s > current_score + 1e-12:
                        current = U_new
                        current_score = s
                        improved = True
                        max_iterations -= 1
                        if max_iterations <= 0:
                            break
                if max_iterations <= 0:
                    break
            
            # Try adding a new element
            if len(current) < 12:
                max_val = current[-1]
                for new_val in range(1, max_val * 2 + 2):
                    if new_val in current:
                        continue
                    U_new = sorted(current + [new_val])
                    s = calc_score(U_new)
                    if s > current_score + 1e-12:
                        current = U_new
                        current_score = s
                        improved = True
                        max_iterations -= 1
                        if max_iterations <= 0:
                            break
                if max_iterations <= 0:
                    break
            
            # Try removing an element (except 0)
            if len(current) > 2:
                for i in range(1, len(current)):
                    U_new = current[:i] + current[i+1:]
                    s = calc_score(U_new)
                    if s > current_score + 1e-12:
                        current = U_new
                        current_score = s
                        improved = True
                        max_iterations -= 1
                        if max_iterations <= 0:
                            break
                if max_iterations <= 0:
                    break
        
        return current, current_score
    
    # Local search from top structured candidates
    top_candidates = []
    for U in structured:
        s = calc_score(U)
        top_candidates.append((s, U))
    top_candidates.sort(key=lambda x: -x[0])
    
    for _, U in top_candidates[:8]:
        if time.time() > deadline:
            break
        improved_set, improved_score = local_search(U, max_iterations=300)
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved_set
    
    # Strategy 3: Random search with local improvement
    random.seed(42)
    
    max_k = 8
    max_m = 60
    samples_per_config = 300
    
    for k in range(3, max_k + 1):
        if time.time() > deadline:
            break
        for m in range(5, max_m + 1):
            if time.time() > deadline:
                break
            
            for _ in range(samples_per_config):
                if time.time() > deadline:
                    break
                
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
                if k > 1:
                    idx = random.randint(1, k - 1)
                    new_val = random.randint(1, m + 15)
                    if new_val not in U:
                        U_new = U[:]
                        U_new[idx] = new_val
                        U_new.sort()
                        s_new = calc_score(U_new)
                        if s_new > best_score:
                            best_score = s_new
                            best_set = U_new[:]
    
    # Final polish: local search from the best found so far
    if time.time() < deadline - 10.0:
        final_set, final_score = local_search(best_set, max_iterations=500)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set

    return best_set
# EVOLVE-BLOCK-END
