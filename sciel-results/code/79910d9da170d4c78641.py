import time
import math
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin: evaluator kills at 90s
    
    N = 22
    candidates = list(range(1, N + 1))
    
    def is_dissociated(S):
        """Exact verification using integer arithmetic."""
        if not S:
            return True
        L = 1
        for x in S:
            L = math.lcm(L, x)
        vals = [L // x for x in S]
        k = len(S)
        
        sums = {0}
        for v in vals:
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums.update(new_sums)
        
        return len(sums) == (1 << k)
    
    def try_greedy(order):
        S = []
        L = 1
        sums = {0}
        
        for x in order:
            L_new = math.lcm(L, x)
            w_new = L_new // x
            scale = L_new // L
            
            collision = False
            
            if L % x != 0:
                collision = False
            else:
                for s in sums:
                    if (s + w_new) in sums:
                        collision = True
                        break
            
            if not collision:
                S.append(x)
                L = L_new
                if scale == 1:
                    new_sums = set(sums)
                    for s in sums:
                        new_sums.add(s + w_new)
                    sums = new_sums
                else:
                    scaled_sums = {s * scale for s in sums}
                    new_sums = set(scaled_sums)
                    for s in scaled_sums:
                        new_sums.add(s + w_new)
                    sums = new_sums
                
                if time.time() - start_time > timeout - 2:
                    break
        return S
    
    best_set = []
    
    # 1. Try descending order
    order_desc = sorted(candidates, reverse=True)
    s1 = try_greedy(order_desc)
    if len(s1) > len(best_set):
        best_set = s1
        
    # 2. Try ascending order
    order_asc = sorted(candidates)
    s2 = try_greedy(order_asc)
    if len(s2) > len(best_set):
        best_set = s2
    
    # 3. Try random orders with more restarts
    max_restarts = 3000
    for i in range(max_restarts):
        if time.time() - start_time > timeout - 20:
            break
        random.shuffle(candidates)
        s_r = try_greedy(candidates)
        if len(s_r) > len(best_set):
            best_set = s_r
    
    # 4. Local search: try to add missing elements
    current_best_size = len(best_set)
    missing = set(candidates) - set(best_set)
    improved = True
    while improved and time.time() - start_time < timeout - 25:
        improved = False
        for x in list(missing):
            test_set = best_set + [x]
            if is_dissociated(test_set):
                best_set = test_set
                missing.remove(x)
                improved = True
                current_best_size += 1
                break
    
    # 5. Try to find a larger set by checking subsets of size current_best_size + 1
    if time.time() - start_time < timeout - 35:
        target_size = current_best_size + 1
        if target_size <= 22:
            best_set_set = set(best_set)
            found = False
            for combo in combinations(candidates, target_size):
                if time.time() - start_time > timeout - 5:
                    break
                # Quick filter: check if this combo shares at least target_size - 2 elements with best_set
                overlap = len(set(combo) & best_set_set)
                if overlap < target_size - 2:
                    continue
                if is_dissociated(list(combo)):
                    best_set = list(combo)
                    current_best_size = target_size
                    found = True
                    break
            if found:
                # Continue trying to add more
                missing = set(candidates) - set(best_set)
                improved = True
                while improved and time.time() - start_time < timeout - 10:
                    improved = False
                    for x in list(missing):
                        test_set = best_set + [x]
                        if is_dissociated(test_set):
                            best_set = test_set
                            missing.remove(x)
                            improved = True
                            current_best_size += 1
                            break
    
    if not best_set:
        best_set = [1]
        
    return best_set
