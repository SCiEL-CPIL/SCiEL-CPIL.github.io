import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 58.0
    
    # Precompute LCM and targets
    # LCM(1..24)
    lcm_all = 53542288800
    targets = {i: lcm_all // i for i in range(1, 25)}
    
    def can_make_sum(target, vals):
        # MITM
        n = len(vals)
        if n == 0:
            return target == 0
        if target == 0:
            return True
        if sum(vals) < target:
            return False
        
        mid = n // 2
        left = vals[:mid]
        right = vals[mid:]
        
        left_sums = set([0])
        for v in left:
            new_sums = set()
            for s in left_sums:
                new_sums.add(s + v)
            left_sums.update(new_sums)
            
        right_sums = set([0])
        for v in right:
            new_sums = set()
            for s in right_sums:
                new_sums.add(s + v)
            right_sums.update(new_sums)
            
        for s in left_sums:
            if (target - s) in right_sums:
                return True
        return False

    def is_valid(S):
        S_list = list(S)
        for x in S_list:
            target = targets[x]
            others = [targets[y] for y in S_list if y != x]
            if can_make_sum(target, others):
                return False
        return True

    best_set = set(range(13, 25))
    best_size = len(best_set)
    
    # Greedy add
    current_set = set(best_set)
    candidates = list(range(1, 13))
    
    while True:
        added = False
        for x in candidates:
            if x in current_set:
                continue
            if time.time() > deadline:
                return list(best_set)
            new_set = current_set | {x}
            if is_valid(new_set):
                current_set = new_set
                added = True
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(best_set)
                break # Restart candidate loop to try next
        if not added:
            break

    # Local Search: Swaps
    found_improvement = True
    while found_improvement and time.time() < deadline:
        found_improvement = False
        
        # Identify large elements (13-24) and small elements (1-12)
        large_elements = [x for x in current_set if x >= 13]
        small_candidates = [x for x in range(1, 13) if x not in current_set]
        
        if not large_elements or not small_candidates:
            break
            
        # Try removing each large element
        for L in large_elements:
            if time.time() > deadline:
                break
                
            reduced_set = current_set - {L}
            
            # Try to add small candidates to temp_set
            temp_set = set(reduced_set)
            
            # Greedily add small candidates to temp_set
            for s in small_candidates:
                if s in temp_set:
                    continue
                new_set = temp_set | {s}
                if is_valid(new_set):
                    temp_set = new_set
                    
            # If we added more than 1 (since we removed 1, net gain > 0 if added > 1)
            # Actually, we removed 1. If we add 2, net gain +1. If we add 1, net gain 0.
            if len(temp_set) > len(current_set):
                current_set = set(temp_set)
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                found_improvement = True
                break # Restart local search loop

    # Random Restarts
    for _ in range(200):
        if time.time() > deadline:
            break
        current_set = set(best_set)
        if current_set:
            x_to_remove = random.choice(list(current_set))
            current_set.remove(x_to_remove)
        
        # Greedy add
        for x in range(1, 25):
            if x in current_set:
                continue
            new_set = current_set | {x}
            if is_valid(new_set):
                current_set = new_set
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)

    return list(best_set)
