"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s kill (evaluator limit)
    N = 24
    universe = list(range(1, N + 1))
    
    # Precompute LCM of all numbers 1..24 for exact integer arithmetic
    total_lcm = 1
    for i in range(1, N + 1):
        total_lcm = lcm(total_lcm, i)
    
    # weight[i] = total_lcm // i
    weights = {i: total_lcm // i for i in universe}
    
    best_set = set()
    best_size = 0
    
    # Fast validity check using Meet-in-the-Middle
    def is_valid(s):
        s_list = list(s)
        n = len(s_list)
        if n <= 1:
            return True
        
        for i, a in enumerate(s_list):
            target = weights[a]
            others = [x for j, x in enumerate(s_list) if j != i]
            m = len(others)
            
            if m == 0:
                continue
                
            # Meet in the middle
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            left_sums = set()
            # Include 0 sum (empty subset)
            left_sums.add(0)
            for mask in range(1, 1 << len(left)):
                s_sum = 0
                for k in range(len(left)):
                    if mask & (1 << k):
                        s_sum += weights[left[k]]
                left_sums.add(s_sum)
            
            # Check if target can be formed by left_sum + right_sum
            # We need target = ls + rs => rs = target - ls
            # Iterate over right subsets
            for mask in range(1 << len(right)):
                r_sum = 0
                for k in range(len(right)):
                    if mask & (1 << k):
                        r_sum += weights[right[k]]
                if (target - r_sum) in left_sums:
                    return False
            
        return True

    def greedy_construct(order):
        current_set = []
        current_set_set = set()
        for elem in order:
            if elem in current_set_set:
                continue
            candidate_set = current_set_set | {elem}
            if is_valid(candidate_set):
                current_set.append(elem)
                current_set_set.add(elem)
        return current_set

    # 1. Deterministic Greedy: Descending (large elements are "smaller" reciprocally, safer)
    order_desc = sorted(universe, reverse=True)
    set_desc = greedy_construct(order_desc)
    if len(set_desc) > best_size:
        best_size = len(set_desc)
        best_set = set(set_desc)
        
    # 2. Deterministic Greedy: Ascending
    order_asc = sorted(universe)
    set_asc = greedy_construct(order_asc)
    if len(set_asc) > best_size:
        best_size = len(set_asc)
        best_set = set(set_asc)
        
    # 3. Randomized Greedy Restart
    # Use random shuffles to explore different configurations
    # Since is_valid is relatively fast for k < 20, we can do many restarts
    attempts = 0
    while time.time() < deadline:
        random.shuffle(universe)
        current_order = universe[:]
        # Slight bias: sometimes prioritize larger numbers first in the random sort?
        # Or just pure random.
        s = greedy_construct(current_order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
            # If we reach a very high size, we might stop early or keep going to find 24?
            # Max possible is likely < 24.
            
        attempts += 1
        # If we've done many attempts and not improving, maybe break?
        # No, keep going until deadline.
        
    return list(best_set)
# EVOLVE-BLOCK-END
