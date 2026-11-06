import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    n = 256
    
    def is_valid_set(S):
        # S is a set of integers
        # Check: for all a < b <= c in S, a does not divide b+c
        S_list = list(S)
        S_list.sort()
        for i, a in enumerate(S_list):
            # b must be > a
            # c must be >= b
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(x, current_set):
        # Check if x can be added to current_set while maintaining validity
        # x is smaller than all elements in current_set (assuming we add in descending order)
        # So x acts as 'a'. We need x not to divide b+c for any b,c in current_set with b<=c.
        c_list = list(current_set)
        for i, b in enumerate(c_list):
            for k in range(i, len(c_list)):
                c = c_list[k]
                if (b + c) % x == 0:
                    return False
        return True

    def greedy_expand(base_set, candidates):
        # Greedily add candidates to base_set
        current_set = set(base_set)
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, current_set):
                current_set.add(x)
        return current_set

    def local_search_polish(current_set):
        # Try to improve by swapping elements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any element from 1 to n that isn't in the set
            for x in range(1, n + 1):
                if time.time() > deadline:
                    return current_set
                if x not in current_set:
                    if can_add(x, current_set):
                        current_set.add(x)
                        improved = True
                        break
        return current_set

    # 1. Find the best interval [L, 256]
    best_L = 1
    best_size = 0
    best_set = set()
    
    for L in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Quick check for a=L
        k_min = (2 * (L + 1) + L - 1) // L
        k_max = (2 * n) // L
        if k_min <= k_max:
            continue
        
        # If L passes the quick check, verify full validity for [L, 256]
        S_test = set(range(L, n + 1))
        if is_valid_set(S_test):
            best_L = L
            best_size = n - L + 1
            best_set = S_test
            break

    if not best_set:
        return list(range(192, 257))

    # 2. Multi-start greedy with random restarts
    best_result = set()
    best_cardinality = 0
    
    # First, do the deterministic greedy expansion
    candidates = list(range(best_L - 1, 0, -1))
    result = greedy_expand(best_set, candidates)
    result = local_search_polish(result)
    if len(result) > best_cardinality:
        best_cardinality = len(result)
        best_result = set(result)
    
    # Now do random restarts with shuffled candidate order
    random.seed(42)
    restart_count = 0
    max_restarts = 50
    
    while time.time() < deadline and restart_count < max_restarts:
        restart_count += 1
        
        # Shuffle the candidates
        shuffled_candidates = list(range(best_L - 1, 0, -1))
        random.shuffle(shuffled_candidates)
        
        result = greedy_expand(best_set, shuffled_candidates)
        result = local_search_polish(result)
        
        if len(result) > best_cardinality:
            best_cardinality = len(result)
            best_result = set(result)
    
    # 3. Try some additional local search strategies
    # Try to swap: remove one element, try to add two others
    if time.time() < deadline:
        current = set(best_result)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each element and see if we can add more
            for y in sorted(current, reverse=True):
                if time.time() > deadline:
                    break
                trial = set(current)
                trial.discard(y)
                # Try to add elements that were previously blocked
                for x in range(1, n + 1):
                    if time.time() > deadline:
                        break
                    if x not in trial and x != y:
                        if can_add(x, trial):
                            trial.add(x)
                            if len(trial) > len(current):
                                current = trial
                                improved = True
                                break
    
    return list(current)

# EVOLVE-BLOCK-END
