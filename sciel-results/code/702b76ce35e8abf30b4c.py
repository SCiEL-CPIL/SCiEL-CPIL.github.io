import random
import time
from itertools import combinations

def construct():
    N = 192
    MAX_SIZE = 48
    
    def is_valid_addition(S, x):
        # S is a list of integers
        # 1. Check if x divides sum of any 3 elements in S
        if len(S) >= 3:
            for t in combinations(S, 3):
                if sum(t) % x == 0:
                    return False
        
        # 2. Check if any a in S divides sum of x and any 2 other elements in S
        if len(S) >= 2:
            for a in S:
                others = [v for v in S if v != a]
                if len(others) < 2:
                    continue
                for b, c in combinations(others, 2):
                    if (b + c + x) % a == 0:
                        return False
        return True

    def is_set_valid(S):
        S_list = list(S)
        for a in S_list:
            others = [v for v in S_list if v != a]
            if len(others) >= 3:
                for t in combinations(others, 3):
                    if sum(t) % a == 0:
                        return False
        return True

    def greedy_init(order):
        current_set = []
        for x in order:
            if is_valid_addition(current_set, x):
                current_set.append(x)
            if len(current_set) >= MAX_SIZE:
                break
        return current_set

    def local_search(current_set, time_limit):
        best_set = list(current_set)
        best_size = len(best_set)
        all_numbers = set(range(1, N + 1))
        candidates = list(all_numbers - set(current_set))
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try Add
            if len(current_set) < MAX_SIZE and candidates:
                random.shuffle(candidates)
                for x in candidates:
                    if is_valid_addition(current_set, x):
                        current_set.append(x)
                        candidates.remove(x)
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = list(current_set)
                        improved = True
                        break
            
            # 2. Try Swap
            if not improved and len(current_set) > 0 and candidates:
                u = random.choice(current_set)
                random.shuffle(candidates)
                for v in candidates:
                    new_set = [x for x in current_set if x != u]
                    if is_valid_addition(new_set, v):
                        current_set = new_set + [v]
                        candidates.remove(v)
                        candidates.append(u)
                        improved = True
                        break
                        
            # 3. Try 2-Swap (Remove u, v; Add w, z)
            if not improved and len(current_set) >= 2 and len(candidates) >= 2:
                # Pick two to remove
                u, v = random.sample(current_set, 2)
                # Pick two to add
                w, z = random.sample(candidates, 2)
                
                # Check if removing u, v and adding w, z is valid
                # This is expensive, so we only do it occasionally or with a limit
                # To keep it fast, we check validity of the resulting set directly
                # But is_set_valid is O(N^3). 
                # We can do a quick check:
                # 1. w, z don't conflict with each other? (They are distinct, so sum of 3 not possible from just them)
                # 2. w, z don't conflict with remaining set?
                # 3. Remaining set doesn't conflict with w, z?
                
                remaining = [x for x in current_set if x != u and x != v]
                new_set = remaining + [w, z]
                
                # Only do full check if size is promising or random chance
                if random.random() < 0.1 or len(new_set) >= best_size:
                     if is_set_valid(new_set):
                         current_set = new_set
                         candidates.remove(w)
                         candidates.remove(z)
                         candidates.append(u)
                         candidates.append(v)
                         if len(current_set) > best_size:
                             best_size = len(current_set)
                             best_set = list(current_set)
                         improved = True

            if not improved:
                # Stuck, try a random perturbation or break to restart
                # For now, just break to allow outer loop to restart if we add that
                pass
                
        return best_set

    # Main Loop: Multiple Restarts
    start_time = time.time()
    time_limit = 58.0
    best_global_set = []
    
    # Try a few different initialization strategies
    strategies = [
        list(range(N, 0, -1)), # Descending
        list(range(1, N + 1)), # Ascending
        [x for x in range(N, 0, -1) if x % 2 != 0], # Odd descending
        [x for x in range(N, 0, -1) if x % 2 == 0], # Even descending
    ]
    
    # Add random shuffles
    for _ in range(5):
        s = list(range(1, N + 1))
        random.shuffle(s)
        strategies.append(s)
    
    for strat in strategies:
        if time.time() - start_time > time_limit:
            break
        init_set = greedy_init(strat)
        # Allocate remaining time to local search, e.g., 5 seconds per restart
        # Or distribute evenly
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            break
        time_for_ls = min(5.0, remaining / 2.0) # Conservative
        ls_set = local_search(init_set, time_for_ls)
        
        if len(ls_set) > len(best_global_set):
            best_global_set = ls_set
            
    return best_global_set
