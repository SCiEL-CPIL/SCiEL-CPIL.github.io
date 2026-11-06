import time
import random

def construct():
    fixed = {1, 2, 5, 11}
    N = 284
    
    def init_sums(S):
        sums = set()
        s_list = sorted(S)
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                sums.add(s_list[i] + s_list[j])
        return sums

    def is_valid_add(S, sums, x):
        if x in S:
            return False
        if 2 * x in sums:
            return False
        for y in S:
            if x + y in sums:
                return False
        return True

    def add_to_set(S, sums, x):
        S.add(x)
        sums.add(2 * x)
        for y in S:
            if y != x:
                sums.add(x + y)
                
    def remove_from_set(S, sums, x):
        S.remove(x)
        sums.discard(2 * x)
        for y in S:
            sums.discard(x + y)

    def greedy_init(strategy):
        S = set(fixed)
        sums = init_sums(S)
        
        if strategy == 'small':
            candidates = list(range(1, N + 1))
        elif strategy == 'large':
            candidates = list(range(N, 0, -1))
        else:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            
        for x in candidates:
            if is_valid_add(S, sums, x):
                add_to_set(S, sums, x)
        return S, sums

    best_set = set(fixed)
    best_size = len(fixed)
    start_time = time.time()
    deadline = start_time + 115.0
    
    strategies = ['small', 'large', 'random', 'random', 'random', 'random']
    
    for strat in strategies:
        if time.time() > deadline:
            break
            
        S, sums = greedy_init(strat)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
        # Local search
        stuck_counter = 0
        
        while time.time() < deadline:
            # Try to add
            x = random.randint(1, N)
            if x not in S and is_valid_add(S, sums, x):
                add_to_set(S, sums, x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
                stuck_counter = 0
                continue
            
            # Try to swap
            removable = [y for y in S if y not in fixed]
            if not removable:
                break
            
            y = random.choice(removable)
            
            # Check if swap is possible without modifying state yet?
            # is_valid_add checks against current S.
            # We need to check if (S - {y}) U {x} is valid.
            # This is equivalent to:
            # 1. x is not in S - {y} (x != y, x not in S)
            # 2. 2x not in sums - sums_related_to_y
            # 3. x+z not in sums - sums_related_to_y for all z in S - {y}
            
            # Easier to just do it tentatively
            remove_from_set(S, sums, y)
            
            if x not in S and is_valid_add(S, sums, x):
                add_to_set(S, sums, x)
                stuck_counter = 0
            else:
                # Revert
                add_to_set(S, sums, y)
                stuck_counter += 1
                
            if stuck_counter > 2000:
                # Restart local search with a new random perturbation of the current set?
                # Or just break to next strategy
                break

    return list(best_set)
