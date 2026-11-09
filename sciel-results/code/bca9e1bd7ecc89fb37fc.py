"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer for safety within 120s limit
    
    prescribed = [1, 2, 5, 11]
    fixed_set = set(prescribed)
    
    # Helper to check if a list is a valid Sidon set
    # Returns (is_valid, conflict_count)
    def check_sidon(candidates):
        n = len(candidates)
        if n != len(set(candidates)):
            return False, 100000
        sums = set()
        conflict = 0
        for i in range(n):
            for j in range(i, n):
                s = candidates[i] + candidates[j]
                if s in sums:
                    conflict += 1
                else:
                    sums.add(s)
        return conflict == 0, conflict

    # 1. Greedy Construction to get a strong baseline
    # The seed program does this, but we can make it more robust or just use it as the starting point.
    a = list(prescribed)
    sums = set()
    for i, x in enumerate(a):
        for y in a[i:]:
            sums.add(x + y)
    
    for x in range(12, 285):
        new = {x + y for y in a} | {2 * x}
        if len(new) == len(a) + 1 and not new & sums:
            a.append(x)
            sums |= new
            
    best_set = a
    best_size = len(best_set)
    
    # If we already hit the theoretical max (20), return immediately
    if best_size == 20:
        return best_set

    # 2. Local Search / Simulated Annealing to improve
    # We will try to find a set of size K, starting from best_size+1 up to 20.
    # Or we can just optimize the current set to see if we can fix conflicts if we force a larger size?
    # No, the current set is valid. We want to ADD elements.
    # Strategy: Try to find a valid set of size TargetSize.
    # We initialize with the prescribed elements and random others.
    
    def try_find_solution(target_size, time_limit):
        nonlocal start_time, deadline
        end_time = start_time + time_limit
        if end_time > deadline:
            end_time = deadline - 0.5
            
        if time.time() >= end_time:
            return None
            
        fixed_elements = set(prescribed)
        
        # We need to generate a random set of size target_size containing prescribed elements.
        # Then run simulated annealing to minimize conflicts.
        
        max_restarts = 50
        restarts = 0
        
        while time.time() < end_time and restarts < max_restarts:
            restarts += 1
            
            # Create initial guess
            current_set = list(prescribed)
            needed = target_size - len(current_set)
            
            if needed < 0:
                return None
                
            pool = [x for x in range(1, 285) if x not in fixed_elements]
            random.shuffle(pool)
            current_set.extend(pool[:needed])
            random.shuffle(current_set)
            
            # Identify fixed indices
            is_fixed = [x in fixed_elements for x in current_set]
            
            # Evaluate initial state
            is_valid, current_conflict = check_sidon(current_set)
            
            if is_valid:
                return current_set
            
            # Simulated Annealing
            temp = 10.0
            min_temp = 0.01
            cooling_rate = 0.9995
            max_steps = 200000
            
            for step in range(max_steps):
                if time.time() > end_time:
                    break
                    
                if temp < min_temp:
                    break
                    
                # Pick a random index to change
                idx = random.randint(0, len(current_set) - 1)
                
                if is_fixed[idx]:
                    continue
                    
                old_val = current_set[idx]
                # Choose a new random value from the pool
                # To be efficient, pick from 1..284
                new_val = random.randint(1, 284)
                
                if new_val == old_val:
                    continue
                    
                # Check if new_val is already in the set (excluding idx)
                # If it is, conflict will spike, which is fine, SA handles it.
                # But we can skip if we want to reduce obvious bad moves? 
                # Actually, allowing duplicates helps escape local minima sometimes, 
                # but check_sidon returns high conflict for duplicates.
                
                current_set[idx] = new_val
                
                # Evaluate new state
                is_valid, new_conflict = check_sidon(current_set)
                
                delta = new_conflict - current_conflict
                
                if delta <= 0:
                    current_conflict = new_conflict
                    if is_valid:
                        return current_set
                else:
                    # Accept worse solution with probability
                    # P = exp(-delta / temp)
                    try:
                        prob = math.exp(-delta / temp)
                    except OverflowError:
                        prob = 0.0
                    
                    if random.random() < prob:
                        current_conflict = new_conflict
                    else:
                        current_set[idx] = old_val
                        
                temp *= cooling_rate

        return None

    # Try to improve the size
    # We know the max is 20.
    # Let's try to find a set of size 20.
    # If that fails, try 19, etc.
    # But we already have a valid set of size best_size.
    # We should try to jump to 20 directly? Or step by step?
    # Jumping to 20 is harder but more rewarding.
    # Let's allocate time to try size 20 first.
    
    # Time allocation:
    # We have ~115s total.
    # Let's give 60s to try size 20.
    # If successful, return.
    # If not, try size 19 (if best_size < 19) with 30s.
    # If not, try size 18...
    
    # However, the greedy baseline might already be quite good.
    # Let's check best_size.
    
    # If best_size is close to 20, trying 20 is good.
    # If best_size is low (e.g. 10), trying 20 from scratch random restarts might be hard.
    # But the "pool" initialization is random.
    
    # A better strategy for larger jumps:
    # Start with the current best_set, add random elements to reach target_size, 
    # then fix conflicts via SA.
    # This is "seeded" SA.
    
    def try_find_solution_seeded(target_size, time_limit, seed_set):
        nonlocal start_time, deadline
        end_time = start_time + time_limit
        if end_time > deadline:
            end_time = deadline - 0.5
            
        if time.time() >= end_time:
            return None
            
        fixed_elements = set(prescribed)
        
        max_restarts = 20
        restarts = 0
        
        while time.time() < end_time and restarts < max_restarts:
            restarts += 1
            
            # Start with seed_set (which is valid)
            # If len(seed_set) > target_size, we can't use it directly as a base for growth?
            # No, we want a set of EXACTLY target_size.
            # If seed_set is smaller, we add randoms.
            # If seed_set is larger, we must remove elements? That's hard.
            # So we only use this if len(seed_set) <= target_size.
            
            if len(seed_set) > target_size:
                return None
                
            current_set = list(seed_set)
            needed = target_size - len(current_set)
            
            if needed < 0:
                return None
                
            pool = [x for x in range(1, 285) if x not in set(current_set)]
            random.shuffle(pool)
            current_set.extend(pool[:needed])
            random.shuffle(current_set)
            
            # Identify fixed indices (prescribed)
            is_fixed = [x in fixed_elements for x in current_set]
            
            # Evaluate initial state
            is_valid, current_conflict = check_sidon(current_set)
            
            if is_valid:
                return current_set
            
            # Simulated Annealing
            temp = 5.0
            min_temp = 0.01
            cooling_rate = 0.9995
            max_steps = 200000
            
            for step in range(max_steps):
                if time.time() > end_time:
                    break
                    
                if temp < min_temp:
                    break
                    
                # Pick a random index to change
                idx = random.randint(0, len(current_set) - 1)
                
                if is_fixed[idx]:
                    continue
                    
                old_val = current_set[idx]
                new_val = random.randint(1, 284)
                
                if new_val == old_val:
                    continue
                    
                current_set[idx] = new_val
                
                is_valid, new_conflict = check_sidon(current_set)
                
                delta = new_conflict - current_conflict
                
                if delta <= 0:
                    current_conflict = new_conflict
                    if is_valid:
                        return current_set
                else:
                    try:
                        prob = math.exp(-delta / temp)
                    except OverflowError:
                        prob = 0.0
                    
                    if random.random() < prob:
                        current_conflict = new_conflict
                    else:
                        current_set[idx] = old_val
                        
                temp *= cooling_rate

        return None

    # Try to reach 20
    if best_size < 20:
        res = try_find_solution_seeded(20, 60, best_set)
        if res:
            return res

    # If that failed, try to reach 19 (if we haven't already)
    if best_size < 19:
        res = try_find_solution_seeded(19, 30, best_set)
        if res:
            return res

    # If that failed, try to reach 18
    if best_size < 18:
        res = try_find_solution_seeded(18, 15, best_set)
        if res:
            return res

    # If nothing improved, return the best we have (the greedy one)
    # We could also try to just optimize the current set to ensure it's valid?
    # It is valid by construction.
    
    return best_set

# EVOLVE-BLOCK-END
