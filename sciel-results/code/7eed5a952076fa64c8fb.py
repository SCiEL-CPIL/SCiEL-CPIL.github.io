import random
import time
import math

def construct():
    mandatory = {1, 2, 5, 11}
    max_val = 284
    all_numbers = set(range(1, max_val + 1))
    
    # Initialize with mandatory elements
    current_set = set(mandatory)
    
    def get_sums(S):
        sums = set()
        L = list(S)
        for i in range(len(L)):
            for j in range(i, len(L)):
                sums.add(L[i] + L[j])
        return sums

    current_sums = get_sums(current_set)
    candidates = all_numbers - current_set
    
    # Simulated Annealing parameters
    start_temp = 5.0
    end_temp = 0.01
    cooling_rate = 0.999
    max_iterations = 50000
    start_time = time.time()
    time_limit = 110.0
    
    best_set = set(current_set)
    best_size = len(current_set)
    
    temp = start_temp
    
    def try_add(x, S, sums):
        """Returns (success, new_sums) if x can be added to S."""
        for y in S:
            if x + y in sums:
                return False, sums
        if 2 * x in sums:
            return False, sums
        new_sums = sums | {x + y for y in S} | {2 * x}
        return True, new_sums

    def try_remove(x, S, sums):
        """Returns (new_S, new_sums) after removing x from S."""
        sums_to_remove = {x + y for y in S}
        new_sums = sums - sums_to_remove
        new_S = S - {x}
        return new_S, new_sums

    for i in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
            
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_set = None
        new_sums = None
        cand_change = None # (to_remove, to_add) or None
        
        if move_type == 'add':
            if candidates:
                x = random.choice(list(candidates))
                success, ns = try_add(x, current_set, current_sums)
                if success:
                    new_set = current_set | {x}
                    new_sums = ns
                    cand_change = (x, None)
        
        elif move_type == 'remove':
            removable = [x for x in current_set if x not in mandatory]
            if removable:
                x = random.choice(removable)
                new_set, new_sums = try_remove(x, current_set, current_sums)
                cand_change = (None, x)
        
        elif move_type == 'swap':
            removable = [x for x in current_set if x not in mandatory]
            if removable and candidates:
                x = random.choice(removable)
                y = random.choice(list(candidates))
                temp_set, temp_sums = try_remove(x, current_set, current_sums)
                success, final_sums = try_add(y, temp_set, temp_sums)
                if success:
                    new_set = temp_set | {y}
                    new_sums = final_sums
                    cand_change = (y, x)

        if new_set is None:
            continue

        old_size = len(current_set)
        new_size = len(new_set)
        
        delta = old_size - new_size # positive if worse (smaller)
        
        accepted = False
        if delta <= 0:
            accepted = True
        else:
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                accepted = True

        if accepted:
            current_set = new_set
            current_sums = new_sums
            if cand_change:
                to_remove, to_add = cand_change
                if to_remove:
                    candidates.discard(to_remove)
                if to_add:
                    candidates.add(to_add)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)

        temp *= cooling_rate
        if temp < end_temp:
            temp = end_temp

    return sorted(best_set)
