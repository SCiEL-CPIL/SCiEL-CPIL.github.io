import random
import time

def construct():
    initial = [1, 2, 5, 11]
    best_set = None
    best_size = 0
    
    start_time = time.time()
    time_limit = 115.0
    
    def compute_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def is_valid_addition(S, x, sums):
        if 2 * x in sums:
            return False
        for y in S:
            if x + y in sums:
                return False
        return True

    def try_add(S, sums):
        candidates = [x for x in range(1, 285) if x not in S]
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_addition(S, x, sums):
                S.append(x)
                sums.add(2 * x)
                for y in S[:-1]:
                    sums.add(x + y)
                return True
        return False

    # Simulated Annealing / Local Search
    current_set = list(initial)
    current_sums = compute_sums(current_set)
    
    # First, greedily fill as much as possible
    while try_add(current_set, current_sums):
        pass
        
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set[:]

    temperature = 50.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to add a random element (if valid)
            candidates = [x for x in range(1, 285) if x not in current_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            if is_valid_addition(current_set, x, current_sums):
                old_set = current_set[:]
                old_sums = current_sums.copy()
                
                current_set.append(x)
                current_sums.add(2 * x)
                for y in current_set[:-1]:
                    current_sums.add(x + y)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
            else:
                # If invalid, we might want to accept it with some probability? 
                # No, the constraint is hard. We can only move to valid states.
                # So this move is rejected.
                pass
                
        elif move_type < 0.6:
            # Remove a random non-prescribed element
            removable = [x for x in current_set if x not in initial]
            if not removable:
                continue
            x = random.choice(removable)
            old_set = current_set[:]
            old_sums = current_sums.copy()
            
            # Remove x
            idx = current_set.index(x)
            current_set.pop(idx)
            # Recalculate sums? Or update? 
            # Recalculating is O(N^2). N is small (~20). 
            current_sums = compute_sums(current_set)
            
            # After removal, try to add something else? 
            # Or just keep it smaller temporarily?
            # Let's try to immediately refill to maintain size.
            # This is a "kick"
            for _ in range(3):
                if not try_add(current_set, current_sums):
                    break
                    
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
                
        else:
            # Swap: Remove one non-prescribed, add one new
            removable = [x for x in current_set if x not in initial]
            if not removable:
                continue
            x = random.choice(removable)
            
            # Find a candidate y to swap with
            candidates = [y for y in range(1, 285) if y not in current_set and y != x]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            # Check if swapping x for y is valid
            # New set = (current_set - {x}) U {y}
            temp_set = [z for z in current_set if z != x]
            if is_valid_addition(temp_set, y, compute_sums(temp_set)):
                # Perform swap
                idx = current_set.index(x)
                current_set[idx] = y
                current_set.sort()
                current_sums = compute_sums(current_set)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]

        # Cool down
        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 0.1

    return best_set
