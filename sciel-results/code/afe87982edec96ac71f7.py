import random
import time
import math

def construct():
    prescribed = [1, 2, 5, 11]
    prescribed.sort()
    
    # Pre-compute sums for prescribed
    pres_sums = set()
    for i in range(len(prescribed)):
        for j in range(i, len(prescribed)):
            pres_sums.add(prescribed[i] + prescribed[j])
            
    best_set = list(prescribed)
    best_size = len(prescribed)
    
    start_time = time.time()
    time_limit = 115.0
    
    # Initialize current state
    current_set = list(prescribed)
    current_sums = set(pres_sums)
    
    # Temperature schedule
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        # Generate a random move
        # We prefer adding if possible, but allow removing to escape local minima
        # Also allow swapping
        
        move_type = random.choices(['add', 'remove', 'swap'], weights=[0.5, 0.2, 0.3])[0]
        
        new_set = list(current_set)
        new_sums = set(current_sums)
        
        move_valid = False
        delta_size = 0
        
        if move_type == 'add':
            # Find valid candidates
            valid_candidates = []
            for x in range(1, 285):
                if x in current_set:
                    continue
                if 2 * x in current_sums:
                    continue
                conflict = False
                for y in current_set:
                    if (x + y) in current_sums:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)
            
            if valid_candidates:
                x = random.choice(valid_candidates)
                new_set.append(x)
                for y in current_set:
                    new_sums.add(x + y)
                new_sums.add(2 * x)
                move_valid = True
                delta_size = 1
        elif move_type == 'remove':
            non_pres = [x for x in current_set if x not in prescribed]
            if non_pres:
                x = random.choice(non_pres)
                new_set.remove(x)
                # Remove sums involving x
                sums_to_remove = {2 * x}
                for y in current_set:
                    if y != x:
                        sums_to_remove.add(x + y)
                for s in sums_to_remove:
                    new_sums.discard(s)
                move_valid = True
                delta_size = -1
        elif move_type == 'swap':
            non_pres = [x for x in current_set if x not in prescribed]
            if non_pres:
                x_to_remove = random.choice(non_pres)
                # 1. Remove x_to_remove from new_set and new_sums
                new_set.remove(x_to_remove)
                sums_to_remove = {2 * x_to_remove}
                for y in current_set:
                    if y != x_to_remove:
                        sums_to_remove.add(x_to_remove + y)
                for s in sums_to_remove:
                    new_sums.discard(s)
                
                # 2. Find valid y to add to this new state
                valid_candidates = []
                for y in range(1, 285):
                    if y in new_set:
                        continue
                    if 2 * y in new_sums:
                        continue
                    conflict = False
                    for z in new_set:
                        if (y + z) in new_sums:
                            conflict = True
                            break
                    if not conflict:
                        valid_candidates.append(y)
                
                if valid_candidates:
                    y_to_add = random.choice(valid_candidates)
                    new_set.append(y_to_add)
                    for z in new_set[:-1]:
                        new_sums.add(y_to_add + z)
                    new_sums.add(2 * y_to_add)
                    move_valid = True
                    delta_size = 0 # Size unchanged
        
        if not move_valid:
            continue
            
        # Acceptance criterion
        # We want to maximize size.
        # If delta_size > 0, always accept.
        # If delta_size < 0, accept with probability exp(delta_size / T) ? 
        # Standard SA: minimize Energy. Energy = -Size.
        # Delta_E = -Delta_Size.
        # If Delta_Size > 0, Delta_E < 0, accept.
        # If Delta_Size < 0, Delta_E > 0, accept with prob exp(-Delta_E/T) = exp(Delta_Size/T).
        # If Delta_Size == 0, accept with prob 1 (or exp(0)=1).
        
        if delta_size > 0:
            accept = True
        elif delta_size < 0:
            # delta_size is -1
            accept_prob = math.exp(delta_size / T)
            accept = random.random() < accept_prob
        else:
            accept = True
            
        if accept:
            current_set = new_set
            current_sums = new_sums
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
                if best_size == 20:
                    break
        else:
            # Revert is implicit as we didn't update current_set
            pass
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_set
