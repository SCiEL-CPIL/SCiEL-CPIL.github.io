import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s limit to be safe
    
    required = [1, 2, 5, 11]
    max_val = 284
    required_set = set(required)
    
    # Initialize sums array for current set
    def compute_sums(seq):
        sums = [False] * (2 * max_val + 2)
        for i in range(len(seq)):
            for j in range(i, len(seq)):
                s = seq[i] + seq[j]
                sums[s] = True
        return sums
    
    def is_valid_addition(seq, sums, x):
        """Check if x can be added to seq given current sums."""
        # Check x + y for all y in seq
        for y in seq:
            if sums[x + y]:
                return False
        # Check 2*x
        if sums[2 * x]:
            return False
        return True
    
    def add_to_set(seq, sums, x):
        """Add x to seq and update sums. Assumes validity already checked."""
        for y in seq:
            sums[x + y] = True
        sums[2 * x] = True
        seq.append(x)
    
    def remove_from_set(seq, sums, x):
        """Remove x from seq and recompute sums."""
        seq.remove(x)
        # Recompute sums from scratch (expensive but safe)
        new_sums = compute_sums(seq)
        # Copy back
        for i in range(len(sums)):
            sums[i] = new_sums[i]
    
    # Initial set
    curr_set = required[:]
    curr_sums = compute_sums(curr_set)
    
    best_set = curr_set[:]
    best_size = len(curr_set)
    best_sums = curr_sums[:]
    
    # Candidates
    candidates = [x for x in range(1, max_val + 1) if x not in required_set]
    
    # Simulated Annealing
    # Temperature schedule
    t_start = 100.0
    t_end = 0.1
    cooling_rate = 0.9995
    
    temperature = t_start
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a random candidate
            if not candidates:
                continue
            x = random.choice(candidates)
            if x in curr_set:
                continue
            if not is_valid_addition(curr_set, curr_sums, x):
                continue
            # Accept the add
            add_to_set(curr_set, curr_sums, x)
            delta = 1  # size increases by 1
            if len(curr_set) > best_size:
                best_size = len(curr_set)
                best_set = curr_set[:]
                best_sums = curr_sums[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        elif move_type < 0.6:
            # Remove a random non-required element
            removable = [x for x in curr_set if x not in required_set]
            if not removable:
                continue
            x = random.choice(removable)
            remove_from_set(curr_set, curr_sums, x)
            delta = -1
            no_improve_count += 1
        else:
            # Swap: remove a non-required, add a candidate
            removable = [x for x in curr_set if x not in required_set]
            if not removable or not candidates:
                continue
            x = random.choice(removable)
            # Try to find a candidate that can be added after removing x
            # First remove x
            old_sums_backup = curr_sums[:]
            remove_from_set(curr_set, curr_sums, x)
            
            # Now try to add a random candidate
            y = random.choice(candidates)
            if y in curr_set:
                # Restore
                curr_set.append(x)
                for i in range(len(curr_sums)):
                    curr_sums[i] = old_sums_backup[i]
                continue
            if not is_valid_addition(curr_set, curr_sums, y):
                # Restore
                curr_set.append(x)
                for i in range(len(curr_sums)):
                    curr_sums[i] = old_sums_backup[i]
                continue
            
            # Accept the swap
            add_to_set(curr_set, curr_sums, y)
            delta = 0  # size stays same
            no_improve_count += 1
        
        # Cooling
        temperature *= cooling_rate
        if temperature < t_end:
            temperature = t_end
        
        # Restart if stuck
        if no_improve_count > max_no_improve:
            # Reset to best and cool down
            curr_set = best_set[:]
            curr_sums = best_sums[:]
            temperature = max(temperature, t_start * 0.5)
            no_improve_count = 0
        
        # Early exit if we reach theoretical max
        if best_size >= 20:
            break
    
    return sorted(best_set)
