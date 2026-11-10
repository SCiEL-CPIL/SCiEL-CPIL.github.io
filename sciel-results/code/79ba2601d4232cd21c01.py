"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _all_block_sums(values):
    """Compute all non-empty consecutive block sums for a sequence."""
    n = len(values)
    sums = []
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            sums.append(total)
    return sums

def _count_duplicates(sums_list):
    """Count the number of duplicate sums (excess over unique count)."""
    if not sums_list:
        return 0
    return len(sums_list) - len(set(sums_list))

def _is_valid_sequence(values):
    """Check if sequence is strictly increasing in [1,256] with all distinct block sums."""
    if not values:
        return True
    if values[-1] > 256 or values[0] < 1:
        return False
    for i in range(1, len(values)):
        if values[i] <= values[i-1]:
            return False
    sums = _all_block_sums(values)
    return len(sums) == len(set(sums))

def _greedy_baseline():
    """Deterministic greedy: pick smallest valid next element."""
    values = []
    sums_set = set()
    for x in range(1, 257):
        if values and x <= values[-1]:
            continue
        # Check if adding x creates duplicate sums
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if sums_set.intersection(tails):
            continue
        values.append(x)
        sums_set.update(tails)
    return values

def _simulated_annealing(seed, deadline, initial_values=None):
    """
    Simulated annealing on difference vectors.
    State: sequence of values (strictly increasing in [1,256]).
    Objective: minimize duplicate block sums, then maximize length.
    """
    rng = random.Random(seed)
    
    # Initialize from greedy or random valid prefix
    if initial_values is None:
        values = _greedy_baseline()
    else:
        values = list(initial_values)
    
    if not values:
        return []
    
    best_values = list(values)
    best_sums = _all_block_sums(values)
    best_dup = _count_duplicates(best_sums)
    best_len = len(values)
    
    # Temperature schedule
    T_start = 50.0
    T_end = 0.01
    n_iter = 50000
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        T = T_start * (T_end / T_start) ** (it / n_iter)
        
        # Generate a perturbation
        move_type = rng.random()
        new_values = list(values)
        
        if move_type < 0.3 and len(values) > 1:
            # Remove a random element (not first or last to keep increasing)
            idx = rng.randint(1, len(values) - 1)
            new_values.pop(idx)
        elif move_type < 0.5:
            # Insert a new element at a random position
            if len(new_values) < 256:
                # Find a position to insert
                pos = rng.randint(0, len(new_values))
                if pos == 0:
                    lo, hi = 1, (new_values[0] - 1) if new_values else 256
                elif pos == len(new_values):
                    lo, hi = (new_values[-1] + 1), 256
                else:
                    lo, hi = (new_values[pos-1] + 1), (new_values[pos] - 1)
                if lo <= hi:
                    x = rng.randint(lo, hi)
                    new_values.insert(pos, x)
        elif move_type < 0.8:
            # Modify a random element
            idx = rng.randint(0, len(new_values) - 1)
            if idx == 0:
                lo, hi = 1, (new_values[1] - 1) if len(new_values) > 1 else 256
            elif idx == len(new_values) - 1:
                lo, hi = (new_values[idx-1] + 1), 256
            else:
                lo, hi = (new_values[idx-1] + 1), (new_values[idx+1] - 1)
            if lo <= hi:
                new_values[idx] = rng.randint(lo, hi)
        else:
            # Swap two adjacent elements and shift (complex, skip for now)
            # Instead: perturb a random element by a small amount
            idx = rng.randint(0, len(new_values) - 1)
            delta = rng.choice([-1, 1, -2, 2])
            new_val = new_values[idx] + delta
            if idx == 0:
                valid = 1 <= new_val < (new_values[1] if len(new_values) > 1 else 257)
            elif idx == len(new_values) - 1:
                valid = (new_values[idx-1] + 1) <= new_val <= 256
            else:
                valid = (new_values[idx-1] + 1) <= new_val <= (new_values[idx+1] - 1)
            if valid:
                new_values[idx] = new_val
            else:
                continue
        
        # Validate strict increasing
        valid_seq = True
        for i in range(1, len(new_values)):
            if new_values[i] <= new_values[i-1]:
                valid_seq = False
                break
        if not valid_seq:
            continue
        if new_values[0] < 1 or new_values[-1] > 256:
            continue
        
        # Compute objective
        new_sums = _all_block_sums(new_values)
        new_dup = _count_duplicates(new_sums)
        new_len = len(new_values)
        
        # Combined objective: primary = duplicates, secondary = -length
        old_obj = best_dup - 0.001 * best_len
        new_obj = new_dup - 0.001 * new_len
        
        delta_obj = new_obj - old_obj
        
        if delta_obj < 0 or (T > 0 and rng.random() < math.exp(-delta_obj / T)):
            values = new_values
            if new_dup < best_dup or (new_dup == best_dup and new_len > best_len):
                best_values = list(new_values)
                best_dup = new_dup
                best_len = new_len
    
    return best_values

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline, safe margin for 90s limit
    
    # Get baseline
    best = _greedy_baseline()
    best_len = len(best)
    
    # Run simulated annealing with multiple restarts
    seed = 0
    max_restarts = 50
    for _ in range(max_restarts):
        if time.time() > deadline - 5:
            break
        result = _simulated_annealing(seed, deadline, initial_values=None)
        if len(result) > best_len:
            best = result
            best_len = len(result)
        seed += 1
    
    # Also try SA starting from the baseline with different seeds
    for _ in range(max_restarts):
        if time.time() > deadline - 2:
            break
        result = _simulated_annealing(seed, deadline, initial_values=best)
        if len(result) > best_len:
            best = result
            best_len = len(result)
        seed += 1
    
    return best
# EVOLVE-BLOCK-END
