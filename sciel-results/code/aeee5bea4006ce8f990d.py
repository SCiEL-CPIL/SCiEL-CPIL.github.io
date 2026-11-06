"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

_DEADLINE = 85.0  # safety margin before 90s kill

def _get_block_sums(seq):
    """Get all consecutive block sums for a sequence."""
    sums = []
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.append(s)
    return sums

def _evaluate(seq):
    """
    Evaluate a sequence:
    - Must be strictly increasing
    - All values in [1, 256]
    - All consecutive block sums must be distinct
    
    Returns (valid, length, penalty) where:
    - valid: bool
    - length: len(seq)
    - penalty: number of constraint violations (0 if valid)
    """
    if not seq:
        return True, 0, 0
    
    # Check strictly increasing
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False, len(seq), 1000 + (len(seq) - i)  # heavy penalty
    
    # Check range
    for v in seq:
        if v < 1 or v > 256:
            return False, len(seq), 1000
    
    # Check distinct block sums
    sums = _get_block_sums(seq)
    n_sums = len(sums)
    unique_sums = len(set(sums))
    collisions = n_sums - unique_sums
    
    if collisions == 0:
        return True, len(seq), 0
    else:
        return False, len(seq), collisions * 10 + (n_sums - unique_sums)

def _greedy_build():
    """Original greedy strategy as a baseline."""
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _random_subset(size, low=1, high=256):
    """Generate a random strictly increasing sequence of given size."""
    if size <= 0:
        return []
    if size >= (high - low + 1):
        return list(range(low, high + 1))
    # Randomly choose 'size' distinct values from [low, high]
    chosen = random.sample(range(low, high + 1), size)
    chosen.sort()
    return chosen

def _local_search(seq, time_limit):
    """
    Simulated annealing to improve sequence length while maintaining validity.
    Strategy: 
    1. Start with a valid sequence (from greedy or random)
    2. Try to add elements (increase length) or swap elements
    3. Accept worse solutions with probability based on 'temperature'
    4. Gradually cool down
    """
    start_time = time.time()
    current_seq = list(seq)
    current_valid, current_len, current_penalty = _evaluate(current_seq)
    
    best_seq = list(current_seq)
    best_len = current_len
    best_penalty = current_penalty if current_valid else float('inf')
    
    # If not valid, try to repair
    if not current_valid:
        # Try greedy from scratch as fallback
        greedy = _greedy_build()
        if len(greedy) > current_len:
            current_seq = greedy
            current_valid, current_len, current_penalty = _evaluate(current_seq)
            best_seq = list(current_seq)
            best_len = current_len
    
    # Simulated annealing parameters
    T = 100.0  # initial temperature
    cooling = 0.995
    min_T = 0.1
    
    # Track best
    iteration = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit and T > min_T:
        iteration += 1
        
        # Generate neighbor
        neighbor = list(current_seq)
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a new element
            if len(neighbor) < 256:
                # Pick a random position and value
                pos = random.randint(0, len(neighbor))
                # Find a valid value to insert
                low = 1 if pos == 0 else neighbor[pos-1] + 1
                high = 256 if pos == len(neighbor) else neighbor[pos] - 1
                if low <= high:
                    val = random.randint(low, high)
                    neighbor.insert(pos, val)
                else:
                    # Can't insert, try swap instead
                    if len(neighbor) >= 2:
                        i = random.randint(0, len(neighbor) - 1)
                        j = random.randint(0, len(neighbor) - 1)
                        if i != j:
                            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            else:
                # Already max size, just swap
                if len(neighbor) >= 2:
                    i = random.randint(0, len(neighbor) - 1)
                    j = random.randint(0, len(neighbor) - 1)
                    if i != j:
                        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        elif move_type < 0.7:
            # Swap two elements
            if len(neighbor) >= 2:
                i = random.randint(0, len(neighbor) - 1)
                j = random.randint(0, len(neighbor) - 1)
                if i != j:
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        elif move_type < 0.85:
            # Remove an element (to escape bad local minima)
            if len(neighbor) > 1:
                pos = random.randint(0, len(neighbor) - 1)
                neighbor.pop(pos)
        
        else:
            # Replace an element with a random valid one
            if len(neighbor) >= 1:
                pos = random.randint(0, len(neighbor) - 1)
                low = 1 if pos == 0 else neighbor[pos-1] + 1
                high = 256 if pos == len(neighbor) else neighbor[pos] - 1
                if low <= high:
                    neighbor[pos] = random.randint(low, high)
        
        # Ensure strictly increasing after modifications
        # Re-sort if needed, but keep it simple: check validity
        neighbor_valid, neighbor_len, neighbor_penalty = _evaluate(neighbor)
        
        # Compute energy (we want to maximize length, minimize penalty)
        # Energy = -length * 100 + penalty (lower is better)
        current_energy = -current_len * 100 + (current_penalty if not current_valid else 0)
        neighbor_energy = -neighbor_len * 100 + (neighbor_penalty if not neighbor_valid else 0)
        
        delta_e = neighbor_energy - current_energy
        
        if delta_e <= 0 or random.random() < pow(2.718281828, -delta_e / T):
            current_seq = neighbor
            current_valid, current_len, current_penalty = neighbor_valid, neighbor_len, neighbor_penalty
            
            # Update best
            if current_valid and current_len > best_len:
                best_seq = list(current_seq)
                best_len = current_len
                best_penalty = 0
                no_improve_count = 0
            elif current_valid and current_len == best_len and current_penalty < best_penalty:
                best_seq = list(current_seq)
                best_penalty = current_penalty
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Cool down
        T *= cooling
        
        # If stuck for too long, restart with a different random seed
        if no_improve_count > max_no_improve:
            # Try a fresh random start
            target_len = min(best_len + random.randint(0, 3), 256)
            new_seq = _random_subset(target_len)
            nv, nl, np_ = _evaluate(new_seq)
            if nv and nl > best_len:
                best_seq = list(new_seq)
                best_len = nl
            current_seq = new_seq
            current_valid, current_len, current_penalty = nv, nl, np_
            no_improve_count = 0
            T = 50.0  # Reset temperature
    
    return best_seq

def construct():
    start_time = time.time()
    
    # Start with greedy baseline
    best_seq = _greedy_build()
    best_len = len(best_seq)
    
    # Try local search (simulated annealing) to improve
    time_budget = _DEADLINE - (time.time() - start_time)
    if time_budget > 5:
        improved = _local_search(best_seq, time_budget * 0.7)
        if len(improved) > best_len:
            best_seq = improved
            best_len = len(best_seq)
    
    # If we still have time, try multiple random restarts
    remaining = _DEADLINE - (time.time() - start_time)
    if remaining > 10:
        for restart in range(20):
            if time.time() > _DEADLINE - 5:
                break
            # Try different random subsets
            target_len = best_len + random.randint(1, 5)
            if target_len > 256:
                target_len = 256
            rand_seq = _random_subset(target_len)
            rv, rl, rp = _evaluate(rand_seq)
            
            # If valid, try to improve with local search
            if rv:
                # Short local search
                time_left = _DEADLINE - (time.time() - start_time)
                if time_left > 2:
                    improved = _local_search(rand_seq, min(time_left * 0.3, 5))
                    if len(improved) > best_len:
                        best_seq = improved
                        best_len = len(best_seq)
            else:
                # Even if not valid, try to repair with local search
                time_left = _DEADLINE - (time.time() - start_time)
                if time_left > 2:
                    improved = _local_search(rand_seq, min(time_left * 0.2, 3))
                    if len(improved) > best_len:
                        best_seq = improved
                        best_len = len(best_seq)
    
    # Final safety check
    final_valid, final_len, final_penalty = _evaluate(best_seq)
    if not final_valid:
        # Fallback to greedy
        best_seq = _greedy_build()
    
    return best_seq
# EVOLVE-BLOCK-END
