import time
import random
import math
import bisect

N = 32
MAX_VAL = 512
MAX_D = 255 # (512-1)//2

def _score_mask(mask):
    """
    Count distinct d>0 such that exists a with a, a+d, a+2d in the set.
    mask: integer where bit i (0-indexed) is 1 if value (i+1) is in the set.
    """
    count = 0
    for d in range(1, MAX_D + 1):
        # Check if there exists a such that a, a+d, a+2d are all in the set
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _list_to_mask(lst):
    """Convert a list of values (1..512) to a bitmask."""
    mask = 0
    for v in lst:
        mask |= (1 << (v - 1))
    return mask

def _generate_candidate(current_list, current_mask, x_to_remove, rng):
    """
    Generate a candidate y to replace x_to_remove.
    Uses heuristic based on pairs in current_list.
    """
    # 10% chance of pure random
    if rng.random() < 0.1:
        y = rng.randint(1, MAX_VAL)
        if y == x_to_remove or ((current_mask >> (y - 1)) & 1):
            return None
        return y
    
    # Heuristic: Pick a random pair from the current list
    n = len(current_list)
    if n < 2:
        y = rng.randint(1, MAX_VAL)
        if y == x_to_remove or ((current_mask >> (y - 1)) & 1):
            return None
        return y

    # Pick two distinct indices
    i = rng.randint(0, n - 1)
    j = rng.randint(0, n - 2)
    if j >= i:
        j += 1
        
    u = current_list[i]
    v = current_list[j]
    
    candidates = []
    
    # y = 2v - u (u, v, y is AP)
    c1 = 2 * v - u
    if 1 <= c1 <= MAX_VAL:
        candidates.append(c1)
        
    # y = 2u - v (v, u, y is AP)
    c2 = 2 * u - v
    if 1 <= c2 <= MAX_VAL:
        candidates.append(c2)
        
    # y = (u + v) / 2 (u, y, v is AP)
    if (u + v) % 2 == 0:
        c3 = (u + v) // 2
        if 1 <= c3 <= MAX_VAL:
            candidates.append(c3)
            
    # Filter valid candidates
    valid_candidates = []
    for c in candidates:
        if c == x_to_remove:
            continue
        if (current_mask >> (c - 1)) & 1:
            continue
        valid_candidates.append(c)
        
    if valid_candidates:
        return rng.choice(valid_candidates)
    else:
        # Fallback to random if heuristic fails
        y = rng.randint(1, MAX_VAL)
        if y == x_to_remove or ((current_mask >> (y - 1)) & 1):
            return None
        return y

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = -1
    best_list = []
    
    # Define seeds
    seeds = [
        list(range(1, 33)), # Interval
        list(range(1, 32)) + [33], # Interval + 1
        [i * 16 + 1 for i in range(32)], # Spread
        [1, 2, 4, 8, 16, 32, 64, 128, 256] + list(range(257, 289)), # Mixed
        list(range(1, 17)) + list(range(256, 272)), # Two intervals
    ]
    
    # Add a few random seeds
    for _ in range(5):
        seeds.append(random.sample(range(1, MAX_VAL + 1), N))
        
    rng = random.Random(42)
    
    for trial_idx, seed in enumerate(seeds):
        if time.time() > deadline:
            break
            
        current_list = sorted(seed[:])
        # Ensure length is N and valid
        if len(current_list) != N:
            current_list = random.sample(range(1, MAX_VAL + 1), N)
            current_list.sort()
            
        current_mask = _list_to_mask(current_list)
        current_score = _score_mask(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
            
        # Simulated Annealing Parameters
        time_per_trial = 10.0
        local_deadline = start_time + time_per_trial
        
        T = 15.0
        T_min = 0.01
        target_steps = 1500
        cooling = (T_min / T) ** (1.0 / target_steps)
        
        while time.time() < local_deadline:
            # Pick x to remove
            x_idx = rng.randint(0, N - 1)
            x = current_list[x_idx]
            
            # Generate candidate y
            y = _generate_candidate(current_list, current_mask, x, rng)
            
            if y is None:
                # Retry with random
                y = rng.randint(1, MAX_VAL)
                if y == x or ((current_mask >> (y - 1)) & 1):
                    continue
            
            # Create proposed state
            # Remove x
            proposed_mask = current_mask ^ (1 << (x - 1))
            # Add y
            proposed_mask ^= (1 << (y - 1))
            
            # Update list
            # Remove x and insert y, keeping sorted
            proposed_list = current_list[:x_idx] + current_list[x_idx+1:]
            ins_idx = bisect.bisect_left(proposed_list, y)
            proposed_list.insert(ins_idx, y)
            
            new_score = _score_mask(proposed_mask)
            
            delta = new_score - current_score
            
            if delta > 0 or (T > 0 and rng.random() < math.exp(delta / T)):
                # Accept
                current_list = proposed_list
                current_mask = proposed_mask
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min

    return sorted(best_list)
