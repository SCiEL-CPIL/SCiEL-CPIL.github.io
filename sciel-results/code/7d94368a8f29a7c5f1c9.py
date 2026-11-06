"""Program for reciprocal weight of three-progression-free sets on [1,256]."""
import time
import random
import math

# EVOLVE-BLOCK-START
N = 256
RECIPS = [0.0] + [1.0 / i for i in range(1, N + 1)]

def _is_3ap_free_add(mask, x):
    """Check if adding x to set represented by mask keeps it 3-AP free.
    x is 1-indexed. mask is int where bit i-1 is set if i is in set.
    """
    if (mask >> (x - 1)) & 1:
        return True
    
    m = mask
    while m:
        lsb = m & -m
        a = lsb.bit_length() # 1-indexed
        m ^= lsb
        
        # x is middle: c = 2x - a
        c = 2 * x - a
        if c > 0 and c <= N and c != a and c != x:
            if (mask >> (c - 1)) & 1:
                return False
        
        # x is end: b = (x + a) / 2
        if (x + a) % 2 == 0:
            b = (x + a) // 2
            if b > 0 and b <= N and b != a and b != x:
                if (mask >> (b - 1)) & 1:
                    return False
    return True

def _get_sum(mask):
    s = 0.0
    m = mask
    while m:
        lsb = m & -m
        i = lsb.bit_length()
        s += RECIPS[i]
        m ^= lsb
    return s

def _greedy_ascending():
    mask = 0
    for x in range(1, N + 1):
        if _is_3ap_free_add(mask, x):
            mask |= (1 << (x - 1))
    return mask

def _greedy_descending():
    """Greedy adding from largest to smallest, but we want small numbers (high reciprocals).
    Actually, let's try greedy by reciprocal value: add smallest first (highest 1/x)."""
    mask = 0
    for x in range(1, N + 1):
        if _is_3ap_free_add(mask, x):
            mask |= (1 << (x - 1))
    return mask

def _greedy_by_reciprocal():
    """Greedy: at each step, add the element with highest reciprocal that doesn't create a 3-AP."""
    mask = 0
    candidates = list(range(1, N + 1))
    while candidates:
        # Sort by reciprocal (descending) = by value (ascending)
        candidates.sort(key=lambda x: x)
        added = False
        for x in candidates:
            if _is_3ap_free_add(mask, x):
                mask |= (1 << (x - 1))
                candidates.remove(x)
                added = True
                break
        if not added:
            break
    return mask

def _local_search(mask, deadline, max_iters=5000):
    """Hill climbing: try single additions and removals."""
    current_mask = mask
    current_score = _get_sum(current_mask)
    improved = True
    iters = 0
    
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        
        # Try adding elements
        for x in range(1, N + 1):
            if not (current_mask >> (x - 1)) & 1:
                if _is_3ap_free_add(current_mask, x):
                    new_mask = current_mask | (1 << (x - 1))
                    new_score = current_score + RECIPS[x]
                    if new_score > current_score:
                        current_mask = new_mask
                        current_score = new_score
                        improved = True
        
        # Try removing elements (only if it allows adding something better)
        # For now, just try removals that lead to net gain via re-addition
        for x in range(1, N + 1):
            if (current_mask >> (x - 1)) & 1:
                # Try removing x and see if we can add something
                trial_mask = current_mask ^ (1 << (x - 1))
                best_gain = 0
                best_add = 0
                for y in range(1, N + 1):
                    if not (trial_mask >> (y - 1)) & 1:
                        if _is_3ap_free_add(trial_mask, y):
                            gain = RECIPS[y] - RECIPS[x]
                            if gain > best_gain:
                                best_gain = gain
                                best_add = y
                if best_gain > 0:
                    current_mask = trial_mask | (1 << (best_add - 1))
                    current_score += best_gain
                    improved = True
    
    return current_mask, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Try multiple initializations
    init_ascending = _greedy_ascending()
    init_descending = _greedy_descending()
    init_reciprocal = _greedy_by_reciprocal()
    
    # Pick the best initialization
    candidates = [init_ascending, init_descending, init_reciprocal]
    best_mask = max(candidates, key=_get_sum)
    best_score = _get_sum(best_mask)
    
    # Simulated Annealing with improved moves
    current_mask = best_mask
    current_score = best_score
    
    # Initial temperature
    T = 0.02
    cooling_rate = 0.995
    
    # Precompute list of set elements for faster access
    while time.time() < deadline:
        if T < 1e-6:
            break
        
        # Choose move type: 70% add/remove, 30% swap
        move_type = random.random()
        
        if move_type < 0.7:
            # Single add or remove
            x = random.randint(1, N)
            is_in_set = (current_mask >> (x - 1)) & 1
            
            if is_in_set:
                # Remove x
                new_mask = current_mask ^ (1 << (x - 1))
                new_score = current_score - RECIPS[x]
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            else:
                # Add x
                if _is_3ap_free_add(current_mask, x):
                    new_mask = current_mask | (1 << (x - 1))
                    new_score = current_score + RECIPS[x]
                    delta = new_score - current_score
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
        else:
            # Swap: remove one, add another
            # Pick element to remove
            m = current_mask
            if m == 0:
                continue
            # Pick random element from set
            # Get all set elements
            elements = []
            mm = m
            while mm:
                lsb = mm & -mm
                i = lsb.bit_length()
                elements.append(i)
                mm ^= lsb
            
            if elements:
                x = random.choice(elements)
                # Try adding a random element not in set
                y = random.randint(1, N)
                if y != x and not (current_mask >> (y - 1)) & 1:
                    # Check if we can remove x and add y
                    trial_mask = current_mask ^ (1 << (x - 1))
                    if _is_3ap_free_add(trial_mask, y):
                        new_mask = trial_mask | (1 << (y - 1))
                        new_score = current_score - RECIPS[x] + RECIPS[y]
                        delta = new_score - current_score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            current_mask = new_mask
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_mask = current_mask
        
        T *= cooling_rate
    
    # Final polish: local search on best solution
    if time.time() < deadline - 5.0:
        polished_mask, polished_score = _local_search(best_mask, deadline, max_iters=2000)
        if polished_score > best_score:
            best_score = polished_score
            best_mask = polished_mask
    
    # Convert best_mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length()
        result.append(i)
        m ^= lsb
    
    return sorted(result)

# EVOLVE-BLOCK-END
