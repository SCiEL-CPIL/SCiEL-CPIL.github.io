import time
import random

def _score_bitset(mask):
    """Count distinct d>0 such that a, a+d, a+2d all in set."""
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _arr_to_mask(arr):
    mask = 0
    for x in arr:
        mask |= (1 << (x - 1))
    return mask

def _mask_to_arr(mask):
    arr = []
    for x in range(1, 513):
        if mask & (1 << (x - 1)):
            arr.append(x)
    return arr

def _covered_diffs(mask):
    """Return set of d values that are covered."""
    covered = set()
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            covered.add(d)
    return covered

def _repair(arr, rng):
    """Ensure arr is a list of 32 distinct integers in 1..512."""
    seen = set()
    cleaned = []
    for x in arr:
        if 1 <= x <= 512 and x not in seen:
            cleaned.append(x)
            seen.add(x)
    
    if len(cleaned) > 32:
        rng.shuffle(cleaned)
        cleaned = cleaned[:32]
    elif len(cleaned) < 32:
        all_vals = set(range(1, 513))
        available = all_vals - seen
        if len(available) < 32 - len(cleaned):
            return list(range(1, 33))
        to_add = rng.sample(list(available), 32 - len(cleaned))
        cleaned.extend(to_add)
        
    return sorted(cleaned)

def _crossover(p1, p2, rng):
    k = rng.randint(1, 31)
    child = p1[:k] + p2[k:]
    return _repair(child, rng)

def _mutate(arr, rng):
    """Point mutation: change one element."""
    arr = arr[:]  # Copy to avoid modifying in place
    idx = rng.randint(0, 31)
    old = arr[idx]
    new = rng.randint(1, 512)
    if new != old:
        arr[idx] = new
    return _repair(arr, rng)

def _mutate_multi(arr, rng, n_mut=1):
    """Mutate multiple elements."""
    arr = arr[:]
    for _ in range(n_mut):
        idx = rng.randint(0, 31)
        old = arr[idx]
        new = rng.randint(1, 512)
        if new != old:
            arr[idx] = new
    return _repair(arr, rng)

def _greedy_construct(deadline, rng, start_mask=0):
    """
    Greedy construction: start with some elements, then add/swap to maximize covered diffs.
    This is a heuristic to get a good starting point.
    """
    mask = start_mask
    arr = _mask_to_arr(mask)
    covered = _covered_diffs(mask)
    
    # If we start empty, pick a random seed or dense start
    if not arr:
        # Try a dense block
        start = rng.randint(1, 512 - 32)
        arr = list(range(start, start + 32))
        mask = _arr_to_mask(arr)
        covered = _covered_diffs(mask)
    
    # If we have fewer than 32, fill up randomly first
    while len(arr) < 32:
        x = rng.randint(1, 512)
        if not (mask & (1 << (x - 1))):
            arr.append(x)
            arr.sort()
            mask |= (1 << (x - 1))
            covered = _covered_diffs(mask)
    
    # Greedy improvement: try swapping each element with a random new one
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try each index
        for idx in range(32):
            old = arr[idx]
            # Try a few candidates
            for _ in range(50):
                new = rng.randint(1, 512)
                if new == old or (mask & (1 << (new - 1))):
                    continue
                
                trial_mask = mask & ~(1 << (old - 1)) | (1 << (new - 1))
                trial_score = _score_bitset(trial_mask)
                cur_score = _score_bitset(mask)
                
                if trial_score > cur_score:
                    # Accept
                    mask = trial_mask
                    arr[idx] = new
                    arr.sort()
                    improved = True
                    break
            if time.time() > deadline - 2:
                break
    return arr, mask, _score_bitset(mask)

def _hill_climb(arr, mask, score, deadline, rng, max_steps=5000):
    """
    Local search: swap elements to improve score.
    """
    cur_arr = arr[:]
    cur_mask = mask
    cur_score = score
    steps = 0
    
    improved = True
    while improved and steps < max_steps and time.time() < deadline - 1:
        improved = False
        steps += 1
        
        # Try random swaps
        # Pick two distinct indices
        i = rng.randint(0, 31)
        j = rng.randint(0, 31)
        if i == j:
            continue
            
        old_i = cur_arr[i]
        old_j = cur_arr[j]
        
        # Option 1: Replace i with random new value
        new_val = rng.randint(1, 512)
        if new_val == old_i or new_val == old_j or (cur_mask & (1 << (new_val - 1))):
            # Option 2: Replace j with random new value
            new_val2 = rng.randint(1, 512)
            if new_val2 != old_i and new_val2 != old_j and not (cur_mask & (1 << (new_val2 - 1))):
                trial_mask = cur_mask & ~(1 << (old_j - 1)) | (1 << (new_val2 - 1))
                trial_score = _score_bitset(trial_mask)
                if trial_score > cur_score:
                    cur_mask = trial_mask
                    cur_arr[j] = new_val2
                    cur_arr.sort()
                    cur_score = trial_score
                    improved = True
        else:
            trial_mask = cur_mask & ~(1 << (old_i - 1)) | (1 << (new_val - 1))
            trial_score = _score_bitset(trial_mask)
            if trial_score > cur_score:
                cur_mask = trial_mask
                cur_arr[i] = new_val
                cur_arr.sort()
                cur_score = trial_score
                improved = True
        
        # Option 3: Swap i and j with random values
        if not improved:
            new_i = rng.randint(1, 512)
            new_j = rng.randint(1, 512)
            if (new_i != old_i and new_j != old_j and new_i != old_j and new_j != old_i and
                not (cur_mask & (1 << (new_i - 1))) and not (cur_mask & (1 << (new_j - 1)))):
                trial_mask = cur_mask & ~(1 << (old_i - 1)) & ~(1 << (old_j - 1)) | (1 << (new_i - 1)) | (1 << (new_j - 1))
                trial_score = _score_bitset(trial_mask)
                if trial_score > cur_score:
                    cur_mask = trial_mask
                    cur_arr[i] = new_i
                    cur_arr[j] = new_j
                    cur_arr.sort()
                    cur_score = trial_score
                    improved = True

    return cur_arr, cur_mask, cur_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    rng = random.Random(42)
    
    best_arr = None
    best_mask = 0
    best_score = -1
    
    # Phase 1: Multiple Greedy Starts
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if i < 5:
            # Dense starts
            start = rng.randint(1, 512 - 32)
            arr = list(range(start, start + 32))
            mask = _arr_to_mask(arr)
            arr, mask, score = _greedy_construct(deadline, rng, mask)
        elif i < 10:
            # Structured starts (APs)
            d = rng.randint(1, 8)
            start = rng.randint(1, 512 - 31 * d)
            arr = [start + j * d for j in range(32)]
            mask = _arr_to_mask(arr)
            arr, mask, score = _greedy_construct(deadline, rng, mask)
        else:
            # Random starts
            arr = rng.sample(range(1, 513), 32)
            mask = _arr_to_mask(arr)
            arr, mask, score = _greedy_construct(deadline, rng, mask)
        
        if score > best_score:
            best_score = score
            best_arr = arr
            best_mask = mask
            
    # Phase 2: Hill Climbing from Best
    if best_arr is not None:
        best_arr, best_mask, best_score = _hill_climb(best_arr, best_mask, best_score, deadline, rng, max_steps=10000)
        
    # Phase 3: Simple GA/SA hybrid if time remains
    if time.time() < deadline - 20 and best_arr is not None:
        # Initialize population with clones of best + random perturbations
        pop = [(best_arr, best_score)]
        for _ in range(19):
            mutant = _mutate_multi(best_arr, rng, n_mut=rng.randint(1, 5))
            s = _score_bitset(_arr_to_mask(mutant))
            pop.append((mutant, s))
        
        pop.sort(key=lambda x: x[1], reverse=True)
        
        generations = 0
        while time.time() < deadline - 5:
            generations += 1
            new_pop = [pop[0]]  # Elitism
            
            while len(new_pop) < 20:
                # Tournament selection
                t1 = max(random.sample(pop, 3), key=lambda x: x[1])
                t2 = max(random.sample(pop, 3), key=lambda x: x[1])
                
                p1 = t1[0]
                p2 = t2[0]
                
                # Crossover
                if rng.random() < 0.3:
                    child = _crossover(p1, p2, rng)
                else:
                    child = p1[:]
                
                # Mutation
                n_mut = rng.randint(1, 3)
                child = _mutate_multi(child, rng, n_mut=n_mut)
                
                s = _score_bitset(_arr_to_mask(child))
                new_pop.append((child, s))
            
            pop = new_pop
            pop.sort(key=lambda x: x[1], reverse=True)
            
            if pop[0][1] > best_score:
                best_score = pop[0][1]
                best_arr = pop[0][0]
                best_mask = _arr_to_mask(best_arr)
                
                # Quick hill climb on new best
                if time.time() < deadline - 5:
                    best_arr, best_mask, best_score = _hill_climb(best_arr, best_mask, best_score, deadline, rng, max_steps=100)

    # Final Validation
    if best_arr is None:
        return list(range(1, 33))
        
    result = sorted(set(best_arr))
    if len(result) != 32:
        # Repair
        result = _repair(result, rng)
    
    # Ensure all in range 1..512
    result = [x for x in result if 1 <= x <= 512]
    if len(result) < 32:
        extra = [x for x in range(1, 513) if x not in result]
        result.extend(extra[:32 - len(result)])
        result = sorted(result[:32])
        
    return result
