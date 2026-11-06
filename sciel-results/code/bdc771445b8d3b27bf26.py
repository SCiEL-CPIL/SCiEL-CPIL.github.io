import time
import random
import itertools

def _is_sidon_bitmask(subset):
    """
    Check if a list of integers forms a Sidon set (including diagonals).
    Max sum is 256.
    """
    mask = 0
    n = len(subset)
    for i in range(n):
        a = subset[i]
        for j in range(i, n):
            s = a + subset[j]
            bit = 1 << s
            if mask & bit:
                return False
            mask |= bit
    return True

def _count_5_sidon(ambient):
    """
    Count the number of 5-element subsets that are Sidon.
    """
    count = 0
    for combo in itertools.combinations(ambient, 5):
        if _is_sidon_bitmask(combo):
            count += 1
    return count

def _has_4_sidon(ambient):
    """
    Check if there is at least one 4-element Sidon subset.
    """
    for combo in itertools.combinations(ambient, 4):
        if _is_sidon_bitmask(combo):
            return True
    return False

def _local_search(ambient, rng, deadline):
    """
    Local search to minimize the number of 5-Sidon subsets.
    """
    current = list(ambient)
    current_count = _count_5_sidon(current)
    
    if current_count == 0:
        return current, 0
        
    best = current
    best_count = current_count
    
    no_improve = 0
    max_iters = 5000
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if best_count == 0:
            break
            
        # Pick a random index to change
        idx = rng.randint(0, 11)
        old_val = current[idx]
        
        # Try a few random new values
        found_improvement = False
        for _ in range(20): # Try 20 random perturbations per step
            new_val = rng.randint(0, 128)
            if new_val == old_val or new_val in current:
                continue
                
            trial = current[:]
            trial[idx] = new_val
            trial.sort()
            
            trial_count = _count_5_sidon(trial)
            
            if trial_count < best_count:
                best = trial
                best_count = trial_count
                current = trial
                no_improve = 0
                found_improvement = True
                break
            elif trial_count == best_count:
                # Accept with small probability to escape local minima
                if rng.random() < 0.05:
                    current = trial
                    no_improve = 0
                    break
        if not found_improvement:
            no_improve += 1
            
        if no_improve > 150:
            break
            
    return best, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 85 # Safety margin for 90s limit
    
    rng = random.Random(12345)
    
    best_set = None
    best_count = float('inf')
    
    # Generate initial candidates
    # 1. Arithmetic Progressions (dense collisions)
    for step in range(1, 11):
        cand = [i * step for i in range(12)]
        if max(cand) <= 128:
            yield_cand = sorted(cand)
            cnt = _count_5_sidon(yield_cand)
            if cnt < best_count:
                best_count = cnt
                best_set = yield_cand
            if best_count == 0:
                # Verify max size is 4 (i.e., 4-Sidon exists)
                # If no 4-Sidon, max size is 3.
                return best_set
                
    # 2. Clustered sets in small ranges
    for max_val in range(15, 50, 1):
        if max_val < 12:
            continue
        for _ in range(50):
            # Sample 12 distinct from [0, max_val]
            cand = sorted(rng.sample(range(max_val + 1), 12))
            cnt = _count_5_sidon(cand)
            if cnt < best_count:
                best_count = cnt
                best_set = cand
            if best_count == 0:
                return best_set

    # 3. Local search on the best candidate found so far
    if best_count > 0:
        improved, improved_count = _local_search(best_set, rng, deadline - 5)
        if improved_count < best_count:
            best_set = improved
            best_count = improved_count
            
    # If we still have 5-Sidons, try more random restarts
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        # Generate a random set
        cand = sorted(rng.sample(range(129), 12))
        cnt = _count_5_sidon(cand)
        if cnt < best_count:
            best_set = cand
            best_count = cnt
            
        # Local search on this candidate
        improved, improved_count = _local_search(cand, rng, deadline - 10)
        if improved_count < best_count:
            best_set = improved
            best_count = improved_count
            
        if best_count == 0:
            return best_set

    # Fallback: return the best set found
    if best_set is None:
        return sorted(rng.sample(range(129), 12))
    return best_set
