import time
import random
import itertools

def _is_sidon_bitmask(subset):
    """
    Check if a list of integers forms a Sidon set (including diagonals).
    Uses a bitmask for sums. Max sum is 256 (128+128).
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

def _max_sidon_size_fast(ambient):
    """
    Compute the maximum size of a Sidon subset in the given ambient set.
    Check from high to low to find the exact maximum.
    """
    n = len(ambient)
    # Try sizes from high to low
    for size in range(min(n, 8), 3, -1):
        for combo in itertools.combinations(ambient, size):
            if _is_sidon_bitmask(combo):
                return size
    # Check size 4
    for combo in itertools.combinations(ambient, 4):
        if _is_sidon_bitmask(combo):
            return 4
    return 3

def _count_k_sidon(ambient, k=5):
    """
    Count the number of k-element subsets that are Sidon.
    """
    count = 0
    for combo in itertools.combinations(ambient, k):
        if _is_sidon_bitmask(combo):
            count += 1
    return count

def _find_all_5_sidon(ambient):
    """
    Find all 5-element subsets that are Sidon.
    Returns list of tuples (each tuple is a 5-tuple of values).
    """
    result = []
    for combo in itertools.combinations(ambient, 5):
        if _is_sidon_bitmask(combo):
            result.append(combo)
    return result

def _random_ambient(rng):
    return sorted(rng.sample(range(129), 12))

def _structured_candidates():
    """
    Generate structured candidates that might have low Sidon numbers.
    """
    candidates = []
    
    # Small APs - these have many sum collisions
    for step in range(1, 11):
        cand = [i * step for i in range(12)]
        if max(cand) <= 128:
            candidates.append(cand)
    
    # Dense clusters - group numbers close together
    for base in range(0, 100, 5):
        for spread in range(1, 8):
            cand = [base + (i % 6) * spread for i in range(12)]
            # Make them distinct
            seen = set()
            distinct = []
            for x in cand:
                if x not in seen and 0 <= x <= 128:
                    seen.add(x)
                    distinct.append(x)
            if len(distinct) == 12:
                candidates.append(sorted(distinct))
    
    # Two-cluster structure
    for gap in range(20, 80, 10):
        for cluster_size in range(3, 7):
            cand1 = list(range(0, cluster_size * 3, 2))
            cand2 = [gap + x for x in range(gap, gap + cluster_size * 3, 2)]
            cand = cand1[:6] + cand2[:6]
            if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
                candidates.append(sorted(cand))
    
    # Fibonacci-like sequences
    fibs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    candidates.append(sorted(set(fibs[:12])))
    
    # Powers of 2 scaled
    for scale in [1, 2, 3]:
        cand = [min(128, i * scale) for i in [1, 2, 4, 8, 16, 32, 64, 128, 1, 3, 6, 12]]
        cand = list(dict.fromkeys(cand))  # preserve order, remove dupes
        if len(cand) >= 12:
            candidates.append(sorted(cand[:12]))
    
    return candidates

def _local_search(ambient, rng, deadline):
    """
    Local search to minimize the number of 5-Sidon subsets.
    Uses targeted moves: pick a 5-Sidon subset and try to break it.
    """
    current = list(ambient)
    current_count = _count_k_sidon(current, 5)
    
    if current_count == 0:
        return current, 4
        
    best = current[:]
    best_count = current_count
    best_sidon_subsets = _find_all_5_sidon(current)
    
    no_improve = 0
    max_iters = 2000
    
    for iter_num in range(max_iters):
        if time.time() > deadline:
            break
        if best_count == 0:
            break
            
        # Strategy 1: Pick a random 5-Sidon subset and try to break it
        if best_sidon_subsets and rng.random() < 0.6:
            sidon_subset = rng.choice(best_sidon_subsets)
            # Pick one element from this subset to change
            idx_in_subset = rng.randint(0, 4)
            old_val = sidon_subset[idx_in_subset]
            idx = current.index(old_val)
            
            # Try several new values
            found_improvement = False
            for _ in range(20):
                new_val = rng.randint(0, 128)
                if new_val == old_val or new_val in current:
                    continue
                    
                trial = current[:]
                trial[idx] = new_val
                trial.sort()
                
                trial_count = _count_k_sidon(trial, 5)
                
                if trial_count < best_count:
                    best = trial
                    best_count = trial_count
                    best_sidon_subsets = _find_all_5_sidon(trial)
                    no_improve = 0
                    found_improvement = True
                    break
                elif trial_count == best_count and rng.random() < 0.1:
                    # Accept with some probability to escape local minima
                    best = trial
                    best_sidon_subsets = _find_all_5_sidon(trial)
                    no_improve = 0
                    found_improvement = True
                    break
            
            if found_improvement:
                continue
                
        # Strategy 2: Random perturbation
        idx = rng.randint(0, 11)
        old_val = current[idx]
        
        for _ in range(10):
            new_val = rng.randint(0, 128)
            if new_val == old_val or new_val in current:
                continue
                
            trial = current[:]
            trial[idx] = new_val
            trial.sort()
            
            trial_count = _count_k_sidon(trial, 5)
            
            if trial_count < best_count:
                best = trial
                best_count = trial_count
                best_sidon_subsets = _find_all_5_sidon(trial)
                no_improve = 0
                break
            elif trial_count == best_count and rng.random() < 0.05:
                best = trial
                best_sidon_subsets = _find_all_5_sidon(trial)
                no_improve = 0
                break
        
        else:
            no_improve += 1
        
        if no_improve > 300:
            # Restart with a random perturbation
            for _ in range(3):
                idx = rng.randint(0, 11)
                new_val = rng.randint(0, 128)
                if new_val not in best:
                    best[idx] = new_val
                    best.sort()
                    break
            no_improve = 0
            best_count = _count_k_sidon(best, 5)
            best_sidon_subsets = _find_all_5_sidon(best)
            
    # Final check
    if best_count == 0:
        return best, 4
    else:
        # Return the max size (which is 5 if count > 0)
        return best, 5

def _select_best_ambient():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    best_set = None
    best_score = 12  # Max possible
    
    rng = random.Random(42)
    
    # Generate candidates
    candidates = _structured_candidates()
    
    # Add random candidates
    for _ in range(200):
        candidates.append(_random_ambient(rng))
    
    # Quick filter: check if any candidate has no 5-Sidon subset
    for cand in candidates:
        if time.time() > deadline - 15:
            break
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
            
        # Quick check: Is there a 5-Sidon subset?
        has_5 = False
        for combo in itertools.combinations(cand, 5):
            if _is_sidon_bitmask(combo):
                has_5 = True
                break
        
        if not has_5:
            # Verify max Sidon size is 4
            max_size = _max_sidon_size_fast(cand)
            if max_size <= 4:
                return cand
    
    # If no 4 found, find the one with fewest 5-Sidons
    best_count = float('inf')
    best_cand = None
    
    for cand in candidates:
        if time.time() > deadline - 20:
            break
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
            
        count = _count_k_sidon(cand, 5)
        if count < best_count:
            best_count = count
            best_cand = cand
    
    if best_cand is None:
        best_cand = candidates[0]
        best_count = _count_k_sidon(best_cand, 5)
    
    # Local search on the best candidate
    if best_count > 0:
        improved, score = _local_search(best_cand, rng, deadline - 2)
        if score < best_score:
            best_score = score
            best_set = improved
        elif score == 4:
            return improved
    
    # Try local search on a few other candidates
    for cand in candidates[:20]:
        if time.time() > deadline - 5:
            break
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
            
        improved, score = _local_search(cand, rng, deadline - 2)
        if score < best_score:
            best_score = score
            best_set = improved
        elif score == 4:
            return improved
    
    if best_set is None:
        # Fallback: return the one with min 5-count
        return best_cand
        
    return best_set

def construct():
    return _select_best_ambient()
