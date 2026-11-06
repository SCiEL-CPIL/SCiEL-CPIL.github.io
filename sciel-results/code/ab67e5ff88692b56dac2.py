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
    We assume the answer is at least 4 for N=12 in [0,128].
    We check for 5 first. If found, we might check 6 just in case, but usually 5 is the bottleneck.
    Actually, we should check from high to low to find the max.
    Max possible is 12. But we want to minimize it.
    For the search, we primarily care if it is <= 4.
    So, check if there is a Sidon subset of size 5.
    If yes, return at least 5. (We can return 5 to indicate "bad").
    If no, return 4.
    
    To be precise for the score, we should return the actual max size.
    However, for optimization, distinguishing 4 from 5 is the main goal.
    Let's check 5, then 6, etc., only if necessary.
    Given N=12, max Sidon size is likely 4 or 5.
    """
    n = len(ambient)
    # Check for size 5
    # If we find one, we know max >= 5.
    # We don't need to know if it's 6, 7, etc. for the primary optimization 
    # unless 4 is impossible. But 4 is likely achievable.
    # So, if any 5-subset is Sidon, return 5 (or higher if we check).
    # If no 5-subset is Sidon, return 4.
    
    # Optimization: Pre-check if a specific known structure allows 5.
    for combo in itertools.combinations(ambient, 5):
        if _is_sidon_bitmask(combo):
            # It has at least 5. 
            # Does it have 6?
            # For the purpose of minimization, 5 is "worse" than 4.
            # We can return 5 immediately to save time, as we are looking for 4.
            # If we find a set with max 4, it will return 4.
            # If it returns 5, we know it's not optimal (assuming 4 is possible).
            return 5
            
    # If no 5-subset is Sidon, the max size is <= 4.
    # We know a 4-subset always exists? 
    # For N=12, yes, it's extremely likely.
    # Let's verify 4 exists to be safe.
    for combo in itertools.combinations(ambient, 4):
        if _is_sidon_bitmask(combo):
            return 4
            
    return 3

def _count_5_sidon(ambient):
    """
    Count the number of 5-element subsets that are Sidon.
    This gives a gradient for local search.
    """
    count = 0
    for combo in itertools.combinations(ambient, 5):
        if _is_sidon_bitmask(combo):
            count += 1
    return count

def _random_ambient(rng):
    return sorted(rng.sample(range(129), 12))

def _local_search(ambient, rng, deadline):
    """
    Local search to minimize the number of 5-Sidon subsets.
    If we reach 0, we have a max Sidon size of 4.
    """
    current = list(ambient)
    current_count = _count_5_sidon(current)
    
    if current_count == 0:
        return current, 4
        
    best = current
    best_count = current_count
    
    no_improve = 0
    max_iters = 1000
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if best_count == 0:
            break
            
        # Pick a random index to change
        idx = rng.randint(0, 11)
        old_val = current[idx]
        
        # Try to find a new value that reduces the count
        # We can try a few random values or a specific strategy
        # Strategy: Pick a 5-Sidon subset containing current[idx] and try to break it.
        
        # Simple random perturbation
        new_val = rng.randint(0, 128)
        if new_val == old_val:
            continue
        if new_val in current:
            continue
            
        trial = current[:]
        trial[idx] = new_val
        trial.sort()
        
        trial_count = _count_5_sidon(trial)
        
        if trial_count < best_count:
            best = trial
            best_count = trial_count
            no_improve = 0
        else:
            no_improve += 1
            # Simulated annealing style acceptance?
            if trial_count == best_count and rng.random() < 0.05:
                best = trial
                no_improve = 0
                
        if no_improve > 200:
            break
            
    # Final check
    if best_count == 0:
        return best, 4
    else:
        # Return the max size (which is 5 if count > 0)
        return best, 5

def _select_best_ambient():
    start_time = time.time()
    deadline = start_time + 85 # Safety margin
    
    best_set = None
    best_score = 12 # Max possible
    
    # 1. Try some structured candidates that might have low Sidon numbers
    # Arithmetic progressions tend to have high Sidon numbers? 
    # Actually, APs have many collisions, so they are good for MINIMIZING Sidon size.
    # Example: 0, 1, 2, 3, 4, 5... 
    # Sums: 0+0=0, 0+1=1, 1+1=2, 0+2=2 (collision 1+1=2). 
    # So APs are very dense in collisions.
    
    candidates = []
    # Small APs
    for step in range(1, 11):
        cand = [i * step for i in range(12)]
        if max(cand) <= 128:
            candidates.append(cand)
            
    # Random candidates
    rng = random.Random(12345)
    for _ in range(100):
        candidates.append(_random_ambient(rng))
        
    # Evaluate candidates
    for cand in candidates:
        if time.time() > deadline - 5:
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
            # Max Sidon size is 4 (or less, but 4 is min realistic)
            # We found a solution with score 4.
            # We can stop if we assume 4 is the minimum.
            # The problem asks to MINIMIZE. 
            # Is 3 possible?
            # N=12. To have max Sidon size 3, every 4-subset must NOT be Sidon.
            # Number of 4-subsets is C(12,4) = 495.
            # Each 4-subset must have a collision.
            # This is a very strong constraint. 
            # Given the density of [0,128], it's highly probable that 4 is the minimum.
            # Let's assume 4 is the target.
            return cand
            
    # If no 4 found, we need to search for the one with the fewest 5-Sidons
    # or just return the best found so far.
    
    best_count = float('inf')
    best_cand = candidates[0]
    
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        count = _count_5_sidon(cand)
        if count < best_count:
            best_count = count
            best_cand = cand
            
    # Local search on the best candidate
    if best_count > 0:
        improved, score = _local_search(best_cand, rng, deadline - 2)
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
