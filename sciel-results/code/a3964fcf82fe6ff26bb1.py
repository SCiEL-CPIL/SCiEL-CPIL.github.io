"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _fast_is_sidon(lst):
    """Fast Sidon check for sorted list (all unordered pair sums unique including diagonals)."""
    n = len(lst)
    sums = set()
    for i in range(n):
        a = lst[i]
        for j in range(i, n):
            s = a + lst[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(ambient):
    """Compute max Sidon subset size for sorted 12-element list."""
    n = len(ambient)
    # We know a 2-element set is always Sidon.
    # Check from largest possible k down to 3. If no 3-element Sidon found, return 2.
    # For 12 elements, checking k=6..3 is the main cost.
    for k in range(min(n, 6), 2, -1):
        for combo in itertools.combinations(ambient, k):
            if _fast_is_sidon(list(combo)):
                return k
    # Check for k=3
    for combo in itertools.combinations(ambient, 3):
        if _fast_is_sidon(list(combo)):
            return 3
    return 2

def _no_5_sidon_check(ambient):
    """
    Returns True if no 5-element subset of ambient is Sidon.
    This is a necessary condition for max_sidon_size <= 4.
    It's a cheap filter to avoid expensive _max_sidon_size calls.
    """
    n = len(ambient)
    for combo in itertools.combinations(ambient, 5):
        if _fast_is_sidon(list(combo)):
            return False
    return True

def _random_ambient(rng):
    return sorted(rng.sample(range(129), 12))

def _local_search(ambient, rng, deadline, current_best_k):
    """
    Local search: swap one element at a time, with occasional double swaps.
    Uses _no_5_sidon_check as a fast filter before calling _max_sidon_size.
    """
    best = list(ambient)
    best_k = current_best_k
    no_improve = 0
    max_iters = 2000
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if best_k <= 3:
            break
            
        # Decide whether to do single or double swap
        if rng.random() < 0.85:
            idx = rng.randint(0, 11)
            new_val = rng.randint(0, 128)
            if new_val in best:
                no_improve += 1
                continue
            trial = best[:]
            trial[idx] = new_val
            trial.sort()
        else:
            idx1, idx2 = rng.sample(range(12), 2)
            new_val1 = rng.randint(0, 128)
            new_val2 = rng.randint(0, 128)
            if new_val1 in best or new_val2 in best or new_val1 == new_val2:
                no_improve += 1
                continue
            trial = best[:]
            trial[idx1] = new_val1
            trial[idx2] = new_val2
            trial.sort()
            
        # Fast filter: if it has a 5-Sidon, it can't have max_sidon_size <= 4
        # If we're currently at k=5 or 6, we only care if it drops to 4 or below.
        # If we're at k=4, we care if it drops to 3.
        # Actually, if best_k <= 4, we want to find k=3.
        # If best_k >= 5, we want to find k <= 4.
        
        # Optimization: If best_k > 4, use the 5-Sidon check as a filter.
        # If _no_5_sidon_check(trial) is True, then max_sidon_size <= 4.
        # We still need to know if it's 3 or 4.
        # If _no_5_sidon_check(trial) is False, max_sidon_size >= 5, so it's not an improvement if best_k <= 4?
        # Wait, if best_k is 5, and trial has 5-Sidon, trial's max is >=5, so not an improvement.
        # If best_k is 6, and trial has 5-Sidon, trial's max is >=5, so not an improvement (unless we want to drop to 5? No, we minimize).
        
        # So:
        # If best_k > 4:
        #   If _no_5_sidon_check(trial) is False: k >= 5, not better than best_k if best_k <= 5? 
        #   Actually if best_k is 6, k>=5 is better than 6? No, 5 < 6. 
        #   So if best_k is 6, we want k=5 or 4 or 3.
        #   If best_k is 5, we want k=4 or 3.
        
        # Let's just compute k if it passes a heuristic or if best_k is low.
        
        if best_k > 4:
            # If it contains a 5-Sidon, k >= 5.
            # If we are at 6, k=5 is an improvement.
            # If we are at 5, k=5 is not an improvement.
            has_5 = not _no_5_sidon_check(trial)
            if has_5:
                if best_k == 6:
                    # It's at least 5. Could be 5 or 6.
                    # We need to check if it's exactly 5? Or just accept it as 5?
                    # If it's 6, it's not an improvement.
                    # If it's 5, it is an improvement.
                    # _max_sidon_size will tell us.
                    # But maybe we can guess?
                    # Let's just run _max_sidon_size to be safe, but maybe skip if we have many?
                    # Actually, _no_5_sidon_check is C(12,5)=792 checks. _max_sidon_size is C(12,6)+C(12,5)...
                    # C(12,6)=924. So _no_5_sidon_check is about as expensive as the k=6 loop.
                    # So it's not a huge saving.
                    pass
                else: # best_k == 5
                    # k >= 5, so not better than 5.
                    no_improve += 1
                    continue
            else:
                # k <= 4. This is an improvement over 5 or 6.
                k = _max_sidon_size(trial)
                if k < best_k:
                    best = trial
                    best_k = k
                    no_improve = 0
                else:
                    no_improve += 1
                    if k == best_k and rng.random() < 0.1:
                        best = trial
                        no_improve = 0
        else:
            # best_k <= 4
            # We want k < best_k, so k=3 or k=2.
            # If best_k is 4, we want 3.
            # If best_k is 3, we want 2.
            
            # If best_k == 4:
            # Check if trial has a 4-Sidon?
            # If not, k <= 3.
            # C(12,4) = 495 checks.
            has_4 = False
            if best_k == 4:
                for combo in itertools.combinations(trial, 4):
                    if _fast_is_sidon(list(combo)):
                        has_4 = True
                        break
            
            if has_4:
                # k >= 4. Not better than 4.
                no_improve += 1
                if k == best_k and rng.random() < 0.1: # Wait, k is not computed here
                    # We don't know k exactly, but we know k>=4.
                    # If best_k is 4, k could be 4, 5, 6.
                    # Sideways move?
                    best = trial
                    no_improve = 0
            else:
                # k <= 3.
                k = _max_sidon_size(trial)
                if k < best_k:
                    best = trial
                    best_k = k
                    no_improve = 0
                else:
                    no_improve += 1
                    if k == best_k and rng.random() < 0.1:
                        best = trial
                        no_improve = 0
                        
        if no_improve >= 300:
            break
    
    return best, best_k

def _candidates():
    cands = []
    # Dense sets
    cands.append(list(range(12)))
    cands.append(list(range(1, 13)))
    cands.append([0,1,2,3,4,5,6,7,8,9,10,12])
    cands.append([0,1,2,3,4,5,6,7,8,9,11,13])
    
    # Arithmetic Progressions
    for step in [2, 3, 4, 5, 6, 7, 8, 10, 11]:
        cands.append(list(range(0, step*12, step)))
        
    # Modular constructions
    cands.append(sorted([(i * 17) % 128 for i in range(12)]))
    cands.append(sorted([(i * 19) % 128 for i in range(12)]))
    cands.append(sorted([(i * 23) % 128 for i in range(12)]))
    cands.append(sorted([(i * 29) % 128 for i in range(12)]))
    
    # Randomized structured sets
    rng_init = random.Random(123)
    for _ in range(5):
        base = rng_init.sample(range(129), 12)
        cands.append(sorted(base))
        
    return cands

def _select_best_ambient():
    deadline = time.time() + 85  # 5s safety margin
    best = None
    best_k = None
    
    # Test hand-picked candidates first
    for cand in _candidates():
        if len(cand) != 12:
            continue
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        k = _max_sidon_size(cand)
        if best_k is None or k < best_k:
            best = cand
            best_k = k
            if best_k <= 3:
                break
        if time.time() > deadline - 10:
            break
    
    # If we haven't found a very good one, do random search + local search
    rng = random.Random(42)
    max_trials = 3000
    
    for trial in range(max_trials):
        if time.time() > deadline - 5:
            break
        if best_k is not None and best_k <= 3:
            break
            
        # Generate a random candidate
        # Mix of pure random and slightly structured
        if rng.random() < 0.5:
            cand = _random_ambient(rng)
        else:
            # Try to cluster elements to force sum collisions
            cluster_center = rng.randint(10, 118)
            cand = []
            while len(cand) < 12:
                val = cluster_center + rng.randint(-10, 10)
                if 0 <= val <= 128 and val not in cand:
                    cand.append(val)
            cand.sort()
            
        # Fast filter before full evaluation
        # If we want k <= 4, check for 5-Sidon
        if best_k is None or best_k > 4:
            if _no_5_sidon_check(cand):
                k = _max_sidon_size(cand)
                if best_k is None or k < best_k:
                    best = cand
                    best_k = k
                    if best_k <= 3:
                        break
            else:
                # k >= 5
                # Only update if best_k is 6 and this is 5?
                # We don't know if it's 5 or 6.
                # If best_k is 6, we might want to evaluate it to see if it's 5.
                # But _no_5_sidon_check is expensive.
                # Let's just skip for now to save time, or evaluate if we have time.
                if best_k is not None and best_k == 6 and time.time() < deadline - 20:
                    k = _max_sidon_size(cand)
                    if k < best_k:
                        best = cand
                        best_k = k
                        if best_k <= 3:
                            break
        else:
            # best_k <= 4
            # We want k < best_k.
            # If best_k is 4, we want 3.
            # Check for 4-Sidon?
            has_4 = False
            if best_k == 4:
                for combo in itertools.combinations(cand, 4):
                    if _fast_is_sidon(list(combo)):
                        has_4 = True
                        break
            if not has_4:
                k = _max_sidon_size(cand)
                if k < best_k:
                    best = cand
                    best_k = k
                    if best_k <= 3:
                        break
        
        # Perform local search from this candidate
        if time.time() < deadline - 10:
            improved, ik = _local_search(cand, rng, deadline - 5, best_k if best_k is not None else 12)
            if best_k is None or ik < best_k:
                best = improved
                best_k = ik
                if best_k <= 3:
                    break
    
    if best is None:
        return list(range(12))
    return best

def construct():
    return _select_best_ambient()
# EVOLVE-BLOCK-END
