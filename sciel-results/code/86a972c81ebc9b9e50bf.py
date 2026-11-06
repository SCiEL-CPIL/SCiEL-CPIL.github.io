import time
import random

def _score_bitset(mask):
    """Count distinct d>0 such that a, a+d, a+2d all in set."""
    count = 0
    # d max is 255
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _arr_to_mask(arr):
    mask = 0
    for x in arr:
        mask |= (1 << (x - 1))
    return mask

def _sa_bitset(arr, deadline, rng):
    """Simulated annealing using bitset representation."""
    n = 32
    cur_arr = list(arr)
    cur_mask = _arr_to_mask(cur_arr)
    cur_score = _score_bitset(cur_mask)
    best_score = cur_score
    best_arr = cur_arr[:]

    T = 10.0
    T_min = 0.1
    alpha = 0.995
    steps_since_best = 0
    max_steps_since_best = 2000

    while True:
        if time.time() > deadline:
            break
        if T < T_min:
            break

        # Move: pick an index, change its value
        idx = rng.randrange(n)
        old = cur_arr[idx]
        # Try a random new value, biased towards nearby values sometimes
        if rng.random() < 0.5:
            new = rng.randint(1, 512)
        else:
            shift = rng.choice([-10, -5, -3, -1, 1, 3, 5, 10, 20])
            new = old + shift
            if new < 1 or new > 512:
                continue

        if new == old or (cur_mask & (1 << (new - 1))):
            continue

        trial_mask = (cur_mask & ~(1 << (old - 1))) | (1 << (new - 1))
        trial_score = _score_bitset(trial_mask)
        delta = trial_score - cur_score
        
        if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
            cur_arr[idx] = new
            cur_arr.sort()
            cur_mask = trial_mask
            cur_score = trial_score
            if cur_score > best_score:
                best_score = cur_score
                best_arr = cur_arr[:]
                steps_since_best = 0
            else:
                steps_since_best += 1
        else:
            steps_since_best += 1

        T *= alpha
        if steps_since_best > max_steps_since_best:
            break

    return best_arr, best_score

def _hill_climb(arr, deadline):
    """Greedy hill climbing."""
    cur_arr = list(arr)
    cur_mask = _arr_to_mask(cur_arr)
    cur_score = _score_bitset(cur_mask)
    best_arr = cur_arr[:]
    best_score = cur_score
    
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for idx in range(32):
            old = cur_arr[idx]
            best_trial_score = cur_score
            best_trial_new = old
            # Try a subset of replacements to speed up
            # Focus on values near the current one or random
            candidates = set()
            for _ in range(50):
                candidates.add(rng.randint(1, 512))
            # Also try neighbors
            for delta in [-5, -3, -1, 1, 3, 5]:
                candidates.add(old + delta)
                
            for new in candidates:
                if new < 1 or new > 512 or new == old or (cur_mask & (1 << (new - 1))):
                    continue
                trial_mask = (cur_mask & ~(1 << (old - 1))) | (1 << (new - 1))
                trial_score = _score_bitset(trial_mask)
                if trial_score > best_trial_score:
                    best_trial_score = trial_score
                    best_trial_new = new
            
            if best_trial_new != old:
                cur_arr[idx] = best_trial_new
                cur_arr.sort()
                cur_mask = (cur_mask & ~(1 << (old - 1))) | (1 << (best_trial_new - 1))
                cur_score = best_trial_score
                best_arr = cur_arr[:]
                best_score = cur_score
                improved = True
                break # Restart loop as array changed
    return best_arr, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    rng = random.Random(42)

    # 1. Start with the known best static solution: Dense block
    # 2. Add other structured candidates
    candidates = []
    candidates.append(list(range(1, 33)))
    candidates.append(list(range(10, 42)))
    candidates.append(list(range(50, 82)))
    
    # Unions of blocks
    candidates.append(list(range(1, 17)) + list(range(33, 49)))
    candidates.append(list(range(1, 9)) + list(range(20, 28)) + list(range(40, 48)) + list(range(60, 68)))
    
    # Random
    for _ in range(5):
        candidates.append(sorted(random.sample(range(1, 513), 32)))

    best_overall = None
    best_overall_score = 0

    # Score all candidates first
    scored_candidates = []
    for c in candidates:
        mask = _arr_to_mask(c)
        score = _score_bitset(mask)
        scored_candidates.append((score, c))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Run SA on the top candidates
    # Budget: ~10 seconds per SA run, top 5 candidates
    time_per_sa = 10.0
    
    for i, (score, arr) in enumerate(scored_candidates[:5]):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        sa_deadline = time.time() + min(time_per_sa, remaining - 2.0)
        
        # Run SA
        sa_arr, sa_score = _sa_bitset(arr, sa_deadline, rng)
        
        if sa_score > best_overall_score:
            best_overall_score = sa_score
            best_overall = sa_arr
            
        # Hill climb on the result
        if time.time() < deadline - 5.0:
            hc_deadline = time.time() + min(5.0, deadline - time.time() - 2.0)
            hc_arr, hc_score = _hill_climb(sa_arr, hc_deadline)
            if hc_score > best_overall_score:
                best_overall_score = hc_score
                best_overall = hc_arr

    if best_overall is None:
        best_overall = list(range(1, 33))
        best_overall_score = _score_bitset(_arr_to_mask(best_overall))

    # Final sanity check
    result = sorted(set(best_overall))
    if len(result) != 32:
        # Fallback
        result = list(range(1, 33))
        
    return result
