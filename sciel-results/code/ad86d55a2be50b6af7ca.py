"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _fast_is_sidon(lst):
    """Fast Sidon check for sorted list with early termination."""
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
    for k in range(min(n, 6), 1, -1):
        for combo in itertools.combinations(ambient, k):
            if _fast_is_sidon(list(combo)):
                return k
    return 2

def _random_ambient(rng):
    return sorted(rng.sample(range(129), 12))

def _local_search(ambient, rng, deadline, current_best_k, max_iters=300):
    """Local search with simulated annealing-style acceptance."""
    best = list(ambient)
    best_k = current_best_k
    no_improve = 0
    temperature = 0.5
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if best_k <= 3:
            break
        
        idx = rng.randint(0, 11)
        new_val = rng.randint(0, 128)
        if new_val in best:
            no_improve += 1
            continue
        
        trial = best[:]
        trial[idx] = new_val
        trial.sort()
        k = _max_sidon_size(trial)
        
        if k < best_k:
            best = trial
            best_k = k
            no_improve = 0
            temperature = 0.5
        else:
            if k == best_k and rng.random() < 0.15:
                best = trial
                no_improve = 0
            else:
                no_improve += 1
                temperature *= 0.995
                if temperature < 0.01:
                    temperature = 0.01
        
        if no_improve >= 100:
            break
    
    return best, best_k

def _candidates():
    """Generate diverse initial candidates."""
    cands = []
    for step in [2, 3, 4, 5, 6, 8, 10, 11, 12]:
        cand = list(range(0, min(129, step * 12), step))
        if len(cand) == 12 and cand[-1] <= 128:
            cands.append(cand)
    cands.append(list(range(12)))
    cands.append([0,1,2,3,4,5,6,7,8,9,10,12])
    cands.append([0,1,2,3,4,5,6,7,8,9,11,13])
    cands.append(list(range(1, 13)))
    cands.append([0,2,4,6,8,10,12,14,16,18,20,22])
    cands.append([0,1,3,5,7,9,11,13,15,17,19,21])
    cands.append([0,1,2,4,6,8,10,12,14,16,18,20])
    cands.append([0,2,3,5,6,8,9,11,12,14,15,17])
    cands.append([0,1,3,4,6,7,9,10,12,13,15,16])
    cands.append([0,2,4,5,7,9,10,12,14,15,17,19])
    cands.append([0,1,4,5,8,9,12,13,16,17,20,21])
    cands.append([0,2,5,7,10,12,15,17,20,22,25,27])
    cands.append([0,1,3,6,10,15,21,28,36,45,55,66])
    cands.append([0,1,2,4,8,16,32,64,96,112,120,128])
    cands.append([0,1,2,3,5,8,13,21,34,55,89,128])
    cands.append([0,1,4,10,20,35,56,84,120,128,127,126])
    valid_cands = []
    for c in cands:
        if len(c) == 12 and len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
            valid_cands.append(sorted(c))
    return valid_cands

def _select_best_ambient():
    start_time = time.time()
    deadline = start_time + 85
    
    best = None
    best_k = None
    
    for cand in _candidates():
        if time.time() > deadline - 2:
            break
        k = _max_sidon_size(cand)
        if best_k is None or k < best_k:
            best = cand
            best_k = k
            if best_k <= 3:
                break
    
    rng = random.Random(42)
    trials_done = 0
    max_trials = 1000
    
    while trials_done < max_trials:
        if time.time() > deadline - 3:
            break
        if best_k is not None and best_k <= 3:
            break
        
        cand = _random_ambient(rng)
        k = _max_sidon_size(cand)
        trials_done += 1
        
        if best_k is None or k < best_k:
            best = cand
            best_k = k
            if best_k <= 3:
                break
        
        if time.time() < deadline - 5:
            improved, ik = _local_search(cand, rng, deadline - 3, best_k if best_k is not None else 12)
            trials_done += 1
            if best_k is None or ik < best_k:
                best = improved
                best_k = ik
                if best_k <= 3:
                    break
    
    if best is not None and best_k > 3 and time.time() < deadline - 1:
        improved, ik = _local_search(best, rng, deadline - 0.5, best_k, max_iters=500)
        if ik < best_k:
            best = improved
            best_k = ik
    
    if best is None:
        return list(range(12))
    return best

def construct():
    return _select_best_ambient()
# EVOLVE-BLOCK-END
