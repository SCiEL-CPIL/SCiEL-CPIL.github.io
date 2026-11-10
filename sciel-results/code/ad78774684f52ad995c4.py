"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _largest_sidon(s):
    """Exact largest subset of s with all unordered pair sums (incl. 2x) unique."""
    n = len(s)
    if n == 0:
        return 0
    best = 0
    # Backtracking with pruning
    def bt(idx, chosen, sumset):
        nonlocal best
        rem = n - idx
        if len(chosen) + rem <= best:
            return
        if idx == n:
            if len(chosen) > best:
                best = len(chosen)
            return
        # Option 1: skip idx
        bt(idx + 1, chosen, sumset)
        # Option 2: take idx if no collision
        v = s[idx]
        new_sums = []
        ok = True
        if (v + v) in sumset:
            ok = False
        else:
            new_sums.append(v + v)
            for j in chosen:
                ss = v + s[j]
                if ss in sumset:
                    ok = False
                    break
                new_sums.append(ss)
        if ok:
            for ns in new_sums:
                sumset.add(ns)
            chosen.append(idx)
            bt(idx + 1, chosen, sumset)
            chosen.pop()
            for ns in new_sums:
                sumset.remove(ns)
    bt(0, [], set())
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin under 90s kill
    
    def try_set(vals):
        nonlocal best_set, best_val
        vals = sorted(set(vals))
        if len(vals) != 12:
            return
        v = _largest_sidon(vals)
        if v < best_val:
            best_val = v
            best_set = list(vals)
    
    best_set = None
    best_val = 10**9
    
    # Start with dense structured candidates
    structured = [
        list(range(12)),
        list(range(1, 13)),
        list(range(2, 14)),
        [0,1,2,3,4,5,6,7,8,9,10,11],
        [0,1,2,3,4,5,6,7,8,9,10,12],
        [0,1,2,3,4,5,6,7,8,9,11,12],
        [0,1,2,3,4,5,6,7,8,10,11,12],
        [0,1,2,3,4,5,6,7,9,10,11,12],
        [0,1,2,3,4,5,6,8,9,10,11,12],
        [0,1,2,3,4,5,7,8,9,10,11,12],
        [0,1,2,3,4,6,7,8,9,10,11,12],
        [0,1,2,3,5,6,7,8,9,10,11,12],
        [0,1,2,4,5,6,7,8,9,10,11,12],
        [0,1,3,4,5,6,7,8,9,10,11,12],
        [0,2,3,4,5,6,7,8,9,10,11,12],
    ]
    
    for sc in structured:
        if time.time() > deadline - 5:
            break
        try_set(sc)
    
    if best_val <= 3:
        return best_set
    
    # Local search: try single-element replacements
    rng = random.Random(42)
    current = list(best_set)
    current_score = best_val
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try all single-element replacements
        for i in range(12):
            if time.time() > deadline - 5:
                break
            old_val = current[i]
            # Try values in a neighborhood first, then broader
            candidates = []
            base = old_val
            for d in range(-15, 16):
                nv = base + d
                if 0 <= nv <= 128 and nv != old_val and nv not in current:
                    candidates.append(nv)
            # Add some random values
            for _ in range(20):
                nv = rng.randint(0, 128)
                if nv != old_val and nv not in current:
                    candidates.append(nv)
            
            for new_val in candidates:
                trial = current.copy()
                trial[i] = new_val
                trial.sort()
                trial_score = _largest_sidon(trial)
                if trial_score < current_score:
                    current = trial
                    current_score = trial_score
                    best_set = list(current)
                    best_val = current_score
                    improved = True
                    break
                if trial_score == current_score and rng.random() < 0.1:
                    # Accept equal with small probability for diversity
                    current = trial
                    best_set = list(current)
                    improved = True
                    break
            if improved:
                break
        
        # If no improvement, try random perturbation
        if not improved:
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                i = rng.randint(0, 11)
                new_val = rng.randint(0, 128)
                if new_val in current:
                    continue
                trial = current.copy()
                trial[i] = new_val
                trial.sort()
                trial_score = _largest_sidon(trial)
                if trial_score < current_score:
                    current = trial
                    current_score = trial_score
                    best_set = list(current)
                    best_val = current_score
                    improved = True
                    break
    
    if best_set is None:
        return list(range(12))
    return sorted(best_set)
# EVOLVE-BLOCK-END
