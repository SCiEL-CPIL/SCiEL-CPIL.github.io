"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
_DEADLINE = 55.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START
def _is_sidon(S):
    """Check if S is a Sidon set (all a+b for a<=b distinct)."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(S, N):
    """Check if S is inclusion-maximal Sidon in [1,N]."""
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if adding x preserves Sidon
        ok = True
        new_sums = [x + a for a in S] + [2 * x]
        # Check distinctness among new_sums
        if len(set(new_sums)) != len(new_sums):
            ok = False
        else:
            # Check new_sums against existing sums
            for s in new_sums:
                if s in sums:
                    ok = False
                    break
        if ok:
            return False  # Can extend, not maximal
    return True

def _sidon_and_maximal(S, N):
    if not S:
        return False
    if not _is_sidon(S):
        return False
    if not _is_maximal(S, N):
        return False
    return True

def _greedy_sidon(randomize=False, seed=None):
    """Build a Sidon set greedily. If randomize, shuffle candidate order."""
    if seed is not None:
        random.seed(seed)
    selected = []
    sums = set()
    candidates = list(range(1, N + 1))
    if randomize:
        random.shuffle(candidates)
    for x in candidates:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    selected.sort()
    return selected

def _try_minimize_by_removal(S, N):
    """Try to remove elements while keeping Sidon + maximal. Returns minimal size set found."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        for i in range(len(S)):
            candidate = S[:i] + S[i+1:]
            if _sidon_and_maximal(candidate, N):
                S = candidate
                improved = True
                break
    return S

def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1
    
    # Phase 1: Try several random greedy constructions, then minimize
    for seed in range(2000):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 5:  # leave 5s safety
            break
        S = _greedy_sidon(randomize=True, seed=seed)
        if len(S) < best_size:
            # Try to minimize further
            S_min = _try_minimize_by_removal(S, N)
            if _sidon_and_maximal(S_min, N) and len(S_min) < best_size:
                best_size = len(S_min)
                best_set = S_min
            if elapsed > _DEADLINE - 10:
                break
    
    # If nothing found, use deterministic greedy
    if best_set is None:
        S = _greedy_sidon(randomize=False)
        S_min = _try_minimize_by_removal(S, N)
        if _sidon_and_maximal(S_min, N):
            best_set = S_min
        else:
            best_set = S
    
    return best_set
# EVOLVE-BLOCK-END
