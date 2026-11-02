"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import itertools

_START_TIME = time.time()
_TIMEOUT = 140.0  # safety margin before 150s wall-clock kill

def _score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or max(U) == 0:
        return 0.0
    s = set(U)
    # Difference set U - U
    diffs = set()
    for a in s:
        for b in s:
            diffs.add(a - b)
    # Sum set U + U
    sums = set()
    for a in s:
        for b in s:
            sums.add(a + b)
    ratio = len(diffs) / len(sums)
    if ratio <= 0:
        return 0.0
    log_arg = 2 * max(s) + 1
    if log_arg <= 1:
        return 0.0
    val = 1.0 + math.log(ratio) / math.log(log_arg)
    return val

def _score_fast(U_list):
    """Faster score using sets, assumes U_list is a list of unique nonneg ints with 0."""
    if not U_list or U_list[0] != 0 or len(U_list) == 1:
        return 0.0
    s = set(U_list)
    diffs = set()
    for a in s:
        for b in s:
            diffs.add(a - b)
    sums = set()
    for a in s:
        for b in s:
            sums.add(a + b)
    ratio = len(diffs) / len(sums)
    if ratio <= 0:
        return 0.0
    log_arg = 2 * max(s) + 1
    if log_arg <= 1:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(log_arg)

def _local_search(U, max_iter=2000, timeout=140.0):
    """Local search: add/remove/swap elements to improve score."""
    random.seed(42)
    best = list(U)
    best_score = _score_fast(best)
    n = len(best)
    
    for it in range(max_iter):
        if time.time() > timeout:
            break
        
        # Choose a random mutation
        r = random.random()
        
        if r < 0.3 and n < 50:
            # Add a new element
            max_el = max(best)
            new_el = random.randint(1, max_el * 2 + 10)
            if new_el not in best:
                candidate = sorted(best + [new_el])
            else:
                continue
        elif r < 0.5 and n > 2:
            # Remove an element (not 0)
            removable = [i for i in range(n) if best[i] != 0]
            if not removable:
                continue
            idx = random.choice(removable)
            candidate = best[:idx] + best[idx+1:]
        elif r < 0.7 and n > 2:
            # Swap an element
            removable = [i for i in range(n) if best[i] != 0]
            if not removable:
                continue
            idx = random.choice(removable)
            max_el = max(best)
            new_el = random.randint(1, max_el * 2 + 10)
            if new_el not in best:
                candidate = best[:idx] + [new_el] + best[idx+1:]
                candidate = sorted(candidate)
            else:
                continue
        else:
            # Perturb an element
            removable = [i for i in range(n) if best[i] != 0]
            if not removable:
                continue
            idx = random.choice(removable)
            delta = random.randint(-5, 5)
            new_val = best[idx] + delta
            if new_val < 0:
                new_val = random.randint(1, 5)
            if new_val not in best:
                candidate = best[:idx] + [new_val] + best[idx+1:]
                candidate = sorted(candidate)
            else:
                continue
        
        cand_score = _score_fast(candidate)
        if cand_score > best_score:
            best = candidate
            best_score = cand_score
            n = len(best)
    
    return best, best_score

def _arithmetic_progression(n, d):
    """Arithmetic progression {0, d, 2d, ..., (n-1)d}."""
    return [i * d for i in range(n)]

def _geometric_like(n, r):
    """Geometric-like {1, r, r^2, ..., r^(n-1)} plus 0."""
    result = [0]
    cur = 1
    for i in range(n - 1):
        result.append(cur)
        cur *= r
    return result

def _modular_residue(M, k, target):
    """Pre-image of target under x mod M == target, in range [0, M*k)."""
    result = [x for x in range(target, M * k, M)]
    if 0 not in result:
        result.append(0)
        result.sort()
    return result

def construct():
    deadline = _START_TIME + _TIMEOUT
    best_set = [0, 1, 3]
    best_score = _score_fast(best_set)
    
    # Phase 1: Try structured families
    candidates = []
    
    # Arithmetic progressions
    for n in range(2, 30):
        for d in range(1, 15):
            U = _arithmetic_progression(n, d)
            candidates.append(U)
    
    # Geometric-like
    for n in range(2, 20):
        for r in range(2, 10):
            U = _geometric_like(n, r)
            candidates.append(U)
    
    # Modular residues
    for M in range(3, 50):
        for k in range(1, 15):
            for target in range(0, min(M, 10)):
                U = _modular_residue(M, k, target)
                if len(U) >= 2:
                    candidates.append(U)
    
    # Random sets
    random.seed(123)
    for trial in range(500):
        n = random.randint(2, 20)
        max_el = random.randint(5, 200)
        U = [0] + sorted(random.sample(range(1, max_el + 1), min(n - 1, max_el)))
        candidates.append(U)
    
    # Evaluate all candidates
    for U in candidates:
        if time.time() > deadline - 20:
            break
        sc = _score_fast(U)
        if sc > best_score:
            best_score = sc
            best_set = list(U)
    
    # Phase 2: Local search from best
    if time.time() < deadline - 30:
        best_set, best_score = _local_search(best_set, max_iter=5000, timeout=deadline - 10)
    
    # Phase 3: More local search rounds from different starting points
    if time.time() < deadline - 15:
        # Try local search from a few other good candidates
        top_candidates = []
        for U in candidates[:200]:
            sc = _score_fast(U)
            top_candidates.append((sc, U))
        top_candidates.sort(reverse=True)
        
        for sc, U in top_candidates[:5]:
            if time.time() > deadline - 10:
                break
            if sc < best_score * 0.95:
                break
            new_set, new_score = _local_search(U, max_iter=2000, timeout=deadline - 5)
            if new_score > best_score:
                best_score = new_score
                best_set = new_set
    
    # Final: ensure 0 is present and sorted
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set.insert(0, 0)
    
    return best_set
# EVOLVE-BLOCK-END
