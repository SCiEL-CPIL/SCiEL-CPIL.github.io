"""Construct a finite set U of nonnegative integers (containing 0) to maximize
   1 + log(|U-U|/|U+U|) / log(2*max(U)+1).
"""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

_DEADLINE = 110.0  # safety margin before 120s kill


def _sumset_size(U):
    """Return number of distinct sums a+b for a,b in U."""
    sums = set()
    n = len(U)
    for i in range(n):
        ai = U[i]
        for j in range(i, n):
            sums.add(ai + U[j])
    return len(sums)


def _diffset_size(U):
    """Return number of distinct differences a-b for a,b in U."""
    diffs = set()
    n = len(U)
    for i in range(n):
        ai = U[i]
        for j in range(n):
            diffs.add(ai - U[j])
    return len(diffs)


def _score(U):
    """Compute the task score for a set U (list of nonnegative ints, 0 in U)."""
    if not U or 0 not in U:
        return -1.0
    U = sorted(set(U))
    mx = U[-1]
    if mx == 0:
        return 1.0
    ss = _sumset_size(U)
    ds = _diffset_size(U)
    if ss == 0:
        return 1.0
    ratio = ds / ss
    if ratio <= 0:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)


def _local_search(U, time_limit):
    """Perform local search to improve score of U."""
    best_U = list(U)
    best_score = _score(best_U)
    t_start = time.time()
    
    # To speed up, we can precompute or use a faster method if n is large,
    # but for now, direct calculation is okay for n < 50.
    
    # We will try to add, remove, or replace elements.
    # To keep max(U) small, we limit the search range.
    # Let's say we search for new elements up to 2 * max(U) or some fixed bound.
    
    improved = True
    while improved and time.time() < t_start + time_limit:
        improved = False
        mx = best_U[-1]
        
        # Candidates to add: 1 to 2*mx + 10
        # Candidates to remove: all except 0
        # Candidates to replace: pairs (x, y)
        
        # 1. Try adding an element
        # We scan a range of potential new elements.
        # To avoid O(N^2) overhead in Python, we limit the number of candidates.
        # Let's try a few random candidates and a few systematic ones.
        
        # Systematic: try small integers not in U
        # And some larger ones.
        
        candidates_add = set()
        # Small integers
        for i in range(1, min(50, 2*mx + 10)):
            if i not in best_U:
                candidates_add.add(i)
        # Random integers in range
        for _ in range(20):
            val = random.randint(1, max(10, 2*mx))
            candidates_add.add(val)
            
        for c in candidates_add:
            if time.time() > t_start + time_limit:
                return best_U, best_score
            new_U = best_U + [c]
            new_U.sort()
            s = _score(new_U)
            if s > best_score:
                best_score = s
                best_U = new_U
                improved = True
        
        # 2. Try removing an element
        for i, val in enumerate(best_U):
            if val == 0:
                continue
            if time.time() > t_start + time_limit:
                return best_U, best_score
            new_U = best_U[:i] + best_U[i+1:]
            s = _score(new_U)
            if s > best_score:
                best_score = s
                best_U = new_U
                improved = True
                break # restart loop with new best_U
        
        # 3. Try replacing an element
        # This is more expensive. Only do it if we have time.
        if time.time() < t_start + time_limit - 5:
            for i, val in enumerate(best_U):
                if val == 0:
                    continue
                # Try replacing val with some other number
                # Let's try a few random replacements
                for _ in range(5):
                    new_val = random.randint(1, max(10, 2*mx))
                    if new_val in best_U:
                        continue
                    new_U = best_U.copy()
                    new_U[i] = new_val
                    new_U.sort()
                    s = _score(new_U)
                    if s > best_score:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue

    return best_U, best_score


def construct():
    t0 = time.time()
    best_U = [0, 1, 3]
    best_score = _score(best_U)

    def consider(U):
        nonlocal best_U, best_score
        if time.time() > _DEADLINE - 3:
            return
        try:
            s = _score(U)
        except Exception:
            return
        if s > best_score:
            best_score = s
            best_U = list(U)

    # 1. Generate diverse initial candidates
    candidates = []
    
    # Geometric
    for base in range(2, 10):
        for k in range(2, 20):
            U = [0]
            v = 1
            for _ in range(k):
                v *= base
                U.append(v)
            candidates.append(U)
            
    # Quadratic
    for k in range(2, 20):
        U = [i*i for i in range(k)]
        candidates.append(U)
        
    # Triangular
    for k in range(2, 20):
        U = [i*(i+1)//2 for i in range(k)]
        candidates.append(U)
        
    # Random geometric-ish
    for seed in range(50):
        rng = random.Random(seed)
        U = [0]
        v = rng.randint(1, 5)
        U.append(v)
        for _ in range(rng.randint(3, 15)):
            v = int(v * rng.uniform(1.1, 2.0))
            U.append(v)
        candidates.append(sorted(set(U)))
        
    # 2. Local search on each candidate
    # We have a time budget. Let's divide it among candidates.
    # But local search might be expensive.
    # Let's run local search with a small time limit per candidate.
    
    # Sort candidates by initial score to prioritize good ones?
    # Or just iterate.
    
    time_per_candidate = 0.5 # seconds
    for U in candidates:
        if time.time() > _DEADLINE - 10:
            break
        # Quick score check
        s_init = _score(U)
        if s_init < best_score - 0.1:
            continue # unlikely to improve much
        
        U_opt, s_opt = _local_search(U, time_per_candidate)
        consider(U_opt)
        
    # 3. If time remains, do a global local search on the best found
    if time.time() < _DEADLINE - 5:
        U_opt, s_opt = _local_search(best_U, _DEADLINE - time.time() - 5)
        consider(U_opt)

    return best_U
# EVOLVE-BLOCK-END
