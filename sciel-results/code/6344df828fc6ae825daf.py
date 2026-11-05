"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def _pair_sums(S):
    """Return set of all a+b for a<=b in S."""
    sums = set()
    n = len(S)
    for i in range(n):
        a = S[i]
        sums.add(2 * a)
        for j in range(i + 1, n):
            sums.add(a + S[j])
    return sums

def _is_sidon(S):
    """Check if S is a Sidon set (all pair sums a+b, a<=b, distinct)."""
    seen = set()
    n = len(S)
    for i in range(n):
        a = S[i]
        s = 2 * a
        if s in seen:
            return False
        seen.add(s)
        for j in range(i + 1, n):
            s = a + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _is_maximal(S, N):
    """Check if S is inclusion-maximal Sidon in [1,N]."""
    S_set = set(S)
    sums = _pair_sums(S)
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if adding x preserves Sidon property
        new_sums = [2 * x] + [x + a for a in S]
        # Check no duplicates within new_sums and no overlap with sums
        if len(set(new_sums)) != len(new_sums):
            continue
        if sums.intersection(new_sums):
            continue
        # x can be added -> not maximal
        return False
    return True

def _try_reduce(S, N):
    """Try to reduce S while maintaining Sidon + maximal property."""
    S = sorted(S)
    improved = True
    while improved:
        improved = False
        for i in range(len(S)):
            candidate = S[:i] + S[i+1:]
            if not candidate:
                continue
            if _is_sidon(candidate) and _is_maximal(candidate, N):
                S = candidate
                improved = True
                break
    return S

def _greedy_random(order, N):
    """Greedy Sidon construction following given order."""
    selected = []
    sums = set()
    for x in order:
        new_sums = [2 * x] + [x + a for a in selected]
        if len(set(new_sums)) != len(new_sums):
            continue
        if sums.intersection(new_sums):
            continue
        selected.append(x)
        sums.update(new_sums)
    return selected

def construct():
    deadline = time.time() + 55.0  # safety margin
    best = None
    best_size = float('inf')
    
    # First: try the seed's ascending greedy as fallback
    asc_order = list(range(1, N + 1))
    S_asc = _greedy_random(asc_order, N)
    if _is_sidon(S_asc) and _is_maximal(S_asc, N):
        best = S_asc
        best_size = len(S_asc)
    
    # Try to reduce
    if best is not None:
        reduced = _try_reduce(best, N)
        if len(reduced) < best_size:
            best = reduced
            best_size = len(reduced)
    
    # Multiple random restarts
    rng = random.Random(42)
    for trial in range(5000):
        if time.time() > deadline:
            break
        
        # Random permutation
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        S = _greedy_random(order, N)
        if not S:
            continue
        
        if not _is_sidon(S):
            continue
        
        if not _is_maximal(S, N):
            continue
        
        size = len(S)
        if size >= best_size:
            continue
        
        # Try to reduce
        reduced = _try_reduce(S, N)
        if len(reduced) < size:
            size = len(reduced)
            S = reduced
        
        if size < best_size:
            best_size = size
            best = S
    
    # Final fallback: if best is None, use ascending
    if best is None:
        best = S_asc
    
    return best

# EVOLVE-BLOCK-END
