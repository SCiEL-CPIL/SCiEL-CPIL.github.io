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

def _get_addable(S, sums, N):
    """Return list of elements in [1,N] not in S that can be added while preserving Sidon."""
    S_set = set(S)
    addable = []
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if adding x preserves Sidon property
        new_sums = [2 * x] + [x + a for a in S]
        if len(set(new_sums)) != len(new_sums):
            continue
        if sums.intersection(new_sums):
            continue
        addable.append(x)
    return addable

def _is_maximal(S, N):
    """Check if S is inclusion-maximal Sidon in [1,N]."""
    sums = _pair_sums(S)
    addable = _get_addable(S, sums, N)
    return len(addable) == 0

def _complete_to_maximal(S, N):
    """Force S to be maximal by greedily adding elements that block the most others."""
    S = sorted(S)
    sums = _pair_sums(S)
    S_set = set(S)
    
    while True:
        addable = _get_addable(S, sums, N)
        if not addable:
            break
        
        # For each addable x, count how many other addable elements it would block
        best_x = None
        best_blocks = -1
        for x in addable:
            new_sums_x = set([2 * x] + [x + a for a in S])
            blocked = 0
            for y in addable:
                if y == x:
                    continue
                new_sums_y = [2 * y] + [y + a for a in S]
                if len(set(new_sums_y)) != len(new_sums_y):
                    continue
                if new_sums_x.intersection(new_sums_y):
                    blocked += 1
            if blocked > best_blocks:
                best_blocks = blocked
                best_x = x
        
        if best_x is None:
            best_x = addable[0]
        
        S.append(best_x)
        S_set.add(best_x)
        new_sums = [2 * best_x] + [best_x + a for a in S if a != best_x]
        sums.update(new_sums)
        S = sorted(S)
    
    return S

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

def _local_search_minimize(S, N, deadline):
    """Local search: try removing elements, then re-complete, to minimize size."""
    S = sorted(S)
    best = S[:]
    best_size = len(S)
    
    while time.time() < deadline:
        # Try removing each element
        improved = False
        for i in range(len(best)):
            if time.time() > deadline:
                break
            candidate = best[:i] + best[i+1:]
            if not candidate:
                continue
            if not _is_sidon(candidate):
                continue
            
            # Complete to maximal
            completed = _complete_to_maximal(candidate, N)
            if len(completed) < best_size:
                best = completed
                best_size = len(completed)
                improved = True
        
        if not improved:
            break
    
    return best

def construct():
    deadline = time.time() + 55.0  # safety margin
    best = None
    best_size = float('inf')
    
    # First: try the seed's ascending greedy as fallback
    asc_order = list(range(1, N + 1))
    S_asc = _greedy_random(asc_order, N)
    if _is_sidon(S_asc):
        S_asc = _complete_to_maximal(S_asc, N)
        if _is_maximal(S_asc, N):
            best = S_asc
            best_size = len(S_asc)
    
    # Try to reduce
    if best is not None:
        reduced = _local_search_minimize(best, N, deadline)
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
        
        # Complete to maximal
        S = _complete_to_maximal(S, N)
        if not _is_maximal(S, N):
            continue
        
        size = len(S)
        if size >= best_size:
            continue
        
        # Try to reduce via local search
        reduced = _local_search_minimize(S, N, deadline)
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
