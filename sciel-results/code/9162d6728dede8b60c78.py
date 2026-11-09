import time
import itertools
import random

_LCM = 232792560
_MOD = 1000000007  # Large prime for modular filtering

def _get_sums(subset):
    """Calculate all subset sums of the scaled reciprocals."""
    scaled = [_LCM // x for x in subset]
    sums = {0}
    for v in scaled:
        sums |= {s + v for s in sums}
    return sums

def _is_dissociated(subset):
    """Check if the set is dissociated."""
    sums = _get_sums(subset)
    return len(sums) == (1 << len(subset))

def _mod_subset_sums(subset, mod=_MOD):
    """Compute all subset sums modulo mod. Returns the set of sums."""
    scaled = [(_LCM // x) % mod for x in subset]
    sums = {0}
    for v in scaled:
        sums = {(s + v) % mod for s in sums} | sums
    return sums

def _modular_dissociated(subset, mod=_MOD):
    """
    Necessary condition: all subset sums distinct modulo mod.
    Fast filter - if this fails, the set is definitely not dissociated.
    """
    n = len(subset)
    sums = _mod_subset_sums(subset, mod)
    return len(sums) == (1 << n)

def _fast_check(subset):
    """Two-stage: modular filter then exact check."""
    if not _modular_dissociated(subset):
        return False
    return _is_dissociated(subset)

def _greedy_randomized(universe, deadline):
    """Greedy construction with random order for diversity."""
    S = []
    exact_sums = {0}
    
    candidates = list(universe)
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        v = _LCM // x
        
        is_valid = True
        for s in exact_sums:
            if (s + v) in exact_sums:
                is_valid = False
                break
        
        if is_valid:
            S.append(x)
            exact_sums |= {s + v for s in exact_sums}
    
    return S

def _local_search_1for2(best_S, universe, deadline):
    """Local search: try 1-for-2 swaps to increase size."""
    best_S = list(best_S)
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        in_set = set(best_S)
        not_in_set = [x for x in universe if x not in in_set]
        
        moves = []
        for a in best_S:
            for x, y in itertools.combinations(not_in_set, 2):
                moves.append((a, x, y))
        
        random.shuffle(moves)
        
        for a, x, y in moves:
            if time.time() > deadline - 5:
                break
            
            candidate = [z for z in best_S if z != a] + [x, y]
            if _fast_check(candidate):
                best_S = candidate
                improved = True
                break
        
        if improved:
            continue
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    universe = list(range(1, 23))
    
    # 1. Multiple greedy starts to find good initial solutions
    best_S = []
    best_size = 0
    
    # Descending greedy (deterministic, tends to find large sets)
    S_desc = []
    exact_sums = {0}
    for x in sorted(universe, reverse=True):
        if time.time() > deadline:
            break
        v = _LCM // x
        is_valid = True
        for s in exact_sums:
            if (s + v) in exact_sums:
                is_valid = False
                break
        if is_valid:
            S_desc.append(x)
            exact_sums |= {s + v for s in exact_sums}
    
    if len(S_desc) > best_size:
        best_S = list(S_desc)
        best_size = len(best_S)
    
    # Randomized greedy restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        S_rand = _greedy_randomized(universe, deadline - 30)
        if len(S_rand) > best_size:
            best_S = list(S_rand)
            best_size = len(best_S)
    
    # 2. Local search on the best solution
    best_S = _local_search_1for2(best_S, universe, deadline)
    best_size = len(best_S)
    
    # 3. Try 2-for-3 moves if time remains
    if time.time() < deadline - 15:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            in_set = set(best_S)
            not_in_set = [x for x in universe if x not in in_set]
            
            if len(not_in_set) < 3:
                break
            
            # Limit the number of 2-for-3 moves to try per iteration
            num_tries = 0
            max_tries = 500
            for a, b in itertools.combinations(best_S, 2):
                if time.time() > deadline - 5 or num_tries >= max_tries:
                    break
                for x, y, z in itertools.combinations(not_in_set, 3):
                    if time.time() > deadline - 5 or num_tries >= max_tries:
                        break
                    num_tries += 1
                    candidate = [v for v in best_S if v not in (a, b)] + [x, y, z]
                    if _fast_check(candidate):
                        best_S = candidate
                        improved = True
                        break
                if improved:
                    break
    
    return best_S
