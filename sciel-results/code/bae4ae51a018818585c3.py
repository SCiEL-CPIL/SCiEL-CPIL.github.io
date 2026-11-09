import time
import itertools
import random

_LCM = 232792560

def _get_sums(subset):
    """Calculate all subset sums of the scaled reciprocals."""
    scaled = [_LCM // x for x in subset]
    sums = {0}
    for v in scaled:
        new_sums = set()
        for s in sums:
            new_sums.add(s + v)
        sums = sums | new_sums
    return sums

def _is_dissociated(subset):
    """Check if the set is dissociated."""
    sums = _get_sums(subset)
    return len(sums) == (1 << len(subset))

def _can_add_element(sums, v):
    """Check if adding element with weight v creates a collision."""
    for s in sums:
        if (s + v) in sums:
            return False
    return True

def _greedy_construct(ordering):
    """Greedy construction with a given ordering of candidates."""
    S = []
    sums = {0}
    
    for x in ordering:
        v = _LCM // x
        if _can_add_element(sums, v):
            S.append(x)
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums = sums | new_sums
    
    return S, sums

def _local_search(S, universe, deadline):
    """Local search to increase cardinality via 1-for-2 swaps."""
    best_S = list(S)
    best_size = len(best_S)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        in_set = set(best_S)
        not_in_set = [x for x in universe if x not in in_set]
        
        if len(not_in_set) < 2:
            break
        
        moves = []
        for a in best_S:
            for x, y in itertools.combinations(not_in_set, 2):
                moves.append((a, x, y))
        
        random.shuffle(moves)
        
        for a, x, y in moves:
            if time.time() > deadline - 5:
                break
            
            candidate = [z for z in best_S if z != a] + [x, y]
            if _is_dissociated(candidate):
                best_S = candidate
                best_size += 1
                improved = True
                break
    
    return best_S

def _backtrack_search(universe, target_size, deadline, current_set=None, sums=None, start_idx=0):
    """Backtracking search to find a dissociated set of target_size."""
    if current_set is None:
        current_set = []
        sums = {0}
    
    if len(current_set) == target_size:
        return current_set
    
    if time.time() > deadline:
        return None
    
    for i in range(start_idx, len(universe)):
        x = universe[i]
        v = _LCM // x
        
        if _can_add_element(sums, v):
            new_sums = sums | {s + v for s in sums}
            result = _backtrack_search(universe, target_size, deadline, current_set + [x], new_sums, i + 1)
            if result is not None:
                return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    universe = list(range(1, 23))
    
    # 1. Try multiple greedy orderings
    orderings = [
        sorted(universe, reverse=True),  # Descending
        sorted(universe),  # Ascending
        sorted(universe, key=lambda x: random.random()),  # Random
        sorted(universe, key=lambda x: random.random()),  # Another random
        sorted(universe, key=lambda x: random.random()),  # Yet another random
    ]
    
    best_S = []
    best_size = 0
    
    for ordering in orderings:
        if time.time() > deadline - 30:
            break
        S, _ = _greedy_construct(ordering)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
    
    # 2. Local search to improve
    best_S = _local_search(best_S, universe, deadline)
    best_size = len(best_S)
    
    # 3. Try backtracking to see if we can find a larger set
    # Start from current best size and try to increase by 1
    for target in range(best_size + 1, min(best_size + 3, 17)):
        if time.time() > deadline - 15:
            break
        result = _backtrack_search(universe, target, deadline)
        if result is not None and len(result) > best_size:
            best_S = result
            best_size = len(result)
    
    return best_S
