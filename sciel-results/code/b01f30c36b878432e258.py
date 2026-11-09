import time
import itertools
import random

_LCM = 232792560

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

def _can_add(v, sums):
    """Check if value v can be added to the set with given sums without collision."""
    for s in sums:
        if (s + v) in sums:
            return False
    return True

def _add_to_sums(sums, v):
    """Add value v to the sum set, returning the new set."""
    return sums | {s + v for s in sums}

def _greedy_construct(order):
    """Construct a dissociated set using a specific order of candidates."""
    S = []
    sums = {0}
    for x in order:
        v = _LCM // x
        if _can_add(v, sums):
            S.append(x)
            sums = _add_to_sums(sums, v)
    return S

def _backtracking_search(universe, target_size, time_limit, random_seed=None):
    """
    Attempt to find a dissociated set of target_size using backtracking.
    Returns a set if found, None otherwise.
    """
    if random_seed is not None:
        random.seed(random_seed)
        
    start_time = time.time()
    
    def dfs(current_set, current_sums, remaining, depth):
        nonlocal start_time
        if time.time() - start_time > time_limit:
            return None
            
        if len(current_set) == target_size:
            return list(current_set)
            
        for i, x in enumerate(remaining):
            v = _LCM // x
            if _can_add(v, current_sums):
                new_sums = _add_to_sums(current_sums, v)
                next_remaining = remaining[:i] + remaining[i+1:]
                result = dfs(current_set + [x], new_sums, next_remaining, depth + 1)
                if result is not None:
                    return result
                    
        return None

    # Try multiple random orders
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
            
        order = universe.copy()
        random.shuffle(order)
        
        result = dfs([], {0}, order, 0)
        if result is not None:
            return result
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    universe = list(range(1, 23))
    
    best_S = []
    best_size = 0
    
    # 1. Try multiple greedy orderings to find a strong initial set
    # Deterministic orders
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        sorted(universe, key=lambda x: _LCM // x),  # Ascending scaled values
        sorted(universe, key=lambda x: _LCM // x, reverse=True)  # Descending scaled values
    ]
    
    # Random orders
    for _ in range(50):
        random_order = universe.copy()
        random.shuffle(random_order)
        orders.append(random_order)
    
    for order in orders:
        if time.time() > deadline - 15:
            break
        S = _greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # 2. Local Search: Try to increase size by swapping.
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        in_set = set(best_S)
        not_in_set = [x for x in universe if x not in in_set]
        
        if not not_in_set:
            break
            
        # Try 1-for-2 swaps (increase size by 1)
        moves_1for2 = []
        for a in best_S:
            for x, y in itertools.combinations(not_in_set, 2):
                moves_1for2.append((a, x, y))
        
        random.shuffle(moves_1for2)
        
        for a, x, y in moves_1for2:
            if time.time() > deadline - 5:
                break
            
            candidate = [z for z in best_S if z != a] + [x, y]
            
            if _is_dissociated(candidate):
                best_S = candidate
                best_size += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try 2-for-3 swaps (increase size by 1 with more flexibility)
        if len(not_in_set) >= 3 and len(best_S) >= 2:
            moves_2for3 = []
            for a, b in itertools.combinations(best_S, 2):
                for x, y, z in itertools.combinations(not_in_set, 3):
                    moves_2for3.append((a, b, x, y, z))
            
            random.shuffle(moves_2for3)
            
            for a, b, x, y, z in moves_2for3:
                if time.time() > deadline - 5:
                    break
                
                candidate = [c for c in best_S if c != a and c != b] + [x, y, z]
                
                if _is_dissociated(candidate):
                    best_S = candidate
                    best_size += 1
                    improved = True
                    break
    
    # 3. If we haven't reached size 17, try backtracking for size 17
    if best_size < 17 and time.time() < deadline - 5:
        time_limit = min(25, deadline - 5 - time.time())
        if time_limit > 1:
            result = _backtracking_search(universe, 17, time_limit)
            if result is not None:
                best_S = result
                best_size = 17
    
    # 4. If we haven't reached size 18, try backtracking for size 18
    if best_size < 18 and time.time() < deadline - 3:
        time_limit = min(10, deadline - 3 - time.time())
        if time_limit > 1:
            result = _backtracking_search(universe, 18, time_limit)
            if result is not None:
                best_S = result
                best_size = 18
    
    return best_S
