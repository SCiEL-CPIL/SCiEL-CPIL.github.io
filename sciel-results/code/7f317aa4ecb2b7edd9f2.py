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

def _greedy_construct(order):
    """Construct a dissociated set using a specific order of candidates."""
    S = []
    sums = {0}
    for x in order:
        v = _LCM // x
        is_valid = True
        for s in sums:
            if (s + v) in sums:
                is_valid = False
                break
        
        if is_valid:
            S.append(x)
            sums |= {s + v for s in sums}
    return S

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
        sorted(universe)
    ]
    
    # Random orders
    # 30 random attempts are cheap and provide diversity
    for _ in range(30):
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
    
    # 2. Local Search: Try to increase size by swapping 1 element for 2.
    # Current size K. Try S' = S - {a} + {x, y} where x, y not in S.
    # This increases size by 1.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        in_set = set(best_S)
        not_in_set = [x for x in universe if x not in in_set]
        
        if not not_in_set:
            break
            
        moves = []
        for a in best_S:
            for x, y in itertools.combinations(not_in_set, 2):
                moves.append((a, x, y))
        
        random.shuffle(moves)
        
        for a, x, y in moves:
            if time.time() > deadline - 2:
                break
            
            candidate = [z for z in best_S if z != a] + [x, y]
            
            if _is_dissociated(candidate):
                best_S = candidate
                best_size += 1
                improved = True
                break
        
        if improved:
            continue

    return best_S
