import time
from math import gcd

N = 512
DEADLINE = 85.0  # Safety margin

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_conflict_bitsets():
    """Build bitsets of conflicts for each number 1..512."""
    n = N
    conflict_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if _lcm(i, j) > N:
                conflict_mask[i] |= (1 << (j - 1))
                conflict_mask[j] |= (1 << (i - 1))
    return conflict_mask

def _greedy_fill(conflict_mask, order=None):
    """Greedy fill using given order (default: 1..N)."""
    if order is None:
        order = list(range(1, N + 1))
    
    S = set()
    in_set = 0
    for x in order:
        if in_set & (1 << (x - 1)):
            continue
        if (conflict_mask[x] & in_set) == 0:
            S.add(x)
            in_set |= (1 << (x - 1))
    return S

def _local_search_add(conflict_mask, S):
    """Try to add any compatible element to S. Returns (new_S, improved)."""
    in_set = 0
    for x in S:
        in_set |= (1 << (x - 1))
    
    for x in range(1, N + 1):
        if not (in_set & (1 << (x - 1))):
            if (conflict_mask[x] & in_set) == 0:
                new_S = S | {x}
                return new_S, True
    
    return S, False

def _swap_improve(conflict_mask, S):
    """Try to find a swap (remove y, add x) that allows adding more elements."""
    in_set = 0
    for x in S:
        in_set |= (1 << (x - 1))
    
    best_new_S = None
    best_size = len(S)
    
    # For each y in S, try removing it and then greedily filling
    for y in S:
        S_without_y = S - {y}
        in_set_wo_y = 0
        for x in S_without_y:
            in_set_wo_y |= (1 << (x - 1))
        
        # Try to add elements greedily
        new_S = set(S_without_y)
        new_in_set = in_set_wo_y
        for x in range(1, N + 1):
            if not (new_in_set & (1 << (x - 1))):
                if (conflict_mask[x] & new_in_set) == 0:
                    new_S.add(x)
                    new_in_set |= (1 << (x - 1))
        
        if len(new_S) > best_size:
            best_size = len(new_S)
            best_new_S = new_S
    
    if best_new_S is not None:
        return best_new_S, True
    return S, False

def construct():
    start_time = time.time()
    
    # Build conflict graph
    conflict_mask = _build_conflict_bitsets()
    
    # Initial greedy solution
    best_S = _greedy_fill(conflict_mask)
    best_size = len(best_S)
    
    # Try multiple randomized greedy orderings
    import random
    rng = random.Random(42)
    
    for trial in range(50):
        if time.time() > DEADLINE - 15:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S = _greedy_fill(conflict_mask, order)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
    
    # Local search: try to add and swap
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        
        # Try to add
        new_S, add_improved = _local_search_add(conflict_mask, best_S)
        if add_improved:
            best_S = new_S
            best_size = len(best_S)
            improved = True
            continue
        
        # Try swap improve
        new_S, swap_improved = _swap_improve(conflict_mask, best_S)
        if swap_improved:
            best_S = new_S
            best_size = len(best_S)
            improved = True
    
    return list(best_S)
