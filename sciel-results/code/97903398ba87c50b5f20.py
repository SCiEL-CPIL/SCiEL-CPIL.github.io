import time
import itertools
import random

# Use multiple primes for modular hashing to detect collisions efficiently
# Product of these primes > LCM(1..22) * 2^22 to ensure uniqueness
_PRIMES = [1000003, 1000033, 1000037, 1000039]
_NUM_PRIMES = len(_PRIMES)

def _scaled_value(x, prime_idx):
    """Get scaled reciprocal value modulo a prime."""
    # Compute LCM // x mod prime
    # LCM(1..22) = 232792560
    lcm_val = 232792560
    return (lcm_val // x) % _PRIMES[prime_idx]

def _check_dissociated_modular(subset):
    """
    Check if subset is dissociated using modular arithmetic.
    Returns True if all subset sums are distinct (with high probability using multiple primes).
    """
    n = len(subset)
    if n == 0:
        return True
    
    # For each prime, track subset sums
    # We need 2^n distinct sums for each prime
    for p_idx in range(_NUM_PRIMES):
        p = _PRIMES[p_idx]
        sums = {0}
        for x in subset:
            v = _scaled_value(x, p_idx)
            new_sums = set()
            for s in sums:
                ns = (s + v) % p
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums = sums | new_sums
            if len(sums) != (1 << (len(sums) // (1 << (len(subset)-1)))):
                # This check is getting complicated; just check at the end
                pass
        
        if len(sums) != (1 << n):
            return False
    
    return True

def _is_dissociated(subset):
    """Check if the set is dissociated using exact LCM arithmetic."""
    if len(subset) == 0:
        return True
    LCM = 232792560
    scaled = [LCM // x for x in subset]
    sums = {0}
    for v in scaled:
        sums |= {s + v for s in sums}
    return len(sums) == (1 << len(subset))

def _can_add(subset, x, current_sums_lcm):
    """Check if x can be added to subset without creating collisions."""
    LCM = 232792560
    v = LCM // x
    for s in current_sums_lcm:
        if (s + v) in current_sums_lcm:
            return False, None
    new_sums = current_sums_lcm | {s + v for s in current_sums_lcm}
    return True, new_sums

def _greedy_construct(order, deadline):
    """Construct a dissociated set using a specific order of candidates."""
    S = []
    LCM = 232792560
    sums = {0}
    for x in order:
        if time.time() > deadline - 1:
            break
        v = LCM // x
        is_valid = True
        for s in sums:
            if (s + v) in sums:
                is_valid = False
                break
        
        if is_valid:
            S.append(x)
            sums = sums | {s + v for s in sums}
    return S, sums

def _local_search_1_for_1(best_S, universe, deadline):
    """Try to replace one element with another to maintain dissociation."""
    in_set = set(best_S)
    not_in_set = [x for x in universe if x not in in_set]
    
    if not not_in_set:
        return None
    
    moves = [(a, x) for a in best_S for x in not_in_set]
    random.shuffle(moves)
    
    for a, x in moves:
        if time.time() > deadline - 2:
            break
        
        candidate = [z for z in best_S if z != a] + [x]
        
        if _is_dissociated(candidate):
            return candidate
    
    return None

def _local_search_1_for_2(best_S, universe, deadline):
    """Try to replace one element with two others to increase size by 1."""
    in_set = set(best_S)
    not_in_set = [x for x in universe if x not in in_set]
    
    if len(not_in_set) < 2:
        return None
    
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
            return candidate
    
    return None

def _backtracking_search(universe, target_size, time_limit, start_set=None, start_sums=None):
    """
    Attempt to find a dissociated set of target_size using backtracking.
    Returns a set if found, None otherwise.
    """
    start_time = time.time()
    LCM = 232792560
    
    if start_set is None:
        start_set = []
    if start_sums is None:
        start_sums = {0}
    
    def dfs(current_set, current_sums, remaining, depth):
        if time.time() - start_time > time_limit:
            return None
            
        if len(current_set) == target_size:
            return list(current_set)
        
        # Pruning: if we can't reach target_size with remaining elements
        if len(current_set) + len(remaining) < target_size:
            return None
        
        # Heuristic: try elements that are "far apart" in value
        # Sort remaining by scaled value to try diverse candidates
        sorted_remaining = sorted(remaining, key=lambda x: LCM // x)
        
        for i, x in enumerate(sorted_remaining):
            v = LCM // x
            is_valid = True
            for s in current_sums:
                if (s + v) in current_sums:
                    is_valid = False
                    break
            
            if is_valid:
                new_sums = current_sums | {s + v for s in current_sums}
                next_remaining = [r for j, r in enumerate(remaining) if j != i]
                result = dfs(current_set + [x], new_sums, next_remaining, depth + 1)
                if result is not None:
                    return result
                    
        return None

    # Try multiple random initial orderings
    for attempt in range(200):
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
    
    # Known good structural sets to try first
    # Powers of 2 up to 16: {1, 2, 4, 8, 16}
    # Primes: {2, 3, 5, 7, 11, 13, 17, 19}
    structural_sets = [
        [1, 2, 4, 8, 16],
        [2, 3, 5, 7, 11, 13, 17, 19],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    ]
    
    # 1. Try structural sets first
    for struct_set in structural_sets:
        if time.time() > deadline - 20:
            break
        if _is_dissociated(struct_set):
            if len(struct_set) > best_size:
                best_size = len(struct_set)
                best_S = list(struct_set)
    
    # 2. Try multiple greedy orderings to find a strong initial set
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        sorted(universe, key=lambda x: -x),
        sorted(universe, key=lambda x: x),
    ]
    
    # Add some structured orders
    orders.append(list(range(22, 0, -1)))  # 22 down to 1
    orders.append(list(range(1, 23)))      # 1 up to 22
    
    # Random orders
    for _ in range(50):
        random_order = universe.copy()
        random.shuffle(random_order)
        orders.append(random_order)
    
    for order in orders:
        if time.time() > deadline - 15:
            break
        S, _ = _greedy_construct(order, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # 3. Local Search: Try to increase size
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # First try 1-for-1 swaps (maintains size)
        if time.time() < deadline - 10:
            candidate = _local_search_1_for_1(best_S, universe, deadline)
            if candidate is not None:
                # Check if it's actually better (shouldn't be for 1-for-1, but verify)
                if _is_dissociated(candidate) and len(candidate) == best_size:
                    # This is a lateral move; only accept if it leads to improvement later
                    # For now, skip lateral moves unless we're stuck
                    pass
        
        # Then try 1-for-2 swaps (increases size by 1)
        candidate = _local_search_1_for_2(best_S, universe, deadline)
        if candidate is not None:
            best_S = candidate
            best_size += 1
            improved = True
        
        if improved:
            continue
    
    # 4. If we haven't reached size 17, try backtracking for size 17
    if best_size < 17 and time.time() < deadline - 5:
        time_limit = min(30, deadline - 5 - time.time())
        if time_limit > 1:
            result = _backtracking_search(universe, 17, time_limit)
            if result is not None:
                best_S = result
                best_size = 17
    
    # 5. Final verification and return
    if not best_S:
        # Fallback: return a known valid small set
        best_S = [1, 2, 4, 8, 16]
    
    # Verify the result is actually dissociated
    if not _is_dissociated(best_S):
        # Find a valid subset
        for size in range(len(best_S), 0, -1):
            for subset in itertools.combinations(best_S, size):
                if _is_dissociated(list(subset)):
                    best_S = list(subset)
                    break
            if best_S and _is_dissociated(best_S):
                break
        if not best_S or not _is_dissociated(best_S):
            best_S = [1, 2, 4, 8, 16]
    
    return best_S
