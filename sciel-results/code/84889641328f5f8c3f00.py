"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from functools import reduce

def _is_dissociated(elems):
    """Check if all subset sums of reciprocals are distinct using integer arithmetic."""
    if not elems:
        return True
    if len(elems) == 1:
        return True
    
    n = len(elems)
    # Calculate LCM
    L = reduce(math.lcm, elems)
    
    # Convert to scaled integers
    weights = [L // e for e in elems]
    
    sums = {0}
    for w in weights:
        new_sums = set()
        for s in sums:
            new_sums.add(s + w)
        # Check for collision between new_sums and sums
        if sums & new_sums:
            return False
        sums = sums | new_sums
        
    return len(sums) == (1 << n)

def _build_sums(elems):
    """Build the set of all subset sums for the given elements (scaled integers).
    Returns (L, weights, sums_set) or None if not dissociated."""
    if not elems:
        return (1, [], {0})
    L = reduce(math.lcm, elems)
    weights = [L // e for e in elems]
    sums = {0}
    for w in weights:
        new_sums = {s + w for s in sums}
        if sums & new_sums:
            return None
        sums |= new_sums
    return (L, weights, sums)

def _can_add_incremental(L, weights, sums, new_elem):
    """Check if we can add new_elem to a set with LCM L, weights, and sums.
    Returns (new_L, new_weights, new_sums) if dissociated, else None."""
    new_L = math.lcm(L, new_elem)
    scale = new_L // L
    # Scale existing weights
    new_weights = [w * scale for w in weights] + [new_L // new_elem]
    new_w = new_L // new_elem
    # Scale existing sums
    new_sums = {s * scale for s in sums}
    added = {s + new_w for s in new_sums}
    if new_sums & added:
        return None
    new_sums |= added
    return (new_L, new_weights, new_sums)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, keep buffer for safety/overhead

    universe = list(range(1, 23))

    def time_left():
        return deadline - time.time()

    def greedy(order, max_size=22):
        """Greedy construction using incremental checks."""
        L = 1
        weights = []
        sums = {0}
        result = []
        for e in order:
            if len(result) >= max_size:
                break
            res = _can_add_incremental(L, weights, sums, e)
            if res is not None:
                L, weights, sums = res
                result.append(e)
        return result

    def local_search_add(s, L, weights, sums):
        """Try to add any missing element."""
        improved = True
        while improved and time_left() > 5:
            improved = False
            s_set = set(s)
            remaining = [e for e in universe if e not in s_set]
            random.shuffle(remaining)
            for e in remaining:
                res = _can_add_incremental(L, weights, sums, e)
                if res is not None:
                    L, weights, sums = res
                    s = s + [e]
                    improved = True
                    break
        return s, L, weights, sums

    def local_search_swap(s, L, weights, sums):
        """Try swaps to escape local maxima."""
        improved = True
        while improved and time_left() > 5:
            improved = False
            s_set = set(s)
            in_list = list(s)
            out_list = [e for e in universe if e not in s_set]
            random.shuffle(out_list)
            for u in in_list:
                if time_left() < 5:
                    break
                for v in out_list:
                    if time_left() < 5:
                        break
                    # Rebuild without u, then try adding v
                    new_elems = [x for x in s if x != u] + [v]
                    res = _build_sums(new_elems)
                    if res is not None:
                        L, weights, sums = res
                        s = new_elems
                        # Try to extend
                        s, L, weights, sums = local_search_add(s, L, weights, sums)
                        improved = True
                        break
                if improved:
                    break
        return s, L, weights, sums

    best_set = []
    best_L = 1
    best_weights = []
    best_sums = {0}

    # 1. Deterministic greedy orders
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        list(universe),
        # Try orders that favor larger elements first (they create more separation)
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    ]

    for order in orders:
        if time_left() < 10:
            break
        result = greedy(order)
        if len(result) > len(best_set):
            best_set = result
            res = _build_sums(result)
            if res:
                best_L, best_weights, best_sums = res

    # 2. Local search add on best
    best_set, best_L, best_weights, best_sums = local_search_add(
        best_set, best_L, best_weights, best_sums)

    # 3. Random restarts with local search
    num_restarts = 0
    while time_left() > 15:
        random.shuffle(universe)
        result = greedy(universe)
        res = _build_sums(result)
        if res:
            L, weights, sums = res
        else:
            L, weights, sums = 1, [], {0}
            result = []
        result, L, weights, sums = local_search_add(result, L, weights, sums)
        
        if len(result) > len(best_set):
            best_set = result
            best_L, best_weights, best_sums = L, weights, sums
            if len(best_set) >= 20:
                break
        num_restarts += 1

    # 4. Swap-based local search
    best_set, best_L, best_weights, best_sums = local_search_swap(
        best_set, best_L, best_weights, best_sums)

    # 5. Final add pass
    best_set, best_L, best_weights, best_sums = local_search_add(
        best_set, best_L, best_weights, best_sums)

    # Final verification
    if best_set and _is_dissociated(best_set):
        return sorted(best_set)
    
    # Fallback
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
