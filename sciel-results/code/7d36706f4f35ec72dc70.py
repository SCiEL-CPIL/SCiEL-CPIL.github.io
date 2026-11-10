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
    new_w = new_L // new_elem
    if scale == 1:
        # No scaling needed - just check if any existing sum + new_w is already in sums
        # Optimization: check collision first
        collision = False
        for s in sums:
            if (s + new_w) in sums:
                collision = True
                break
        if collision:
            return None
        new_weights = weights + [new_w]
        new_sums = sums | {s + new_w for s in sums}
        return (new_L, new_weights, new_sums)
    else:
        # Need to scale
        scaled_sums = {s * scale for s in sums}
        # Check collision
        collision = False
        for s in scaled_sums:
            if (s + new_w) in scaled_sums:
                collision = True
                break
        if collision:
            return None
        new_weights = [w * scale for w in weights] + [new_w]
        new_sums = scaled_sums | {s + new_w for s in scaled_sums}
        return (new_L, new_weights, new_sums)

def construct():
    start_time = time.time()
    deadline = start_time + 55

    universe = list(range(1, 23))

    def time_left():
        return deadline - time.time()

    def greedy(order):
        L = 1
        weights = []
        sums = {0}
        result = []
        for e in order:
            res = _can_add_incremental(L, weights, sums, e)
            if res is not None:
                L, weights, sums = res
                result.append(e)
        return result, L, weights, sums

    def local_search_add(s, L, weights, sums):
        improved = True
        while improved and time_left() > 3:
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
        improved = True
        while improved and time_left() > 3:
            improved = False
            s_set = set(s)
            in_list = list(s)
            out_list = [e for e in universe if e not in s_set]
            random.shuffle(out_list)
            for u in in_list:
                if time_left() < 3:
                    break
                for v in out_list:
                    if time_left() < 3:
                        break
                    new_elems = [x for x in s if x != u] + [v]
                    res = _build_sums(new_elems)
                    if res is not None:
                        L, weights, sums = res
                        s = new_elems
                        s, L, weights, sums = local_search_add(s, L, weights, sums)
                        improved = True
                        break
                if improved:
                    break
        return s, L, weights, sums

    def remove_and_add(s, L, weights, sums, remove_elem, add_elem):
        """Remove one element and add another, checking dissociation."""
        new_elems = [x for x in s if x != remove_elem] + [add_elem]
        return _build_sums(new_elems)

    best_set = []
    best_L = 1
    best_weights = []
    best_sums = {0}

    # 1. Multiple deterministic greedy orders
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        [22, 19, 17, 15, 14, 13, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 16, 18, 20, 21],
        [22, 21, 19, 17, 15, 14, 13, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 16, 18, 20],
        [22, 19, 17, 15, 14, 13, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 21, 20, 18, 16, 12],
    ]

    for order in orders:
        if time_left() < 5:
            break
        result, L, weights, sums = greedy(order)
        if len(result) > len(best_set):
            best_set = result
            best_L, best_weights, best_sums = L, weights, sums

    # 2. Local search add
    best_set, best_L, best_weights, best_sums = local_search_add(
        best_set, best_L, best_weights, best_sums)

    # 3. Many random restarts with varied strategies
    while time_left() > 8:
        strategy = random.random()
        
        if strategy < 0.3:
            # Pure random order
            order = list(universe)
            random.shuffle(order)
        elif strategy < 0.6:
            # Biased toward larger elements
            order = list(universe)
            random.shuffle(order)
            order.sort(key=lambda x: -x + random.random() * 8)
        elif strategy < 0.8:
            # Start with large, then random
            large = sorted([e for e in universe if e >= 10], reverse=True)
            small = sorted([e for e in universe if e < 10])
            random.shuffle(small)
            order = large + small
        else:
            # Random with some structure
            order = list(universe)
            random.shuffle(order)
            # Put a few large elements at the front
            large_elems = random.sample([e for e in universe if e >= 15], min(3, len([e for e in universe if e >= 15])))
            for le in large_elems:
                if le in order:
                    order.remove(le)
                    order.insert(random.randint(0, 3), le)
        
        result, L, weights, sums = greedy(order)
        result, L, weights, sums = local_search_add(result, L, weights, sums)
        
        if len(result) > len(best_set):
            best_set = result
            best_L, best_weights, best_sums = L, weights, sums
            if len(best_set) >= 20:
                break

    # 4. Swap-based local search
    if time_left() > 5:
        best_set, best_L, best_weights, best_sums = local_search_swap(
            best_set, best_L, best_weights, best_sums)

    # 5. 2-for-1 swap: remove one, try adding two different ones
    improved = True
    while improved and time_left() > 4:
        improved = False
        s_set = set(best_set)
        in_list = list(best_set)
        out_list = [e for e in universe if e not in s_set]
        random.shuffle(out_list)
        for u in in_list:
            if time_left() < 4:
                break
            elems_without = [x for x in best_set if x != u]
            res = _build_sums(elems_without)
            if res is None:
                continue
            L_tmp, w_tmp, s_tmp = res
            added_count = 0
            for v in out_list:
                if time_left() < 4:
                    break
                res2 = _can_add_incremental(L_tmp, w_tmp, s_tmp, v)
                if res2 is not None:
                    L_tmp, w_tmp, s_tmp = res2
                    elems_without.append(v)
                    added_count += 1
                    if added_count >= 2:
                        break
            if added_count >= 2:
                best_set = elems_without
                best_L, best_weights, best_sums = L_tmp, w_tmp, s_tmp
                improved = True
                best_set, best_L, best_weights, best_sums = local_search_add(
                    best_set, best_L, best_weights, best_sums)
                break

    # 6. Final add pass
    if time_left() > 2:
        best_set, best_L, best_weights, best_sums = local_search_add(
            best_set, best_L, best_weights, best_sums)

    if best_set and _is_dissociated(best_set):
        return sorted(best_set)
    
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
