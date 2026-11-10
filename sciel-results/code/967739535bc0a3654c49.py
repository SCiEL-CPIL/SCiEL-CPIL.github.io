"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from functools import reduce

def _lcm_all(lst):
    return reduce(math.lcm, lst, 1)

# Precompute LCM of 1..22 and scaled weights
_UNIVERSE = list(range(1, 23))
_LCM_22 = _lcm_all(_UNIVERSE)
# weights[i-1] = LCM // i for i in 1..22
_WEIGHTS = {i: _LCM_22 // i for i in _UNIVERSE}

def _is_dissociated(elems):
    """Check if all subset sums of reciprocals are distinct using precomputed scaled integers."""
    if not elems:
        return True
    if len(elems) == 1:
        return True
    
    n = len(elems)
    # Use precomputed weights
    weights = [_WEIGHTS[e] for e in elems]
    
    # Generate all subset sums incrementally and check for collisions
    sums = {0}
    for w in weights:
        new_sums = set()
        for s in sums:
            new_sums.add(s + w)
        if len(new_sums) < len(sums):
            return False
        sums = sums | new_sums
    
    return len(sums) == (1 << n)

def _conflict_degree(e, current_set):
    """Quick check: does adding e to current_set cause a collision?
    This is a necessary condition check (not sufficient), used for pruning."""
    # If e is already in current_set, conflict
    if e in current_set:
        return True
    # Quick pairwise check: for any a, b in current_set, 
    # check if 1/a + 1/b = 1/e or 1/a = 1/b + 1/e etc.
    # This is a subset of the full collision check but catches many cases fast
    w_e = _WEIGHTS[e]
    for a in current_set:
        w_a = _WEIGHTS[a]
        # Check if w_e equals w_a (impossible since all distinct)
        # Check if w_e + w_a equals any w_b in current_set
        for b in current_set:
            if b == a:
                continue
            w_b = _WEIGHTS[b]
            if w_e + w_a == w_b:
                return True
            if w_e + w_b == w_a:
                return True
            if w_a + w_b == w_e:
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 70  # 90s limit, keep 20s buffer for safety/overhead
    
    universe = list(range(1, 23))
    
    # Greedy function with a given order
    def greedy(order):
        cur = []
        cur_set = set()
        for e in order:
            if e in cur_set:
                continue
            test_set = cur + [e]
            if _is_dissociated(test_set):
                cur = test_set
                cur_set.add(e)
        return cur

    best_set = []
    best_set_set = set()
    
    # 1. Try deterministic orders sorted by conflict degree (ascending)
    # Elements with fewer conflicts are "safer" to include early
    def conflict_count(e):
        """Count how many pairs (a,b) in universe have 1/a + 1/b = 1/e or similar"""
        count = 0
        w_e = _WEIGHTS[e]
        for a in universe:
            if a == e:
                continue
            w_a = _WEIGHTS[a]
            for b in universe:
                if b == a or b == e:
                    continue
                w_b = _WEIGHTS[b]
                if w_e + w_a == w_b or w_e + w_b == w_a or w_a + w_b == w_e:
                    count += 1
        return count
    
    conflict_counts = {e: conflict_count(e) for e in universe}
    
    orders = [
        sorted(universe, key=lambda x: conflict_counts[x]),  # Low conflict first
        sorted(universe, reverse=True),
        sorted(universe),
        list(universe)
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        result = greedy(order)
        if len(result) > len(best_set):
            best_set = result
            best_set_set = set(result)
    
    # 2. Local search: try to add any missing element
    improved = True
    while improved and time.time() < deadline:
        improved = False
        remaining = [e for e in universe if e not in best_set_set]
        random.shuffle(remaining)
        for e in remaining:
            test_set = best_set + [e]
            if _is_dissociated(test_set):
                best_set = test_set
                best_set_set.add(e)
                improved = True
                break  # Restart the loop to check new remaining order
    
    # 3. Local search with swaps: remove one, add one or two
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            remaining = [e for e in universe if e not in best_set_set]
            random.shuffle(remaining)
            for e_out in list(best_set):
                if time.time() > deadline:
                    break
                # Try swapping e_out with each remaining element
                for e_in in remaining:
                    test_set = [x for x in best_set if x != e_out] + [e_in]
                    if _is_dissociated(test_set):
                        # Same size, but might allow future additions
                        # Only accept if it enables adding another element
                        best_set_set2 = set(test_set)
                        for e_add in [x for x in universe if x not in best_set_set2]:
                            if _is_dissociated(test_set + [e_add]):
                                best_set = test_set + [e_add]
                                best_set_set = set(best_set)
                                improved = True
                                break
                        if improved:
                            break
                    if time.time() > deadline:
                        break
                if improved:
                    break
    
    # 4. Random restarts
    num_restarts = 0
    while time.time() < deadline - 5:
        random.shuffle(universe)
        result = greedy(universe)
        if len(result) > len(best_set):
            best_set = result
            best_set_set = set(best_set)
            # Try to improve this new best
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                remaining = [e for e in range(1, 23) if e not in best_set_set]
                random.shuffle(remaining)
                for e in remaining:
                    if _is_dissociated(best_set + [e]):
                        best_set.append(e)
                        best_set_set.add(e)
                        improved = True
                        break
        num_restarts += 1
        
        # If we found a very large set, we might stop early?
        if len(best_set) >= 18:
            break

    # Final verification
    if best_set and _is_dissociated(best_set):
        return sorted(best_set)
    
    # Fallback to a known valid set if something went wrong
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
