"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm
from functools import reduce

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def _greedy_select(order, deadline=None):
    """
    Greedily select elements from the given order.
    Returns the list of selected elements.
    """
    current_sums = {0}
    selected = []
    for a in order:
        if deadline is not None and time.time() > deadline:
            break
        v = _L // a
        # Check for collision: if any s+v is in current_sums, collision
        # Equivalent: (current_sums + v) intersects current_sums
        # Fast check: compute shifted set and intersect
        shifted = {s + v for s in current_sums}
        if shifted & current_sums:
            continue
        # No collision, accept
        selected.append(a)
        current_sums |= shifted
        
    return selected

def _local_search(selected, deadline):
    """
    Try to improve the set by adding remaining elements or swapping.
    Returns improved list.
    """
    selected_set = set(selected)
    remaining = [a for a in range(1, 23) if a not in selected_set]
    
    # Rebuild sums
    current_sums = {0}
    for a in selected:
        v = _L // a
        current_sums |= {s + v for s in current_sums}
    
    # Try adding each remaining element
    for a in remaining:
        if time.time() > deadline:
            break
        v = _L // a
        shifted = {s + v for s in current_sums}
        if not (shifted & current_sums):
            selected.append(a)
            current_sums |= shifted
    
    # Try swapping: remove one selected, try to add two from remaining
    improved = list(selected)
    for i, rem_a in enumerate(remaining):
        if time.time() > deadline:
            break
        v_rem = _L // rem_a
        for j in range(len(selected)):
            if time.time() > deadline:
                break
            # Try removing selected[j] and adding rem_a
            # This is expensive; do a quick heuristic check
            pass  # Skip full swap for now; focus on additions
    
    return improved

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Strategy 1: Largest denominators first (22 down to 1)
    order1 = list(range(22, 0, -1))
    res1 = _greedy_select(order1, deadline)
    
    if time.time() > deadline:
        return res1 if res1 else [1]
        
    # Strategy 2: Smallest denominators first (1 to 23)
    order2 = list(range(1, 23))
    res2 = _greedy_select(order2, deadline)
    
    if time.time() > deadline:
        best = res1 if len(res1) >= len(res2) else res2
        return best if best else [1]
        
    # Strategy 3: Randomized restarts
    best_res = res1 if len(res1) >= len(res2) else res2
    best_len = len(best_res)
    
    # Try random orders
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
        order3 = list(range(1, 23))
        random.shuffle(order3)
        res3 = _greedy_select(order3, deadline)
        if len(res3) > best_len:
            best_res = res3
            best_len = len(res3)
    
    # Local search: try to add more elements to best result
    if time.time() < deadline - 2.0:
        best_res = _local_search(best_res, deadline)
        best_len = len(best_res)
    
    # Final: try a few more random restarts with the improved baseline
    for _ in range(20):
        if time.time() > deadline - 1.0:
            break
        order4 = list(range(1, 23))
        random.shuffle(order4)
        res4 = _greedy_select(order4, deadline)
        if len(res4) > best_len:
            best_res = res4
            best_len = len(res4)
    
    return best_res if best_res else [1]

# EVOLVE-BLOCK-END
