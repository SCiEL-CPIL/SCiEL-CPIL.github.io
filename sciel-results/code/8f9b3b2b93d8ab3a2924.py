"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def _conflicts(a, b):
    """Return True if (a*b) % (a+b) == 0 for distinct a,b."""
    s = a + b
    return (a * b) % s == 0

def _build_conflict_bitset(n=256):
    """For each i in 1..n, build a bitmask of j != i such that conflict(i,j)."""
    # Use Python ints as bitsets; bit k set means conflict with (k+1)
    masks = [0] * (n + 1)
    for a in range(1, n + 1):
        m = 0
        for b in range(1, n + 1):
            if a == b:
                continue
            if _conflicts(a, b):
                m |= (1 << b)
        masks[a] = m
    return masks

def _greedy_independent_set(masks, n=256, order=None):
    """Greedy: add elements in given order if no conflict with already-selected."""
    selected = set()
    sel_mask = 0
    for v in order:
        if v in selected:
            continue
        if (masks[v] & sel_mask) == 0:
            selected.add(v)
            sel_mask |= (1 << v)
    return selected

def _local_search(selected, masks, n=256, max_iters=200000, deadline=None):
    """Try to improve by adding any element, or swapping to increase size."""
    sel_mask = 0
    for v in selected:
        sel_mask |= (1 << v)
    
    all_vals = list(range(1, n + 1))
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        # Try adding any non-selected vertex
        improved = False
        for v in all_vals:
            if v in selected:
                continue
            if (masks[v] & sel_mask) == 0:
                selected.add(v)
                sel_mask |= (1 << v)
                improved = True
                break
        if improved:
            continue
        # Try swap: remove u, add v if net gain
        # Only useful if we can add 2 for every 1 removed, etc.
        # Simple: try removing one u and adding any v that becomes free
        found_swap = False
        for u in list(selected):
            # If we remove u, which new vertices become addable?
            new_mask = sel_mask & ~(1 << u)
            # Vertices that conflict only with u among selected
            # A vertex v not in selected can be added if masks[v] & new_mask == 0
            # and it wasn't addable before (i.e., masks[v] & sel_mask != 0, meaning it conflicted with u)
            # For a swap to increase size, we need at least 2 such v's
            candidates = []
            for v in all_vals:
                if v in selected:
                    continue
                if (masks[v] & new_mask) == 0 and (masks[v] & sel_mask) != 0:
                    candidates.append(v)
            if len(candidates) >= 2:
                # Remove u, add two candidates
                selected.discard(u)
                for v in candidates[:2]:
                    selected.add(v)
                    sel_mask |= (1 << v)
                sel_mask &= ~(1 << u)
                found_swap = True
                break
        if not found_swap:
            break
    return selected

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    n = 256
    masks = _build_conflict_bitset(n)
    
    # Initial: all odds (known valid, 128 elements)
    odds = list(range(1, n + 1, 2))
    selected = set(odds)
    
    # Greedy add: try to add evens in order of fewest conflicts with selected
    sel_mask = 0
    for v in selected:
        sel_mask |= (1 << v)
    
    evens = list(range(2, n + 1, 2))
    # Sort evens by number of conflicts with current selected
    def conflict_count(v):
        m = masks[v] & sel_mask
        return bin(m).count('1')
    evens.sort(key=conflict_count)
    
    for v in evens:
        if (masks[v] & sel_mask) == 0:
            selected.add(v)
            sel_mask |= (1 << v)
    
    # Local search to improve
    selected = _local_search(selected, masks, n, deadline=deadline)
    
    # Final validation
    result = sorted(selected)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                # Should not happen; remove b
                result.pop(j)
                j -= 1
                break
    
    if not result:
        return [1]
    return result
# EVOLVE-BLOCK-END
