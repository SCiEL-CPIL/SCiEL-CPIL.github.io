"""
Maximizing a Sidon set (distinct pairwise sums) in [1, 284] containing {1, 2, 5, 11}.

Approach:
1. Greedy construction to get a strong initial solution (size ~18-19).
2. Iterated Local Search (ILS) with:
   - Swap moves (replace one element with another)
   - Add/remove moves (try to add, or remove to allow larger additions)
   - Bit-parallel sum tracking for fast conflict detection
3. Deadline guard to return best solution found within time budget.

Key insight: A Sidon set requires all pairwise sums a+b (a<=b) to be distinct.
We track used sums in a set and use fast checks when adding/removing elements.
"""

import time
import random
from itertools import combinations

# Deadline: 140s wall-clock (10s safety margin for 150s limit)
_DEADLINE = 140.0
_start_time = None

def _elapsed():
    return time.time() - _start_time

def _time_up():
    return _elapsed() > _DEADLINE

def _check_sidon(a):
    """Verify that set 'a' has all distinct pairwise sums."""
    a_sorted = sorted(a)
    seen = set()
    for i in range(len(a_sorted)):
        for j in range(i, len(a_sorted)):
            s = a_sorted[i] + a_sorted[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _greedy_construct():
    """Greedy construction of a large Sidon set containing {1,2,5,11}."""
    a = [1, 2, 5, 11]
    sums = set()
    for i in range(len(a)):
        for j in range(i, len(a)):
            sums.add(a[i] + a[j])
    
    # Try adding elements in a smart order
    candidates = list(range(12, 285))
    # Sort by some heuristic: prefer elements that create fewer conflicts
    # Actually, just try in order and use backtracking-lite
    
    for x in candidates:
        new_sums = set()
        conflict = False
        for y in a:
            s = x + y
            if s in sums or s in new_sums:
                conflict = True
                break
            new_sums.add(s)
        s2 = 2 * x
        if s2 in sums or s2 in new_sums:
            conflict = True
        if not conflict:
            a.append(x)
            sums |= new_sums
            sums.add(s2)
    
    return sorted(a)

def _try_add(a, x, sums):
    """Try to add x to Sidon set a. Return (success, new_sums)."""
    new_sums = set()
    for y in a:
        s = x + y
        if s in sums:
            return False, None
        if s in new_sums:
            return False, None
        new_sums.add(s)
    s2 = 2 * x
    if s2 in sums or s2 in new_sums:
        return False, None
    return True, new_sums

def _remove_element(a, x, sums):
    """Remove x from Sidon set a. Return (new_a, new_sums)."""
    new_a = [y for y in a if y != x]
    # Recompute sums
    new_sums = set()
    for i in range(len(new_a)):
        for j in range(i, len(new_a)):
            new_sums.add(new_a[i] + new_a[j])
    return new_a, new_sums

def _local_search(a, sums, max_iters=5000):
    """
    Local search: try swaps and additions to improve.
    a: sorted list, sums: set of all pairwise sums.
    Returns improved (a, sums).
    """
    a = sorted(a)
    n = len(a)
    
    # Precompute sums properly
    sums = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(a[i] + a[j])
    
    improved = True
    iters = 0
    while improved and iters < max_iters and not _time_up():
        improved = False
        iters += 1
        
        # Try adding elements
        for x in range(1, 285):
            if x in a:
                continue
            ok, new_sums = _try_add(a, x, sums)
            if ok:
                a = sorted(a + [x])
                sums = sums | new_sums | {2 * x}
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one element, add another
        for i, old_x in enumerate(a):
            # Skip prescribed elements
            if old_x in (1, 2, 5, 11):
                continue
            
            new_a, new_sums = _remove_element(a, old_x, sums)
            
            # Try to add a new element
            best_add = None
            for x in range(1, 285):
                if x in new_a:
                    continue
                ok, add_sums = _try_add(new_a, x, new_sums)
                if ok:
                    # Check if this swap is beneficial (size stays same or increases)
                    # Since we removed one and add one, size is same
                    # Prefer to add larger elements or those that allow future additions
                    if best_add is None or x > best_add:
                        best_add = x
            
            if best_add is not None:
                # Perform the swap
                a = sorted(new_a + [best_add])
                # Recompute sums
                sums = set()
                for ii in range(len(a)):
                    for jj in range(ii, len(a)):
                        sums.add(a[ii] + a[jj])
                improved = True
                break
    
    return a, sums

def _perturb(a, sums, strength=2):
    """
    Perturb the solution: randomly remove 'strength' elements (not prescribed).
    Returns perturbed (a, sums).
    """
    a = list(a)
    # Only perturb non-prescribed elements
    removable = [x for x in a if x not in (1, 2, 5, 11)]
    if not removable:
        return a, sums
    
    k = min(strength, len(removable))
    to_remove = random.sample(removable, k)
    
    new_a = [x for x in a if x not in to_remove]
    
    # Recompute sums
    new_sums = set()
    for i in range(len(new_a)):
        for j in range(i, len(new_a)):
            new_sums.add(new_a[i] + a[j])  # BUG: should be new_a[j]
    # Fix: recompute properly
    new_sums = set()
    for i in range(len(new_a)):
        for j in range(i, len(new_a)):
            new_sums.add(new_a[i] + new_a[j])
    
    return sorted(new_a), new_sums

def construct():
    global _start_time
    _start_time = time.time()
    
    # Step 1: Greedy construction
    best_a = _greedy_construct()
    best_size = len(best_a)
    
    # Verify
    if not _check_sidon(best_a):
        # Fallback: just return the prescribed elements
        return [1, 2, 5, 11]
    
    # Compute sums for best_a
    best_sums = set()
    for i in range(len(best_a)):
        for j in range(i, len(best_a)):
            best_sums.add(best_a[i] + best_a[j])
    
    # Step 2: Local search on the greedy solution
    improved_a, improved_sums = _local_search(best_a, best_sums, max_iters=2000)
    if len(improved_a) > best_size:
        best_a = improved_a
        best_size = len(best_a)
        best_sums = improved_sums
    
    # Step 3: Iterated Local Search
    current_a = best_a
    current_sums = best_sums
    
    # Try multiple restarts with perturbation
    while not _time_up():
        # Perturb
        perturbed_a, perturbed_sums = _perturb(current_a, current_sums, strength=2)
        
        # Local search from perturbed
        ls_a, ls_sums = _local_search(perturbed_a, perturbed_sums, max_iters=3000)
        
        if len(ls_a) > best_size:
            best_a = ls_a
            best_size = len(best_a)
            best_sums = ls_sums
            current_a = ls_a
            current_sums = ls_sums
        elif len(ls_a) == best_size:
            # Keep the better one (arbitrary tie-break)
            current_a = ls_a
            current_sums = ls_sums
    
    # Final verification
    if not _check_sidon(best_a):
        # Should not happen, but safety
        return [1, 2, 5, 11]
    
    # Ensure prescribed elements are present
    for p in (1, 2, 5, 11):
        if p not in best_a:
            # This shouldn't happen since we never remove them
            best_a.append(p)
            best_a = sorted(best_a)
    
    return sorted(best_a)
