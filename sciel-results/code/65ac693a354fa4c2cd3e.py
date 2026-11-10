"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd, isqrt

def construct():
    deadline = time.time() + 80  # 80s budget, 10s safety margin
    N = 512
    numbers = np.arange(1, N + 1)
    
    # Precompute compatibility matrix
    # compat[i,j] = True if lcm(numbers[i], numbers[j]) <= 512
    n = N
    # Use vectorized approach
    a = numbers[:, np.newaxis].astype(np.int64)
    b = numbers[np.newaxis, :].astype(np.int64)
    # lcm(a,b) = a*b/gcd(a,b)
    # We need a*b/gcd(a,b) <= 512
    # Equivalent: a*b <= 512 * gcd(a,b)
    # Or: (a/gcd(a,b)) * (b/gcd(a,b)) <= 512
    # Let's compute gcd matrix
    # For efficiency, compute row by row
    compat = np.ones((n, n), dtype=bool)
    for i in range(n):
        ai = numbers[i]
        # lcm(ai, bj) = ai * bj / gcd(ai, bj)
        # Need ai * bj / gcd(ai, bj) <= 512
        # bj <= 512 * gcd(ai, bj) / ai
        # This is tricky to vectorize easily. Let's just compute directly.
        g = np.array([gcd(ai, int(bj)) for bj in numbers], dtype=np.int64)
        lcms = ai * numbers.astype(np.int64) // g
        compat[i, :] = lcms <= N
    # Symmetrize (should already be symmetric)
    # compat = (compat | compat.T)
    
    # Active set: start with all
    active = np.ones(n, dtype=bool)
    
    # Iterative removal: remove the element with most conflicts
    while True:
        if time.time() > deadline - 5:
            break
        # Find conflicts for each active element
        # For each active i, count how many active j have compat[i,j] == False
        active_indices = np.where(active)[0]
        if len(active_indices) <= 1:
            break
        sub_compat = compat[np.ix_(active_indices, active_indices)]
        # Count conflicts (False entries) for each row
        conflict_counts = (~sub_compat).sum(axis=1)
        # Remove the one with most conflicts
        max_conflict_idx = np.argmax(conflict_counts)
        if conflict_counts[max_conflict_idx] == 0:
            break  # No conflicts, we're done
        # Remove it
        remove_global = active_indices[max_conflict_idx]
        active[remove_global] = False
    
    # Now try to add back removed elements if they're compatible with all active
    removed = np.where(~active)[0]
    added_any = True
    while added_any:
        if time.time() > deadline - 5:
            break
        added_any = False
        for r in removed:
            if active[r]:
                continue
            # Check if r is compatible with all active elements
            active_idx = np.where(active)[0]
            if len(active_idx) == 0:
                active[r] = True
                added_any = True
                continue
            if np.all(compat[r, active_idx]):
                active[r] = True
                added_any = True
                if time.time() > deadline - 5:
                    break
    
    # Local search: try swapping - remove one active, add one removed
    # Multiple passes
    for iteration in range(50):
        if time.time() > deadline - 3:
            break
        improved = False
        active_idx = np.where(active)[0]
        removed_idx = np.where(~active)[0]
        for r in removed_idx:
            if time.time() > deadline - 3:
                break
            # Check if r can be added
            if not np.all(compat[r, active_idx]):
                continue
            # Find which active elements block r
            # Actually r can be added directly, so just add it
            active[r] = True
            improved = True
            active_idx = np.where(active)[0]
        
        # Try swaps: for each removed r, find if removing some active a and adding r helps
        # This is more complex. Let's try: for each pair (a in active, r in removed),
        # check if removing a and adding r keeps all pairs compatible and doesn't reduce size
        # Actually, since we want to maximize size, a swap only helps if we can add more than we remove.
        # A simple swap (remove 1, add 1) doesn't help size. We need to add without removing,
        # or remove 1 and add 2, etc.
        
        # Better local search: try to add any removed element that's compatible
        for r in removed_idx:
            if time.time() > deadline - 3:
                break
            if active[r]:
                continue
            active_idx = np.where(active)[0]
            if len(active_idx) == 0:
                active[r] = True
                improved = True
                continue
            if np.all(compat[r, active_idx]):
                active[r] = True
                improved = True
    
    # Final local search: try removing one active element to allow adding multiple removed elements
    for iteration in range(20):
        if time.time() > deadline - 2:
            break
            # Try: for each active element a, temporarily remove it, then greedily add removed elements
        best_size = int(active.sum())
        best_state = active.copy()
        
        active_idx = np.where(active)[0]
        removed_idx = np.where(~active)[0]
        
        for a_idx in active_idx:
            if time.time() > deadline - 2:
                break
            # Temporarily remove a_idx
            active[a_idx] = False
            # Try to add as many removed elements as possible
            added = []
            for r in removed_idx:
                if active[r]:
                    continue
                current_active = np.where(active)[0]
                if len(current_active) == 0 or np.all(compat[r, current_active]):
                    active[r] = True
                    added.append(r)
            # Check size
            current_size = int(active.sum())
            if current_size > best_size:
                best_size = current_size
                best_state = active.copy()
            # Restore
            active[a_idx] = True
            for r in added:
                active[r] = False
        
        if not np.array_equal(active, best_state):
            active = best_state
        else:
            break
    
    result = [int(x) for x in numbers[active]]
    return result
# EVOLVE-BLOCK-END
