"""
Strategy: Greedy construction of a large family of 4-subsets of [1,16] with no
three-petal sunflowers (no three distinct sets A,B,C with A∩B = A∩C = B∩C).

Uses bitmask representation for efficiency. Multiple greedy passes with different
orderings, followed by local improvement (swap) passes to refine the solution.
"""

import time
import random
from itertools import combinations

def _check_conflict(family, x):
    """Check if adding x to family creates a sunflower triple.
    A sunflower triple A,B,C has A∩B = A∩C = B∩C.
    For new set x, check all pairs (A,B) in family:
    if (A&B) == (A&x) == (B&x), then conflict.
    """
    m = len(family)
    for i in range(m):
        a = family[i]
        ax = a & x
        for j in range(i + 1, m):
            b = family[j]
            if (a & b) == ax and (b & x) == ax:
                return True
    return False

def _greedy_pass(all_sets, order, deadline):
    """Run one greedy pass with given order."""
    family = []
    for idx in order:
        if time.time() > deadline:
            break
        x = all_sets[idx]
        if not _check_conflict(family, x):
            family.append(x)
    return family

def _local_improve(family, all_sets, n_total, deadline, max_iter=5000):
    """Try to improve family by swapping out a set for a better one.
    Returns improved family."""
    family_set = set(family)
    improved = list(family)
    
    # Try removing each set and adding a different one
    for i in range(len(improved)):
        if time.time() > deadline:
            break
        removed = improved[i]
        improved[i] = None  # mark as removed
        
        # Try to find a replacement
        for j in range(n_total):
            if time.time() > deadline:
                break
            candidate = all_sets[j]
            if candidate in family_set and candidate != removed:
                continue
            # Check if candidate can replace removed
            temp_family = improved[:i] + improved[i+1:]
            if not _check_conflict(temp_family, candidate):
                improved[i] = candidate
                family_set.discard(removed)
                family_set.add(candidate)
                break
        else:
            # No replacement found, put back
            improved[i] = removed
    
    return improved

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Generate all 4-subsets of [1,16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_total = len(all_sets)  # 1820
    best_family = []
    best_size = 0
    
    # Precompute a set of all masks for quick lookup
    all_masks_set = set(all_sets)
    
    # Multiple greedy passes with different orderings
    rng = random.Random(42)
    
    num_trials = 10
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = list(range(n_total))
        elif trial == 1:
            order = list(range(n_total - 1, -1, -1))
        elif trial == 2:
            # Sort by popcount of complement (not very useful for fixed-size sets)
            # Instead, sort by the sum of elements
            order = sorted(range(n_total), key=lambda i: sum((i >> b) & 1 for b in range(16)))
        elif trial == 3:
            # Sort by minimum element
            order = sorted(range(n_total), key=lambda i: i & -i)
        else:
            # Random shuffle
            order = list(range(n_total))
            rng.shuffle(order)
        
        family = _greedy_pass(all_sets, order, deadline)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
        
        # Local improvement on the best family found so far
        if time.time() < deadline - 10.0:
            improved = _local_improve(family, all_sets, n_total, deadline)
            if len(improved) > best_size:
                best_size = len(improved)
                best_family = improved
    
    # Final local improvement on best family
    if time.time() < deadline - 5.0 and best_family:
        final = _local_improve(best_family, all_sets, n_total, deadline)
        if len(final) > best_size:
            best_family = final
    
    # Convert bitmasks back to sorted lists
    result = []
    for mask in best_family:
        elements = []
        for i in range(16):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
    
    return result
