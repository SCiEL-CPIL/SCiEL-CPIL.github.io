"""
Strategy: Greedy construction of a large family of 4-subsets of [1,16] with no
three-petal sunflowers (no three distinct sets A,B,C with A∩B = A∩C = B∩C).

Uses bitmask representation for efficiency. Multiple greedy passes with different
orderings, followed by local improvement (swap and 2-swap) passes to refine the solution.
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

def _local_improve_1swap(family, all_sets, n_total, deadline, max_iter=2000):
    """Try to improve family by swapping out a set for a better one."""
    family_set = set(family)
    improved = list(family)
    changed = False
    
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
                changed = True
                break
        else:
            # No replacement found, put back
            improved[i] = removed
    
    return improved, changed

def _local_improve_2swap(family, all_sets, n_total, deadline, max_iter=500):
    """Try to improve family by swapping out two sets for two new ones."""
    family_set = set(family)
    improved = list(family)
    changed = False
    n = len(improved)
    
    if n < 2:
        return improved, changed

    # Limit the scope of 2-swaps to avoid O(N^4) explosion
    # We pick two indices i, j to remove, and try to find two candidates c1, c2 to add.
    # To be efficient, we only try a subset of candidate pairs or iterate smartly.
    
    # Strategy: For each pair (i, j) in current family, remove them.
    # Then try to add any two valid sets from the remaining pool.
    # This is expensive, so we limit the number of (i,j) pairs we test or use random sampling.
    
    # Better approach: Randomly sample pairs of indices to remove.
    rng = random.Random(int(time.time() * 1000))
    
    attempts = 0
    max_attempts = 1000
    
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        
        removed1 = improved[i]
        removed2 = improved[j]
        
        # Create family without i and j
        temp_family = [x for k, x in enumerate(improved) if k != i and k != j]
        
        # Find two candidates to add
        # We need to find c1, c2 such that:
        # 1. c1, c2 not in temp_family
        # 2. c1 != c2
        # 3. c1 not conflict with temp_family
        # 4. c2 not conflict with temp_family + {c1}
        # 5. c1, c2, and any a in temp_family don't form a sunflower with each other? 
        #    Actually, _check_conflict handles this if we add sequentially.
        
        # To speed up, we can pre-filter candidates that don't conflict with temp_family.
        # But that's O(N * M). Let's just try a few random candidates.
        
        # Pick random candidates from all_sets
        c1_idx = rng.randint(0, n_total - 1)
        c2_idx = rng.randint(0, n_total - 1)
        
        if c1_idx == c2_idx:
            continue
            
        c1 = all_sets[c1_idx]
        c2 = all_sets[c2_idx]
        
        # Ensure candidates are not already in the family (except the ones we removed)
        if c1 in temp_family or c2 in temp_family:
            continue
        if c1 == c2:
            continue
            
        # Check if c1 can be added to temp_family
        if _check_conflict(temp_family, c1):
            continue
            
        # Check if c2 can be added to temp_family + [c1]
        temp_family_with_c1 = temp_family + [c1]
        if _check_conflict(temp_family_with_c1, c2):
            continue
            
        # Success! Replace i and j with c1 and c2
        # Update improved list
        # Remove i and j, add c1 and c2
        # Since order doesn't matter for the set, we can just swap them in place
        improved[i] = c1
        improved[j] = c2
        changed = True
        
        # Update family_set
        family_set.discard(removed1)
        family_set.discard(removed2)
        family_set.add(c1)
        family_set.add(c2)
        
        # If we found an improvement, we can break or continue to find more
        # Breaking is safer to avoid oscillation, but continuing might find better.
        # Let's break to keep it stable and let the outer loop handle restarts.
        break
        
    return improved, changed

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
    
    num_trials = 20
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
        
        # Local improvement on the current family
        if time.time() < deadline - 15.0:
            # 1-swap improvement
            improved, changed = _local_improve_1swap(family, all_sets, n_total, deadline)
            if len(improved) > best_size:
                best_size = len(improved)
                best_family = improved
            
            # 2-swap improvement
            if time.time() < deadline - 5.0:
                improved2, changed2 = _local_improve_2swap(improved, all_sets, n_total, deadline)
                if len(improved2) > best_size:
                    best_size = len(improved2)
                    best_family = improved2
    
    # Final local improvement on best family
    if time.time() < deadline - 5.0 and best_family:
        final, _ = _local_improve_1swap(best_family, all_sets, n_total, deadline)
        if len(final) > best_size:
            best_family = final
        if time.time() < deadline - 1.0:
            final2, _ = _local_improve_2swap(final, all_sets, n_total, deadline)
            if len(final2) > best_size:
                best_family = final2

    # Convert bitmasks back to sorted lists
    result = []
    for mask in best_family:
        elements = []
        for i in range(16):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
    
    return result
