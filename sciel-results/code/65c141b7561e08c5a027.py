"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

# Precompute all 4-subsets of [1,16] as bitmasks (0-indexed bits for 0..15)
_ALL_MASKS = []
for c in combinations(range(16), 4):
    mask = 0
    for i in c:
        mask |= (1 << i)
    _ALL_MASKS.append(mask)

_N = len(_ALL_MASKS)  # 1820

# Precompute sunflower relation: for each pair (i,j), which k's form a sunflower with them
# sunflower_conflicts[i][j] = set of k such that (i,j,k) is a sunflower
# This is expensive to precompute fully, so we'll compute on-the-fly but cache

def _is_sunflower(a, b, c):
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _can_add(candidate, chosen_indices):
    """Check if candidate bitmask can be added to family of chosen_indices."""
    n = len(chosen_indices)
    for i in range(n):
        for j in range(i + 1, n):
            if _is_sunflower(_ALL_MASKS[chosen_indices[i]], 
                            _ALL_MASKS[chosen_indices[j]], 
                            candidate):
                return False
    return True

def _greedy_build(order, deadline):
    """Build a family using greedy addition in the given order."""
    chosen = []
    for idx in order:
        if time.time() > deadline:
            break
        mask = _ALL_MASKS[idx]
        if _can_add(mask, chosen):
            chosen.append(idx)
    return chosen

def _simulated_annealing(initial_family, deadline, rng):
    """Improve family using simulated annealing with add/remove/swap moves."""
    family = list(initial_family)
    family_set = set(family)
    current_size = len(family)
    best_family = list(family)
    best_size = current_size
    
    # Temperature schedule
    t = 1.0
    cooling = 0.9995
    
    while time.time() < deadline and t > 0.01:
        t *= cooling
        
        # Choose a random move type
        move_type = rng.random()
        
        if move_type < 0.4 and current_size < 250:
            # Try to add a random set
            candidate = rng.randint(0, _N - 1)
            if candidate in family_set:
                continue
            mask = _ALL_MASKS[candidate]
            if _can_add(mask, family):
                family.append(candidate)
                family_set.add(candidate)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_family = list(family)
            else:
                # With some probability, accept a "conflict" to escape local optima
                if rng.random() < t * 0.1:
                    # Remove a random set to make room
                    if family:
                        rem_idx = rng.randint(0, len(family) - 1)
                        removed = family.pop(rem_idx)
                        family_set.discard(removed)
                        current_size -= 1
                        # Try adding the candidate
                        if _can_add(mask, family):
                            family.append(candidate)
                            family_set.add(candidate)
                            current_size += 1
        
        elif move_type < 0.7:
            # Swap: remove one, add another
            if not family:
                continue
            rem_idx = rng.randint(0, len(family) - 1)
            removed = family[rem_idx]
            # Find a candidate that can be added if we remove 'removed'
            # Try a few random candidates
            for _ in range(5):
                if time.time() > deadline:
                    break
                candidate = rng.randint(0, _N - 1)
                if candidate in family_set:
                    continue
                # Temporarily remove and test
                test_family = family[:rem_idx] + family[rem_idx+1:]
                mask = _ALL_MASKS[candidate]
                if _can_add(mask, test_family):
                    # Accept the swap
                    family[rem_idx] = candidate
                    family_set.discard(removed)
                    family_set.add(candidate)
                    # current_size unchanged
                    break
        
        else:
            # Remove a random set (to escape bad configurations)
            if family and rng.random() < t * 0.5:
                rem_idx = rng.randint(0, len(family) - 1)
                removed = family.pop(rem_idx)
                family_set.discard(removed)
                current_size -= 1
    
    return best_family

def _local_search(family, deadline, rng):
    """Local search: repeatedly try to add any missing set."""
    family = list(family)
    family_set = set(family)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(_N):
            if time.time() > deadline:
                break
            if idx in family_set:
                continue
            mask = _ALL_MASKS[idx]
            if _can_add(mask, family):
                family.append(idx)
                family_set.add(idx)
                improved = True
    return family

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    rng = random.Random(12345)
    
    best_family = []
    best_size = 0
    
    # Phase 1: Multiple greedy restarts with different orderings
    # Use different heuristic orderings to diversify
    for trial in range(30):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = list(range(_N))  # Lexicographic
        elif trial == 1:
            order = list(range(_N))
            rng.shuffle(order)
        else:
            # Sort by some heuristic: e.g., number of elements in common with existing
            # For now, use random shuffle with different seeds
            order = list(range(_N))
            random.Random(trial * 7919).shuffle(order)
        
        family = _greedy_build(order, deadline)
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
        
        # Quick local search on this family
        if time.time() < deadline - 10:
            improved = _local_search(family, deadline - 5, rng)
            if len(improved) > best_size:
                best_size = len(improved)
                best_family = list(improved)
    
    # Phase 2: Simulated annealing from best family
    if time.time() < deadline - 15:
        sa_family = _simulated_annealing(best_family, deadline - 5, rng)
        if len(sa_family) > best_size:
            best_size = len(sa_family)
            best_family = list(sa_family)
        
        # Local search after SA
        if time.time() < deadline - 5:
            improved = _local_search(best_family, deadline - 2, rng)
            if len(improved) > best_size:
                best_size = len(improved)
                best_family = list(improved)
    
    # Phase 3: Final local search
    if time.time() < deadline - 2:
        improved = _local_search(best_family, deadline - 1, rng)
        if len(improved) > best_size:
            best_size = len(improved)
            best_family = list(improved)
    
    # Convert to sorted lists of 1-based integers
    result = []
    for idx in best_family:
        mask = _ALL_MASKS[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
