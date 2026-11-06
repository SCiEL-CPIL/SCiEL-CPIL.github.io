"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

# Precompute all 4-subsets of [1,16] as bitmasks (0-indexed bits for 0..15)
_ALL_MASKS = []
_MASK_TO_IDX = {}
for c in combinations(range(16), 4):
    mask = 0
    for i in c:
        mask |= (1 << i)
    _ALL_MASKS.append(mask)
    _MASK_TO_IDX[mask] = len(_ALL_MASKS) - 1

_N = len(_ALL_MASKS)  # 1820

# Precompute pairwise intersection masks for speed
_PAIR_INTERSECT = np.zeros((_N, _N), dtype=np.int32)
for i in range(_N):
    mi = _ALL_MASKS[i]
    for j in range(_N):
        mj = _ALL_MASKS[j]
        _PAIR_INTERSECT[i, j] = mi & mj

def _is_sunflower(a, b, c):
    """Check if three 4-element sets (as bitmasks) form a sunflower."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _can_add(candidate, chosen_indices):
    """Check if candidate bitmask can be added to family of chosen_indices."""
    n = len(chosen_indices)
    for i in range(n):
        ai = _ALL_MASKS[chosen_indices[i]]
        for j in range(i + 1, n):
            aj = _ALL_MASKS[chosen_indices[j]]
            if _is_sunflower(ai, aj, candidate):
                return False
    return True

def _can_add_fast(candidate_idx, chosen_indices):
    """Faster check using precomputed intersections where possible."""
    n = len(chosen_indices)
    cand = _ALL_MASKS[candidate_idx]
    for i in range(n):
        ai = _ALL_MASKS[chosen_indices[i]]
        ac = ai & cand
        for j in range(i + 1, n):
            aj = _ALL_MASKS[chosen_indices[j]]
            ab = ai & aj
            bc = aj & cand
            if ab == ac == bc:
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

def _try_add_any(family_indices, deadline):
    """Try to add any remaining subset to the family."""
    family_set = set(family_indices)
    for idx in range(_N):
        if time.time() > deadline:
            break
        if idx in family_set:
            continue
        if _can_add(_ALL_MASKS[idx], family_indices):
            return family_indices + [idx], idx
    return family_indices, None

def _local_improve(family_indices, deadline):
    """Try to improve the family by swapping out sets to allow more additions."""
    family_set = set(family_indices)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(family_indices)):
            if time.time() > deadline:
                break
            removed = family_indices[i]
            remaining = family_indices[:i] + family_indices[i+1:]
            best_candidate = None
            for idx in range(_N):
                if idx in family_set:
                    continue
                if _can_add(_ALL_MASKS[idx], remaining):
                    best_candidate = idx
                    break
            if best_candidate is not None:
                family_indices[i] = best_candidate
                family_set.remove(removed)
                family_set.add(best_candidate)
                improved = True
    return family_indices

def _simulated_annealing(initial_family, deadline, rng):
    """Improve family using simulated annealing with add/swap/remove moves."""
    family = list(initial_family)
    family_set = set(family)
    current_size = len(family)
    best_family = list(family)
    best_size = current_size
    
    t = 1.0
    cooling = 0.999
    
    while time.time() < deadline and t > 0.005:
        t *= cooling
        
        move_type = rng.random()
        
        if move_type < 0.35:
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
                # With some probability, remove a random set and try again
                if rng.random() < t * 0.05 and family:
                    rem_idx = rng.randint(0, len(family) - 1)
                    removed = family.pop(rem_idx)
                    family_set.discard(removed)
                    current_size -= 1
                    if _can_add(mask, family):
                        family.append(candidate)
                        family_set.add(candidate)
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_family = list(family)
                    else:
                        # Put it back
                        family.append(removed)
                        family_set.add(removed)
                        current_size += 1
        
        elif move_type < 0.65:
            # Swap: remove one, try to add another
            if not family:
                continue
            rem_idx = rng.randint(0, len(family) - 1)
            removed = family[rem_idx]
            test_family = family[:rem_idx] + family[rem_idx+1:]
            # Try a few random candidates
            for _ in range(8):
                if time.time() > deadline:
                    break
                candidate = rng.randint(0, _N - 1)
                if candidate in family_set:
                    continue
                mask = _ALL_MASKS[candidate]
                if _can_add(mask, test_family):
                    family[rem_idx] = candidate
                    family_set.discard(removed)
                    family_set.add(candidate)
                    improved = True
                    break
        
        else:
            # Remove a random set (to escape local optima)
            if family and rng.random() < t * 0.02:
                rem_idx = rng.randint(0, len(family) - 1)
                removed = family.pop(rem_idx)
                family_set.discard(removed)
                current_size -= 1
    
    return best_family

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    best_family = []
    
    # Strategy 1: Lexicographic order
    lex_order = list(range(_N))
    family1 = _greedy_build(lex_order, deadline)
    if len(family1) > len(best_family):
        best_family = family1
    
    # Strategy 2: Multiple random restarts
    rng = random.Random(42)
    num_restarts = 150
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        order = list(range(_N))
        rng.shuffle(order)
        family = _greedy_build(order, deadline)
        if len(family) > len(best_family):
            best_family = family
    
    # Strategy 3: Simulated annealing on best family
    if time.time() < deadline:
        sa_family = _simulated_annealing(list(best_family), deadline, rng)
        if len(sa_family) > len(best_family):
            best_family = sa_family
    
    # Strategy 4: Try to add more to the best family
    if time.time() < deadline:
        new_family, _ = _try_add_any(list(best_family), deadline)
        if len(new_family) > len(best_family):
            best_family = new_family
    
    # Strategy 5: Local improvement on best family
    if time.time() < deadline:
        improved = _local_improve(list(best_family), deadline)
        if len(improved) > len(best_family):
            best_family = improved
        # Try to add more after improvement
        if time.time() < deadline:
            new_family, _ = _try_add_any(list(best_family), deadline)
            if len(new_family) > len(best_family):
                best_family = new_family
    
    # Convert back to sorted lists of 1-based integers
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
