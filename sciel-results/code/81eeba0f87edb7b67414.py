"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import sys
from itertools import combinations

def _precompute_conflicts():
    """For each x in 1..24, find all minimal subsets T of [1,24]\\{x}
    such that sum(1/y for y in T) == 1/x.
    Return dict: x -> list of frozensets (minimal conflict sets).
    """
    conflicts = {x: [] for x in range(1, 25)}
    full = list(range(1, 25))
    
    for x in range(1, 25):
        target = Fraction(1, x)
        available = [y for y in full if y != x]
        n = len(available)
        
        # Use meet-in-the-middle for each x
        mid = n // 2
        left = available[:mid]
        right = available[mid:]
        
        # Generate all subset sums for left half
        left_sums = {}  # sum -> list of masks
        for mask in range(1 << len(left)):
            s = Fraction(0, 1)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += Fraction(1, left[i])
            if s <= target:
                if s not in left_sums:
                    left_sums[s] = []
                left_sums[s].append(mask)
        
        # Generate all subset sums for right half
        right_sums = {}  # sum -> list of masks
        for mask in range(1 << len(right)):
            s = Fraction(0, 1)
            for i in range(len(right)):
                if mask & (1 << i):
                    s += Fraction(1, right[i])
            if s <= target:
                if s not in right_sums:
                    right_sums[s] = []
                right_sums[s].append(mask)
        
        # Find all pairs that sum to target
        found_sets = set()
        
        # Case 1: left subset alone sums to target
        if target in left_sums:
            for mask in left_sums[target]:
                elems = frozenset(left[i] for i in range(len(left)) if mask & (1 << i))
                found_sets.add(elems)
        
        # Case 2: right subset alone sums to target
        if target in right_sums:
            for mask in right_sums[target]:
                elems = frozenset(right[i] for i in range(len(right)) if mask & (1 << i))
                found_sets.add(elems)
        
        # Case 3: left + right subset sums to target
        for sL, masksL in left_sums.items():
            need = target - sL
            if need in right_sums:
                for maskL in masksL:
                    for maskR in right_sums[need]:
                        elems = set()
                        for i in range(len(left)):
                            if maskL & (1 << i):
                                elems.add(left[i])
                        for i in range(len(right)):
                            if maskR & (1 << i):
                                elems.add(right[i])
                        if elems:
                            found_sets.add(frozenset(elems))
        
        # Keep only minimal sets (no proper subset also sums to target)
        minimal = []
        for fs in found_sets:
            is_minimal = True
            for other in found_sets:
                if other != fs and other < fs:
                    is_minimal = False
                    break
            if is_minimal:
                minimal.append(fs)
        
        conflicts[x] = minimal
    
    return conflicts

def _is_safe(x, S, conflicts):
    """Check if x is safe in set S: no conflict set for x is subset of S\{x}."""
    others = frozenset(S) - {x}
    for cset in conflicts[x]:
        if cset <= others:
            return False
    return True

def _full_validate(S, conflicts):
    """Check if entire set S is valid."""
    for x in S:
        if not _is_safe(x, S, conflicts):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Precompute conflicts
    conflicts = _precompute_conflicts()
    
    full = list(range(1, 25))
    
    # Phase 1: Greedy from largest to smallest
    # Key insight: adding smaller elements never invalidates larger ones
    # because 1/small > 1/large, so any subset containing the small element
    # has sum > 1/small > 1/large
    selected = set()
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        if _is_safe(i, selected | {i}, conflicts):
            selected.add(i)
    
    best = set(selected)
    best_size = len(best)
    
    # Phase 2: Try different starting points for greedy
    for first in range(24, 0, -1):
        if time.time() > deadline:
            break
        sel2 = {first}
        for i in range(24, 0, -1):
            if i == first:
                continue
            if time.time() > deadline:
                break
            if _is_safe(i, sel2 | {i}, conflicts):
                sel2.add(i)
        if len(sel2) > best_size:
            best = set(sel2)
            best_size = len(sel2)
    
    # Phase 3: Try to add elements to current best
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        if i in best:
            continue
        if _is_safe(i, best | {i}, conflicts):
            # Adding i is safe for i itself; need to check it doesn't break others
            # But by the monotonicity property, adding a smaller element
            # cannot break larger elements. However, it could break other smaller elements.
            # Actually, we need to verify the full set.
            trial = set(best)
            trial.add(i)
            if _full_validate(trial, conflicts):
                best = trial
                best_size = len(best)
    
    # Phase 4: Local search - try swaps to maintain or increase size
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = set(best)
        current_size = len(current)
        
        # Try removing one element and adding a different one
        for remove in sorted(current, reverse=True):
            if time.time() > deadline:
                break
            for add in range(24, 0, -1):
                if add in current:
                    continue
                candidate = set(current)
                candidate.remove(remove)
                candidate.add(add)
                if _full_validate(candidate, conflicts):
                    if len(candidate) > current_size:
                        best = candidate
                        best_size = len(candidate)
                        improved = True
                        break
                    # If same size, we could also accept for diversity,
                    # but let's focus on size improvement first
            if improved:
                break
        
        # Try removing one element and adding two others
        if not improved and time.time() < deadline:
            for remove in sorted(current, reverse=True):
                if time.time() > deadline:
                    break
                base = set(current)
                base.remove(remove)
                for add1 in range(24, 0, -1):
                    if add1 in base:
                        continue
                    if time.time() > deadline:
                        break
                    for add2 in range(add1 - 1, 0, -1):
                        if add2 in base:
                            continue
                        if time.time() > deadline:
                            break
                        candidate = set(base)
                        candidate.add(add1)
                        candidate.add(add2)
                        if len(candidate) > current_size:
                            if _full_validate(candidate, conflicts):
                                best = candidate
                                best_size = len(candidate)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
    
    # Phase 5: Final attempt - try all single additions again
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        if i in best:
            continue
        trial = set(best)
        trial.add(i)
        if _full_validate(trial, conflicts):
            best = trial
            best_size = len(best)
    
    # Convert to sorted list
    result = sorted(best)
    
    # Ensure non-empty
    if not result:
        result = [24]
    
    return result

# EVOLVE-BLOCK-END
