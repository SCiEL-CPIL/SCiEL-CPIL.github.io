"""Improved program: fast iterative random greedy restarts with sorted-list collision detection for dissociated reciprocal sets."""
import time
import random
from bisect import bisect_left

# EVOLVE-BLOCK-START

# LCM of 1..22 is 232792560
_LCM = 232792560
_INV_INT = {x: _LCM // x for x in range(1, 23)}

def _build_sorted_sums(elements):
    """
    Build sorted list of all subset sums for elements.
    Returns None if a collision is detected.
    """
    if not elements:
        return [0]
    sums = [0]
    for x in elements:
        inv = _INV_INT[x]
        new_sums = []
        for s in sums:
            new_sums.append(s + inv)
        # Merge sums and new_sums, checking for duplicates
        merged = []
        i, j = 0, 0
        while i < len(sums) or j < len(new_sums):
            if j >= len(new_sums) or (i < len(sums) and sums[i] < new_sums[j]):
                val = sums[i]
                i += 1
            elif i >= len(sums) or new_sums[j] < sums[i]:
                val = new_sums[j]
                j += 1
            else:
                # sums[i] == new_sums[j] -> collision
                return None
            if merged and merged[-1] == val:
                return None
            merged.append(val)
        sums = merged
    return sums

def _is_dissociated_fast(elements):
    """
    Fast check: can we add element to current set while maintaining dissociation?
    Uses sorted list approach.
    """
    return _build_sorted_sums(elements) is not None

def _can_add(current_sums, x):
    """
    Check if element x can be added to a set with current_sums (sorted list of all subset sums).
    Returns new sorted sums list, or None if collision.
    """
    inv = _INV_INT[x]
    new_sums = []
    for s in current_sums:
        new_sums.append(s + inv)
    # Merge current_sums and new_sums, checking for duplicates
    merged = []
    i, j = 0, 0
    while i < len(current_sums) or j < len(new_sums):
        if j >= len(new_sums) or (i < len(current_sums) and current_sums[i] < new_sums[j]):
            val = current_sums[i]
            i += 1
        elif i >= len(current_sums) or new_sums[j] < current_sums[i]:
            val = new_sums[j]
            j += 1
        else:
            # Collision
            return None
        if merged and merged[-1] == val:
            return None
        merged.append(val)
    return merged

def construct():
    start_time = time.time()
    # Use 55s out of 60s for safety
    deadline = start_time + 55
    
    all_elements = list(range(1, 23))
    
    best_set = set()
    best_size = 0
    
    # Initial greedy pass to get a decent starting point
    for _ in range(30):
        perm = all_elements[:]
        random.shuffle(perm)
        current_set = set()
        current_sums = [0]
        for x in perm:
            new_sums = _can_add(current_sums, x)
            if new_sums is not None:
                current_set.add(x)
                current_sums = new_sums
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    if best_size == 0:
        best_set = {1}
        best_size = 1

    # Random greedy restarts
    while time.time() < deadline:
        perm = all_elements[:]
        random.shuffle(perm)
        
        current_set = set()
        current_sums = [0]
        for x in perm:
            if time.time() > deadline - 1:
                break
            new_sums = _can_add(current_sums, x)
            if new_sums is not None:
                current_set.add(x)
                current_sums = new_sums
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        # Local search: try to add more elements to the current set
        if len(current_set) >= best_size and time.time() < deadline - 2:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                candidates = [x for x in all_elements if x not in current_set]
                random.shuffle(candidates)
                for x in candidates:
                    if time.time() > deadline - 2:
                        break
                    new_sums = _can_add(current_sums, x)
                    if new_sums is not None:
                        current_set.add(x)
                        current_sums = new_sums
                        improved = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                        break

    return sorted(best_set)

# EVOLVE-BLOCK-END
