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

def _local_search(current_set, current_sums, deadline, all_elements):
    """Try to add more elements using multiple ordering strategies."""
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        candidates = [x for x in all_elements if x not in current_set]
        if not candidates:
            break
        # Try several orderings
        orderings = [
            candidates[:],  # natural order
            sorted(candidates, reverse=True),  # largest first
            sorted(candidates, key=lambda x: random.random()),  # random
        ]
        for ordering in orderings:
            for x in ordering:
                if time.time() > deadline - 2:
                    break
                new_sums = _can_add(current_sums, x)
                if new_sums is not None:
                    current_set.add(x)
                    current_sums = new_sums
                    improved = True
                    break
    return current_set, current_sums

def construct():
    start_time = time.time()
    # Use 55s out of 60s for safety
    deadline = start_time + 55
    
    all_elements = list(range(1, 23))
    
    best_set = set()
    best_size = 0
    
    # Initial greedy passes with different strategies
    for strategy in range(30):
        if strategy < 10:
            # Random order
            perm = all_elements[:]
            random.shuffle(perm)
        elif strategy < 20:
            # Larger elements first (less likely to collide)
            perm = sorted(all_elements, reverse=True)
        else:
            # Smaller elements first
            perm = all_elements[:]
        
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

    # Random greedy restarts with local search
    while time.time() < deadline:
        # Mix of strategies
        r = random.random()
        if r < 0.4:
            perm = all_elements[:]
            random.shuffle(perm)
        elif r < 0.7:
            perm = sorted(all_elements, reverse=True)
        else:
            perm = all_elements[:]
        
        current_set = set()
        current_sums = [0]
        for x in perm:
            if time.time() > deadline - 3:
                break
            new_sums = _can_add(current_sums, x)
            if new_sums is not None:
                current_set.add(x)
                current_sums = new_sums
        
        # Local search to try to improve
        current_set, current_sums = _local_search(current_set, current_sums, deadline, all_elements)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        # If we have a large set, do additional refinement
        if best_size >= 10 and time.time() < deadline - 5:
            # Try removing one element and adding two others
            for remove_x in list(current_set):
                if time.time() > deadline - 5:
                    break
                # Try to build a new set without remove_x
                new_set = current_set - {remove_x}
                # Rebuild sums from scratch for the new set
                new_sums = [0]
                for x in sorted(new_set):
                    ns = _can_add(new_sums, x)
                    if ns is None:
                        break
                    new_sums = ns
                else:
                    # Successfully rebuilt, now try to add elements
                    new_set, new_sums = _local_search(new_set, new_sums, deadline, all_elements)
                    if len(new_set) > best_size:
                        best_size = len(new_set)
                        best_set = set(new_set)

    return sorted(best_set)

# EVOLVE-BLOCK-END
