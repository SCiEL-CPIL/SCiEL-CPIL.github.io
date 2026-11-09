import time
import random
import numpy as np
from itertools import combinations

N = 256
DEADLINE = time.time() + 55

# EVOLVE-BLOCK-START

def _build_sums(selected):
    """Build the set of all unordered pair sums a+b with a<=b."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums

def _is_maximal(selected, sums):
    """
    Check if the set is inclusion-maximal.
    """
    sset = set(selected)
    for x in range(1, N + 1):
        if x in sset:
            continue
        
        # Check if x can be added
        if 2 * x in sums:
            continue
        
        # Check if x conflicts with any existing sum via x+a
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        
        if not blocked:
            return False
            
    return True

def _get_addable_elements(selected, sums):
    """
    Return a list of elements x in [1, N] \ selected that can be added to selected.
    """
    sset = set(selected)
    addable = []
    for x in range(1, N + 1):
        if x in sset:
            continue
        
        if 2 * x in sums:
            continue
        
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        
        if not blocked:
            addable.append(x)
            
    return addable

def _greedy_core_and_cover(rng, core_size=15):
    """
    1. Generate a random Sidon set of size core_size.
    2. Iteratively add elements to block all addable elements (Greedy Set Cover).
    """
    # Step 1: Generate Core
    selected = []
    sums = set()
    perm = list(range(1, N + 1))
    rng.shuffle(perm)
    
    for x in perm:
        if len(selected) >= core_size:
            break
        if 2 * x in sums:
            continue
        conflict = False
        for a in selected:
            if x + a in sums:
                conflict = True
                break
        if conflict:
            continue
            
        selected.append(x)
        sums.add(2 * x)
        for a in selected[:-1]:
            sums.add(x + a)
            
    # Step 2: Cover Gaps
    while time.time() < DEADLINE - 2:
        addable = _get_addable_elements(selected, sums)
        if not addable:
            break
            
        # Find the best candidate to add
        # Candidates should be elements that can be added to current selected
        # AND block the most 'addable' elements
        best_candidate = None
        max_blocked = -1
        
        # We can only add elements that are themselves addable to the current set
        # Actually, any element that keeps the set Sidon is a candidate.
        # But if an element is NOT addable, it's already blocked, so adding it is redundant?
        # No, adding a blocked element might block OTHER elements.
        # Example: S={1,3}. 2 is blocked (2+1=3=1+2? No. 1+3=4, 2+2=4. Conflict).
        # So 2 is blocked.
        # If we add 2, S becomes {1,2,3} which is not Sidon.
        # So we can only add elements that are ADDABLE to the current set.
        
        candidates = addable
        
        if not candidates:
            break
            
        # Sample if too many
        if len(candidates) > 50:
            candidates = rng.sample(candidates, 50)
            
        for c in candidates:
            # Check if c is addable (it is, by definition of 'addable')
            # Calculate how many 'addable' elements would be blocked if we add c
            # New set: S_new = S U {c}
            # New sums: sums U {2c} U {c+a for a in S}
            
            new_sums = set(sums)
            new_sums.add(2 * c)
            for a in selected:
                new_sums.add(c + a)
                
            blocked_count = 0
            for y in addable:
                if y == c:
                    continue
                # Check if y is blocked in S_new
                # y is blocked if 2y in new_sums or y+a in new_sums for some a in S_new
                if 2 * y in new_sums:
                    blocked_count += 1
                else:
                    # Check against all a in S_new
                    # a can be in S or a=c
                    blocked = False
                    for a in selected:
                        if y + a in new_sums:
                            blocked = True
                            break
                    if not blocked and y + c in new_sums:
                        blocked = True
                        
                    if blocked:
                        blocked_count += 1
                        
            if blocked_count > max_blocked:
                max_blocked = blocked_count
                best_candidate = c
                
        if best_candidate is None:
            # Should not happen if addable is not empty
            break
            
        # Add best_candidate
        c = best_candidate
        selected.append(c)
        sums.add(2 * c)
        for a in selected[:-1]:
            sums.add(c + a)
            
    return selected, sums

def _local_search_swap(selected, sums, rng, time_limit):
    """
    Local search using swaps to potentially find a configuration that allows
    a different cover strategy? 
    Actually, since we can't remove, swaps don't reduce size.
    However, a swap might lead to a set that is "easier" to cover in a restart?
    No, we are already in the cover phase.
    
    Let's stick to the construction.
    """
    pass

def construct():
    rng = random.Random()
    best_size = N + 1
    best_result = None
    best_sums = None
    
    start = time.time()
    
    # Try different core sizes
    core_sizes = [10, 12, 15, 20, 25, 30]
    
    # Phase 1: Core + Cover
    for core_size in core_sizes:
        for _ in range(10): # 10 restarts per core size
            if time.time() > DEADLINE - 5:
                break
            selected, sums = _greedy_core_and_cover(rng, core_size)
            if len(selected) < best_size:
                best_size = len(selected)
                best_result = list(selected)
                best_sums = set(sums)
                
    # Phase 2: Pure Dominating Greedy (from original) as backup
    if time.time() < DEADLINE - 10:
        for _ in range(5):
            perm = list(range(1, N + 1))
            rng.shuffle(perm)
            selected = []
            sums = set()
            remaining = set(perm)
            
            while remaining and time.time() < DEADLINE - 2:
                best_candidate = None
                best_block_count = -1
                
                candidates = list(remaining)
                if len(candidates) > 30:
                    candidates = rng.sample(candidates, 30)
                
                for x in candidates:
                    if 2 * x in sums:
                        continue
                    conflict = False
                    for a in selected:
                        if x + a in sums:
                            conflict = True
                            break
                    if conflict:
                        continue
                    
                    block_count = 0
                    new_sums = sums | {2 * x}
                    for a in selected:
                        new_sums.add(x + a)
                    
                    for y in remaining:
                        if y == x:
                            continue
                        if 2 * y in new_sums:
                            block_count += 1
                        else:
                            for a in selected + [x]:
                                if y + a in new_sums:
                                    block_count += 1
                                    break
                                    
                    if block_count > best_block_count:
                        best_block_count = block_count
                        best_candidate = x
                        
                if best_candidate is None:
                    break
                    
                x = best_candidate
                selected.append(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
                
                remaining.discard(x)
                
                still_remaining = set()
                for y in remaining:
                    if 2 * y in sums:
                        continue
                    blocked = False
                    for a in selected:
                        if y + a in sums:
                            blocked = True
                            break
                    if not blocked:
                        still_remaining.add(y)
                remaining = still_remaining
                
            if len(selected) < best_size:
                best_size = len(selected)
                best_result = list(selected)
                best_sums = set(sums)

    if best_result is None:
        # Fallback
        selected, sums = [], set()
        for x in range(1, N + 1):
            if 2 * x in sums: continue
            conflict = False
            for a in selected:
                if x + a in sums:
                    conflict = True
                    break
            if conflict: continue
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
        best_result = selected

    return best_result

# EVOLVE-BLOCK-END
