import time
import numpy as np
from itertools import combinations
import random
import math

# EVOLVE-BLOCK-START
# Precompute weights based on LCM of 1..22
_LCM_22 = 1
for _i in range(1, 23):
    _LCM_22 = math.lcm(_LCM_22, _i)

_WEIGHTS = {}
for _i in range(1, 23):
    _WEIGHTS[_i] = _LCM_22 // _i

def _can_add(sums, seen, w):
    """
    Check if adding weight w to the set of sums 'sums' creates any duplicates.
    Returns True if valid, and the new list of sums if valid.
    """
    new_sums = [s + w for s in sums]
    for ns in new_sums:
        if ns in seen:
            return False, None
    return True, new_sums

def _rebuild_sums(elems):
    """
    Rebuild sums and seen set for a given set of elements.
    Returns (sums, seen, valid).
    """
    sums = [0]
    seen = {0}
    for e in elems:
        w = _WEIGHTS[e]
        valid, new_sums = _can_add(sums, seen, w)
        if not valid:
            return None, None, False
        sums.extend(new_sums)
        seen.update(new_sums)
    return sums, seen, True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    best_set = []
    best_size = 0
    
    # We will use randomized local search with incremental sum updates.
    # State: current_set (set of ints), current_sums (list of int sums), seen_sums (set of int sums)
    
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        
        # Randomize the order of elements to try
        elements = list(range(1, 23))
        random.shuffle(elements)
        
        current_set = set()
        current_sums = [0]
        seen_sums = {0}
        
        # Greedy phase: try to add elements from the randomized list
        for e in elements:
            if e in current_set:
                continue
            w = _WEIGHTS[e]
            valid, new_sums = _can_add(current_sums, seen_sums, w)
            if valid:
                current_set.add(e)
                current_sums.extend(new_sums)
                seen_sums.update(new_sums)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        # Local search: try to improve current_set by swaps or additions
        # We try to add any element not in the set. If none, try swapping one out and one in.
        # To avoid getting stuck, we do a limited number of swap attempts.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any element
            added = False
            for e in range(1, 23):
                if e in current_set:
                    continue
                w = _WEIGHTS[e]
                valid, new_sums = _can_add(current_sums, seen_sums, w)
                if valid:
                    current_set.add(e)
                    current_sums.extend(new_sums)
                    seen_sums.update(new_sums)
                    improved = True
                    added = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    break # Restart the loop to try adding more
            
            if added:
                continue # Restart while loop to try adding more
            
            # 2. If no addition possible, try swaps
            # A. Try 1-for-1 swap (remove 1, add 1)
            swapped = False
            for _ in range(20): # Limit swap attempts to keep restarts fast
                if time.time() > deadline:
                    break
                
                if not current_set:
                    break
                    
                remove_e = random.choice(list(current_set))
                
                # Rebuild sums for temp_set
                temp_set = current_set - {remove_e}
                temp_sums, temp_seen, valid_temp = _rebuild_sums(temp_set)
                if not valid_temp:
                    continue
                
                # Try to add a different element to temp_set
                for add_e in range(1, 23):
                    if add_e in temp_set:
                        continue
                    w = _WEIGHTS[add_e]
                    valid, new_sums = _can_add(temp_sums, temp_seen, w)
                    if valid:
                        # Perform the swap
                        current_set = temp_set
                        current_set.add(add_e)
                        current_sums = temp_sums
                        current_sums.extend(new_sums)
                        seen_sums = temp_seen
                        seen_sums.update(new_sums)
                        
                        improved = True
                        swapped = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = list(current_set)
                        break # Restart the while loop to try adding more
                
                if swapped:
                    break # Break out of swap attempts, go back to while loop
            
            if swapped:
                continue

            # B. Try 1-for-2 swap (remove 1, add 2) if 1-for-1 failed
            # This allows net +1 size increase if direct add was blocked.
            if not swapped and len(current_set) >= 1:
                for _ in range(10): # Fewer attempts as this is more expensive
                    if time.time() > deadline:
                        break
                    
                    remove_e = random.choice(list(current_set))
                    temp_set = current_set - {remove_e}
                    temp_sums, temp_seen, valid_temp = _rebuild_sums(temp_set)
                    if not valid_temp:
                        continue
                    
                    # Try to add TWO elements to temp_set
                    # We need to find two distinct elements a, b not in temp_set such that
                    # temp_set + {a} is valid, and (temp_set + {a}) + {b} is valid.
                    # Or more generally, the set of sums of temp_set U {a, b} is distinct.
                    
                    candidates = [e for e in range(1, 23) if e not in temp_set]
                    random.shuffle(candidates)
                    
                    found_pair = False
                    for a in candidates:
                        if time.time() > deadline:
                            break
                        w_a = _WEIGHTS[a]
                        valid_a, sums_a = _can_add(temp_sums, temp_seen, w_a)
                        if not valid_a:
                            continue
                        
                        # Now try to add b to the set (temp_set U {a})
                        # We need to check against sums_a and seen_a (which is temp_seen U sums_a)
                        # Actually, we just need to check if adding b to sums_a creates duplicates with (temp_seen U sums_a).
                        # The set of sums for temp_set U {a} is temp_sums + sums_a.
                        # The seen set is temp_seen U sums_a.
                        seen_a = temp_seen.union(sums_a)
                        
                        for b in candidates:
                            if b == a:
                                continue
                            w_b = _WEIGHTS[b]
                            # Check if adding w_b to sums_a creates any sum already in seen_a
                            # The new sums will be [s + w_b for s in sums_a]
                            # We must ensure none of these are in seen_a.
                            # Note: sums_a are the NEW sums generated by adding a.
                            # The existing sums are temp_sums.
                            # The total sums are temp_sums + sums_a.
                            # We are adding b. The new sums are [s + w_b for s in (temp_sums + sums_a)].
                            # Wait, the standard _can_add assumes 'sums' is the list of ALL current sums.
                            # Here, the "current sums" for the set temp_set U {a} is temp_sums + sums_a.
                            # So we should call _can_add(temp_sums + sums_a, seen_a, w_b).
                            
                            # Optimization: We can just check if any s+w_b is in seen_a for s in (temp_sums + sums_a).
                            # But _can_add does exactly this.
                            valid_b, new_sums_b = _can_add(temp_sums + sums_a, seen_a, w_b)
                            if valid_b:
                                # Perform the 1-for-2 swap
                                current_set = temp_set
                                current_set.add(a)
                                current_set.add(b)
                                
                                # Update sums and seen
                                # Current sums were temp_sums + sums_a.
                                # New sums are (temp_sums + sums_a) + new_sums_b.
                                current_sums = temp_sums + sums_a + new_sums_b
                                seen_sums = seen_a.union(new_sums_b)
                                
                                improved = True
                                swapped = True
                                if len(current_set) > best_size:
                                    best_size = len(current_set)
                                    best_set = list(current_set)
                                found_pair = True
                                break
                        if found_pair:
                            break
                    if swapped:
                        break # Break out of 1-for-2 attempts

    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
