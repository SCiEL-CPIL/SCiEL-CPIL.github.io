"""
Optimized search for maximum-cardinality reciprocal-dissociated subset of [1..22].
Uses systematic backtracking with multiple ordering strategies and local search.
"""
import time
import random
from itertools import combinations

# LCM of 1..22
_LCM = 232792560

def _can_add(v, sums):
    """Check if adding value v to current sums causes any collision."""
    # Collision if (s + v) is already in sums for any s in sums
    # Equivalently, if there exist s1, s2 in sums such that s1 + v = s2
    # i.e., s2 - s1 = v
    for s in sums:
        if (s + v) in sums:
            return False
    return True

def _add_value(v, sums):
    """Add value v to the sums set, returning the new set."""
    new_sums = sums | {s + v for s in sums}
    return new_sums

def _backtrack_search(elements, current_set, current_sums, best_set, best_size, deadline):
    """
    Systematic backtracking: try to add each remaining element.
    Returns the best set found.
    """
    if time.time() > deadline:
        return best_set, best_size
    
    if not elements:
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        return best_set, best_size
    
    # Try adding each element in elements
    for idx in range(len(elements)):
        if time.time() > deadline:
            break
        
        elem = elements[idx]
        v = _LCM // elem
        
        # Check if we can add this element
        if _can_add(v, current_sums):
            new_sums = _add_value(v, current_sums)
            new_set = current_set + [elem]
            
            # Recurse with remaining elements
            remaining = elements[:idx] + elements[idx+1:]
            best_set, best_size = _backtrack_search(
                remaining, new_set, new_sums, best_set, best_size, deadline
            )
            
            # If we found a maximal set, no need to continue
            if best_size == 22:
                return best_set, best_size
        else:
            # Skip this element
            remaining = elements[:idx] + elements[idx+1:]
            best_set, best_size = _backtrack_search(
                remaining, current_set, current_sums, best_set, best_size, deadline
            )
            
            if best_size == 22:
                return best_set, best_size
    
    return best_set, best_size

def _local_search_add(subset, deadline):
    """
    Try to add any element not in subset that maintains dissociation.
    Returns the improved subset.
    """
    current_set = subset[:]
    values = [_LCM // i for i in current_set]
    sums = {0}
    for v in values:
        sums = _add_value(v, sums)
    
    all_elements = set(range(1, 23))
    current_set_set = set(current_set)
    
    # Try adding each missing element
    for i in sorted(all_elements - current_set_set, key=lambda x: _LCM // x):
        if time.time() > deadline:
            break
        
        v = _LCM // i
        if _can_add(v, sums):
            current_set.append(i)
            sums = _add_value(v, sums)
    
    return current_set

def _local_search_swap(subset, deadline):
    """
    Try to swap one element out and two in to increase cardinality.
    Returns the improved subset.
    """
    current_set = subset[:]
    current_set_set = set(current_set)
    all_elements = set(range(1, 23))
    missing = all_elements - current_set_set
    
    best_set = current_set[:]
    best_size = len(current_set)
    
    # Try removing each element and adding two others
    for remove_elem in list(current_set):
        if time.time() > deadline:
            break
        
        reduced_set = [x for x in current_set if x != remove_elem]
        values = [_LCM // i for i in reduced_set]
        sums = {0}
        for v in values:
            sums = _add_value(v, sums)
        
        # Try adding pairs of missing elements
        for add1 in missing:
            if time.time() > deadline:
                break
            
            v1 = _LCM // add1
            if not _can_add(v1, sums):
                continue
            
            sums1 = _add_value(v1, sums)
            
            for add2 in missing:
                if add2 == add1:
                    continue
                
                v2 = _LCM // add2
                if _can_add(v2, sums1):
                    new_set = reduced_set + [add1, add2]
                    if len(new_set) > best_size:
                        best_size = len(new_set)
                        best_set = new_set[:]
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin (90s limit, leave buffer)
    
    all_elements = list(range(1, 23))
    best_set = [1]  # Fallback
    best_size = 1
    
    random.seed(42)
    
    # Strategy 1: Systematic backtracking with different orderings
    # Sort by various criteria to guide the search
    
    orderings = [
        all_elements[:],           # Natural order
        sorted(all_elements, reverse=True),  # Descending
        sorted(all_elements, key=lambda x: _LCM // x),  # By weight ascending
        sorted(all_elements, key=lambda x: -(_LCM // x)),  # By weight descending
    ]
    
    # Add some random orderings
    for _ in range(5):
        perm = all_elements[:]
        random.shuffle(perm)
        orderings.append(perm)
    
    for perm in orderings:
        if time.time() > deadline:
            break
        
        # Run backtracking search with this ordering
        best_set, best_size = _backtrack_search(
            perm, [], {0}, best_set, best_size, deadline
        )
        
        # Apply local search improvements
        if time.time() < deadline:
            improved = _local_search_add(best_set, deadline)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
        
        if time.time() < deadline:
            improved = _local_search_swap(best_set, deadline)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
        
        if best_size == 22:
            break
    
    # If we still have time, try more random restarts
    while time.time() < deadline:
        perm = all_elements[:]
        random.shuffle(perm)
        
        best_set, best_size = _backtrack_search(
            perm, [], {0}, best_set, best_size, deadline
        )
        
        if time.time() < deadline:
            improved = _local_search_add(best_set, deadline)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
        
        if best_size == 22:
            break
    
    return best_set
