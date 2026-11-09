import time
import sys

START_TIME = time.time()
DEADLINE = 55.0

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

import random

def can_add_element(current_bitsets, x, n):
    """Check if adding x to a set with current bitsets is valid.
    current_bitsets[j] has bit s set if sum s achievable with cardinality j.
    Returns new bitsets if valid, None otherwise.
    """
    new_bitsets = [0] * (n + 1)
    for j in range(n + 1):
        new_bitsets[j] = current_bitsets[j]
    
    # Add x: for each cardinality j, new sums = old_sums + x
    for j in range(n, -1, -1):
        if current_bitsets[j] != 0:
            new_sums = current_bitsets[j] << x
            # Check collision with all other cardinalities (including j+1 after update)
            for k in range(n + 1):
                if k == j + 1:
                    # This is the target cardinality, will be OR'd
                    continue
                if new_sums & current_bitsets[k]:
                    return None
            # Also check against newly added sums in higher cardinalities
            # Since we process from high to low, j+1 already has new_sums from previous iteration
            # Actually, we need to check against ALL existing bitsets including those just updated
            # Let me redo this more carefully
            
    # Better approach: compute all new bitsets first, then check
    result = [0] * (n + 1)
    for j in range(n + 1):
        result[j] = current_bitsets[j]
        if current_bitsets[j] != 0:
            result[j] |= (current_bitsets[j] << x)
    
    # Check for collisions between different cardinalities
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if result[j] & result[k]:
                return None
    
    return result

def greedy_construct():
    """Greedy construction: add elements one by one, trying candidates in order."""
    # Start with empty set
    bitsets = [1]  # Only cardinality 0, sum 0
    current_set = []
    
    # Try adding elements from 1 to 256
    # Use a priority: prefer elements that create more "space"
    candidates = list(range(1, 257))
    
    while not is_done():
        added = False
        for x in candidates:
            if x in current_set:
                continue
            new_bitsets = can_add_element(bitsets, x, len(current_set))
            if new_bitsets is not None:
                bitsets = new_bitsets
                current_set.append(x)
                added = True
                break
        if not added:
            break
    
    return current_set

def local_improve(initial_set):
    """Try to improve the set by swapping elements."""
    current_set = initial_set[:]
    n = len(current_set)
    
    # Compute current bitsets
    bitsets = [0] * (n + 1)
    bitsets[0] = 1
    for x in current_set:
        new_bitsets = [0] * (n + 1)
        for j in range(n + 1):
            new_bitsets[j] = bitsets[j]
            if bitsets[j]:
                new_bitsets[j] |= (bitsets[j] << x)
        bitsets = new_bitsets
    
    best_set = current_set[:]
    best_size = n
    
    # Try swapping each element with a candidate
    all_elements = set(range(1, 257))
    current_elements = set(current_set)
    candidates = list(all_elements - current_elements)
    
    random.shuffle(candidates)
    
    improved = True
    while improved and not is_done():
        improved = False
        for i in range(len(current_set)):
            for x_new in candidates:
                if is_done():
                    return best_set
                # Try swapping current_set[i] with x_new
                test_set = current_set[:]
                test_set[i] = x_new
                test_set.sort()
                
                # Check validity
                n_test = len(test_set)
                test_bitsets = [0] * (n_test + 1)
                test_bitsets[0] = 1
                valid = True
                for x in test_set:
                    new_test = [0] * (n_test + 1)
                    for j in range(n_test + 1):
                        new_test[j] = test_bitsets[j]
                        if test_bitsets[j]:
                            new_test[j] |= (test_bitsets[j] << x)
                    test_bitsets = new_test
                
                for j in range(n_test + 1):
                    for k in range(j + 1, n_test + 1):
                        if test_bitsets[j] & test_bitsets[k]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Check if this swap allows adding more elements
                    # For now, just accept if it doesn't reduce size
                    # Actually, swaps keep same size, so we need a different criterion
                    # Let's just do swaps to enable future additions
                    # Skip for now, focus on additions
                    pass
    
    return best_set

def construct():
    # Try greedy construction
    result = greedy_construct()
    
    # Also try a different initialization: start with large gaps
    # Geometric-like: 1, 3, 9, 27, ... but capped at 256
    geo = [1]
    while geo[-1] * 3 <= 256:
        geo.append(geo[-1] * 3)
    
    # Try to build from geometric base
    bitsets = [1]
    current_set = []
    for x in geo:
        new_bitsets = can_add_element(bitsets, x, len(current_set))
        if new_bitsets is not None:
            bitsets = new_bitsets
            current_set.append(x)
    
    if len(current_set) > len(result):
        result = current_set
    
    # Try to add more elements to the best result
    if not is_done():
        all_elements = set(range(1, 257))
        current_elements = set(result)
        candidates = list(all_elements - current_elements)
        random.shuffle(candidates)
        
        bitsets = [0] * (len(result) + 1)
        bitsets[0] = 1
        for x in result:
            new_bitsets = [0] * (len(result) + 1)
            for j in range(len(result) + 1):
                new_bitsets[j] = bitsets[j]
                if bitsets[j]:
                    new_bitsets[j] |= (bitsets[j] << x)
            bitsets = new_bitsets
        
        for x in candidates:
            if is_done():
                break
            new_bitsets = can_add_element(bitsets, x, len(result))
            if new_bitsets is not None:
                bitsets = new_bitsets
                result.append(x)
    
    # Ensure we return a valid list of ints
    result = sorted(set(result))
    result = [int(x) for x in result]
    
    # Final validation
    if len(result) == 0:
        return [1, 2]
    
    n = len(result)
    final_bitsets = [0] * (n + 1)
    final_bitsets[0] = 1
    for x in result:
        new_final = [0] * (n + 1)
        for j in range(n + 1):
            new_final[j] = final_bitsets[j]
            if final_bitsets[j]:
                new_final[j] |= (final_bitsets[j] << x)
        final_bitsets = new_final
    
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if final_bitsets[j] & final_bitsets[k]:
                # Invalid, fall back to simple valid set
                return [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
