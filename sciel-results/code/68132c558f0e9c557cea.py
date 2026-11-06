import time
import random

def _has_distinct_subset_sums(arr):
    """Check if all subset sums are distinct."""
    sums = {0}
    for x in arr:
        new_sums = set()
        for s in sums:
            ns = s + x
            if ns in sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Initial valid set: powers of 2
    A = [1 << i for i in range(12)]
    best_A = A[:]
    best_max = max(A)
    
    random.seed(42)
    
    while time.time() < deadline:
        # 1. Try to reduce the maximum element of best_A
        improved = False
        max_idx = best_A.index(max(best_A))
        old_max_val = best_A[max_idx]
        
        for new_val in range(old_max_val - 1, 0, -1):
            if new_val in best_A:
                continue
            test_A = best_A[:]
            test_A[max_idx] = new_val
            if _has_distinct_subset_sums(test_A):
                best_A = test_A
                best_max = new_val
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Perturb to escape local minimum
        # Strategy: Randomly swap two elements or change one element to a random value
        # We want to create gaps in the subset sums.
        
        # Pick 2 random indices
        i, j = random.sample(range(12), 2)
        
        # Try swapping
        if best_A[i] != best_A[j]:
            test_A = best_A[:]
            test_A[i], test_A[j] = test_A[j], test_A[i]
            # Swap doesn't change the set of values, so subset sums are the same.
            # Wait, swapping elements doesn't change the set of subset sums!
            # The set of values is the same.
            # So swapping is useless for changing validity.
            # We must change the *values*.
            
        # Change one element to a random value
        idx = random.randint(0, 11)
        # Try to pick a value that is not in the set
        # And hopefully creates gaps.
        # Values in [1, best_max]
        
        # To be effective, we might want to try values that are "in the gaps"
        # But we don't know the gaps.
        # Just random.
        
        new_val = random.randint(1, best_max)
        
        if new_val == best_A[idx]:
            continue
        if new_val in best_A:
            continue
            
        test_A = best_A[:]
        test_A[idx] = new_val
        
        if _has_distinct_subset_sums(test_A):
            new_max = max(test_A)
            # Accept if max is not too large
            if new_max <= best_max + 100:
                best_A = test_A
                best_max = new_max
                
    return best_A
