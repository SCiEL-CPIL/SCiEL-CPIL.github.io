import time
import numpy as np
from math import lcm
from functools import reduce

def construct():
    """
    Find a nonempty subset of [1,22] with maximum cardinality such that
    all reciprocal subset sums (including empty sum 0) are distinct.
    """
    # Deadline guard: 85 seconds to be safe against 90s kill
    deadline = time.time() + 85
    
    # Compute LCM of 1..22 to convert reciprocal sums to integer sums
    L = reduce(lcm, range(1, 23))
    
    # Map each element a to its integer contribution L // a
    vals = {a: L // a for a in range(1, 23)}
    
    # Elements sorted by value descending.
    elements = sorted(range(1, 23), key=lambda a: vals[a], reverse=True)
    
    # Precompute values in this sorted order for faster access
    sorted_vals = [vals[a] for a in elements]

    def is_dissociated(arr, v):
        """
        Check if adding value v to the set of current sums results in
        a set with distinct sums.
        arr: sorted NumPy array of current sums
        v: integer
        Returns True if (arr + v) is disjoint from arr.
        """
        # shifted = arr + v
        # Check intersection
        # Since arr is sorted, arr+v is sorted.
        # np.intersect1d with assume_unique=True is fast.
        shifted = arr + v
        # intersect1d returns the common elements
        # If length is 0, they are disjoint
        return len(np.intersect1d(arr, shifted, assume_unique=True)) == 0

    def find_subset_of_size(target_size):
        """
        Use backtracking to find a subset of exactly target_size elements
        from the sorted elements list that is dissociated.
        Returns the subset (list of original element values) or None.
        """
        # Start with sum 0
        current_sums = np.array([0], dtype=np.int64)
        current_subset = []
        
        def backtrack(idx, current_size, current_sums, current_subset):
            if time.time() > deadline:
                return None
            
            # Pruning 1: Not enough elements left to reach target_size
            if current_size + (len(elements) - idx) < target_size:
                return None
            
            # Base case: We found a valid subset of the required size
            if current_size == target_size:
                return list(current_subset)
            
            # Try to include elements from idx to end
            for i in range(idx, len(elements)):
                if time.time() > deadline:
                    return None
                
                a = elements[i]
                v = sorted_vals[i]
                
                # Check if adding v is valid
                if is_dissociated(current_sums, v):
                    # Calculate new sums
                    # new_sums = sort(union(current_sums, current_sums + v))
                    # Since we know they are disjoint, we can just concatenate and sort
                    new_sums = np.sort(np.concatenate((current_sums, current_sums + v)))
                    
                    # Recurse
                    current_subset.append(a)
                    result = backtrack(i + 1, current_size + 1, new_sums, current_subset)
                    current_subset.pop()
                    
                    if result is not None:
                        return result
            
            return None

        return backtrack(0, 0, current_sums, [])

    # Search for the maximum cardinality
    for size in range(22, 0, -1):
        if time.time() > deadline - 1:
            break
            
        result = find_subset_of_size(size)
        if result is not None:
            return result
        
        # If we didn't find one, continue to smaller sizes
    
    # Fallback: return a single element (always valid)
    return [1]
