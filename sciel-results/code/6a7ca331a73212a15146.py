import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    n = 30
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1), 1)
    
    # Weights in integer domain
    weights = [L // i for i in range(1, n + 1)]
    
    # Split into two halves
    half = n // 2
    weights1 = weights[:half]
    weights2 = weights[half:]
    
    def generate_sums(weights_list):
        sums = np.array([0], dtype=np.int64)
        parents_list = []
        choices_list = []
        
        for w in weights_list:
            # Current size
            curr_size = len(sums)
            # New sums
            # We append -w, 0, +w versions
            # Note: np.concatenate is efficient
            new_sums = np.concatenate([
                sums - w,
                sums,
                sums + w
            ])
            
            # Parents and Choices
            # For each element in new_sums, we need to know which parent it came from
            # and what choice was made.
            # The structure is:
            # [0..N-1] came from [0..N-1] with choice -1
            # [N..2N-1] came from [0..N-1] with choice 0
            # [2N..3N-1] came from [0..N-1] with choice 1
            
            # We need to store this mapping for backtracking.
            # parents: index in previous layer
            # choices: -1, 0, 1
            
            # Create arrays for this step
            # parent indices
            p1 = np.arange(curr_size)
            p2 = np.arange(curr_size)
            p3 = np.arange(curr_size)
            
            c1 = np.full(curr_size, -1, dtype=np.int8)
            c2 = np.full(curr_size, 0, dtype=np.int8)
            c3 = np.full(curr_size, 1, dtype=np.int8)
            
            parents = np.concatenate([p1, p2, p3])
            choices = np.concatenate([c1, c2, c3])
            
            parents_list.append(parents)
            choices_list.append(choices)
            
            sums = new_sums
            
            # Check time
            if time.time() > deadline - 10:
                break
                
        return sums, parents_list, choices_list

    # Generate sums for first half
    sums1, parents1, choices1 = generate_sums(weights1)
    sums1.sort()
    # We lost the parent/choice mapping after sorting!
    # We need to sort sums1 AND keep track of the original indices.
    # Or, we can sort (sum, index) pairs.
    
    # Let's redo generate_sums to return original indices or sort with indices.
    
    def generate_sums_with_idx(weights_list):
        sums = np.array([0], dtype=np.int64)
        idxs = np.array([0], dtype=np.int32) # Original index in the unsorted list? No, we need to backtrack.
        # The backtracking relies on the position in the array at each step.
        # If we sort the final array, we need to map back to the position in the unsorted array.
        
        # Better approach:
        # Generate unsorted sums, parents, choices.
        # Then, create an array of indices `np.arange(len(sums))`.
        # Sort sums using these indices.
        # `sorted_sums = sums[order]`
        # `sorted_idx = order`
        # When we find a best index `best_sorted_idx` in the sorted array,
        # we get `orig_idx = sorted_idx[best_sorted_idx]`.
        # Then we use `orig_idx` to backtrack using the stored parents/choices.
        
        parents_list = []
        choices_list = []
        
        for w in weights_list:
            curr_size = len(sums)
            
            new_sums = np.concatenate([
                sums - w,
                sums,
                sums + w
            ])
            
            p1 = np.arange(curr_size)
            p2 = np.arange(curr_size)
            p3 = np.arange(curr_size)
            
            c1 = np.full(curr_size, -1, dtype=np.int8)
            c2 = np.full(curr_size, 0, dtype=np.int8)
            c3 = np.full(curr_size, 1, dtype=np.int8)
            
            parents = np.concatenate([p1, p2, p3])
            choices = np.concatenate([c1, c2, c3])
            
            parents_list.append(parents)
            choices_list.append(choices)
            
            sums = new_sums
            
        return sums, parents_list, choices_list

    sums1, parents1, choices1 = generate_sums_with_idx(weights1)
    order1 = np.argsort(sums1)
    sorted_sums1 = sums1[order1]
    
    sums2, parents2, choices2 = generate_sums_with_idx(weights2)
    order2 = np.argsort(sums2)
    sorted_sums2 = sums2[order2]
    
    # Two pointer scan
    n1 = len(sorted_sums1)
    n2 = len(sorted_sums2)
    
    i = 0
    j = n2 - 1
    
    best_diff = float('inf')
    best_i_orig = -1
    best_j_orig = -1
    
    # We need to find the minimum absolute sum that is NOT 0.
    # If we find 0, we ignore it.
    
    while i < n1 and j >= 0:
        s1 = sorted_sums1[i]
        s2 = sorted_sums2[j]
        diff = s1 + s2
        
        if diff != 0 and abs(diff) < best_diff:
            best_diff = abs(diff)
            # Map back to original indices
            orig_i = order1[i]
            orig_j = order2[j]
            best_i_orig = orig_i
            best_j_orig = orig_j
            
        if diff > 0:
            j -= 1
        else:
            i += 1
            
        if time.time() > deadline - 5:
            break

    if best_i_orig == -1:
        # Fallback: return a simple solution
        return [0] * 28 + [1, -1]

    # Reconstruct coefficients
    def backtrack(orig_idx, parents_list, choices_list, num_weights):
        coeffs = [0] * num_weights
        idx = orig_idx
        for k in range(num_weights - 1, -1, -1):
            c = choices_list[k][idx]
            coeffs[k] = int(c)
            idx = int(parents_list[k][idx])
        return coeffs

    coeffs1 = backtrack(best_i_orig, parents1, choices1, len(weights1))
    coeffs2 = backtrack(best_j_orig, parents2, choices2, len(weights2))
    
    final_coeffs = coeffs1 + coeffs2
    
    return final_coeffs
