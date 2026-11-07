import time
import numpy as np
import sys

def construct():
    start_time = time.time()
    N = 10_000_000
    
    # Step 1: Compute sigma(n) for n=1..N
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Check deadline
    if time.time() - start_time > 40:
        return [11, 6]

    # Step 2: Group indices by sigma value
    # We only care about n > 1? No, sigma(1)=1. 
    # If sigma(a)=sigma(b)=1, then a=b=1. a!=b required? 
    # "Approximate a ratio... sigma(a)=sigma(b)". 
    # If a=b, ratio is 1. |1 - 1.5| = 0.5.
    # If a!=b, we look for better.
    # The problem doesn't explicitly say a != b, but "ratio a/b" implies distinct usually?
    # "Minimize |a/b - 3/2|". If a=b, error is 0.5.
    # If we find a pair with error < 0.5, we are better.
    # [11, 6] has error 1/3 = 0.333.
    # So we are already better than a=b cases.
    
    indices = np.arange(1, N + 1)
    # Sort indices by their sigma value
    # Use stable sort? Not necessary if we just group them.
    # But we need the indices to be sorted within the group for binary search.
    # argsort on sigma gives order.
    # Within same sigma, the order is arbitrary.
    # So we must sort the indices within each group.
    
    order = np.argsort(sigma[1:N+1])
    sorted_sigmas = sigma[1:N+1][order]
    sorted_indices = indices[order]
    
    # Find group boundaries
    # diff != 0 indicates a change in sigma value
    changes = np.diff(sorted_sigmas) != 0
    boundaries = np.concatenate(([0], np.where(changes)[0] + 1, [N]))
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Iterate over groups
    # boundaries contains start indices of each group and the end (N)
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        size = end - start
        
        if size <= 1:
            continue
            
        # Extract group indices
        group_indices = sorted_indices[start:end]
        
        # Sort the group indices to allow binary search
        # group_indices is a numpy array. np.sort is fast.
        group_indices = np.sort(group_indices)
        
        # We want to find a, b in group_indices such that |a/b - 1.5| is minimized.
        # This is equivalent to minimizing |2a - 3b|.
        # For each b, a should be close to 1.5 * b.
        
        # Vectorized approach for the group:
        # targets = 1.5 * group_indices
        # pos = np.searchsorted(group_indices, targets)
        # Candidates are at pos and pos-1.
        
        targets = 1.5 * group_indices.astype(np.float64)
        pos = np.searchsorted(group_indices, targets, side='left')
        
        # pos contains insertion points.
        # We need to check group_indices[pos] and group_indices[pos-1].
        # Handle boundaries: pos might be 0 or len(group_indices).
        
        # Create candidate arrays
        # For each k, candidate 1 is group_indices[pos[k]] if pos[k] < size
        # candidate 2 is group_indices[pos[k]-1] if pos[k] > 0
        
        # We can do this in a loop for simplicity, as group sizes are likely small.
        # But if a group is large, vectorization is better.
        # Let's check the max group size. If it's small, loop is fine.
        # Most groups are size 1 or 2.
        
        for k in range(size):
            b = int(group_indices[k])
            p = int(pos[k])
            
            # Check index p
            if p < size:
                a = int(group_indices[p])
                if a != b:
                    # Calculate error
                    # err = abs(a/b - 1.5)
                    # To avoid float division issues, use exact rational comparison?
                    # The score is binary64 serialization of the rational.
                    # So float error is what matters.
                    err = abs(a / b - 1.5)
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        
            # Check index p-1
            if p > 0:
                a = int(group_indices[p-1])
                if a != b:
                    err = abs(a / b - 1.5)
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        
        # Check deadline
        if time.time() - start_time > 50:
            break

    return [int(best_a), int(best_b)]
