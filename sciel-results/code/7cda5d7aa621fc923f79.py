"""Optimized subset-sum construction via vectorized local search."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

N = 12
MAX_VAL = 1048576

# Precompute masks for vectorized subset sum calculation
# We represent the array of 2^N sums as a 2D array of shape (N, 2^(N-1))
# Or simpler: use the property that subset sums can be computed via broadcasting
# Let's use a standard iterative vectorized approach or precomputed indices.

# Precompute the bits for each element
# For element i, it is included in masks where bit i is set.
# We can compute all subset sums at once if we have the array.
# sums = np.zeros(1 << N)
# for i in range(N):
#     bit = 1 << i
#     for mask in range(bit):
#         sums[bit | mask] = sums[mask] + arr[i]
# This loop is too slow.

# Vectorized method:
# Let A be the array of N elements.
# We want to compute all 2^N subset sums.
# We can view the subset sum as the dot product of the mask vector and A.
# Masks are 0..2^N-1.
# Create a matrix M of shape (2^N, N) where M[k, i] is the i-th bit of k.
# Then sums = M @ A.
# This is fast in NumPy.

# Precompute M
M = np.array([(i >> j) & 1 for i in range(1 << N) for j in range(N)], dtype=np.int64).reshape(1 << N, N)

def _score(arr):
    """
    Returns 0 if all subset sums are distinct, else count of collisions.
    Uses NumPy for speed.
    """
    # arr is a list or numpy array of length N
    # Ensure it's int64 for precision
    a = np.array(arr, dtype=np.int64)
    sums = M @ a
    # Count unique sums
    # Using set is fast for 4096 elements
    return len(sums) - len(set(sums.tolist()))

def construct():
    deadline = time.time() + 50  # Safety margin
    
    # Initial valid set: Powers of 2
    best_arr = [1 << i for i in range(N)]
    best_max = max(best_arr)
    
    random.seed(42)
    np.random.seed(42)
    
    # Strategy: Local search to minimize max element
    # Start from powers of 2
    current = best_arr[:]
    
    # We will perform many iterations.
    # Move types:
    # 1. Decrease a random element by a random amount.
    # 2. Swap two elements.
    # 3. Randomly perturb an element.
    
    # To make it effective, we should try to reduce the largest element first,
    # or any element, but check validity.
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Determine move
        move_type = random.random()
        
        if move_type < 0.7:
            # Try to decrease a random element
            idx = random.randint(0, N - 1)
            old_val = current[idx]
            if old_val <= 1:
                continue
            
            # Try a random decrease
            # Bias towards larger decreases for larger numbers?
            # Or just random step.
            max_dec = min(old_val - 1, 100)
            dec = random.randint(1, max_dec)
            new_val = old_val - dec
            
            current[idx] = new_val
            score = _score(current)
            
            if score == 0:
                # Valid
                new_max = max(current)
                if new_max < best_max:
                    best_max = new_max
                    best_arr = current[:]
            else:
                current[idx] = old_val
                
        elif move_type < 0.9:
            # Try to swap two elements
            i, j = random.sample(range(N), 2)
            current[i], current[j] = current[j], current[i]
            score = _score(current)
            if score == 0:
                new_max = max(current)
                if new_max < best_max:
                    best_max = new_max
                    best_arr = current[:]
            else:
                current[i], current[j] = current[j], current[i]
        else:
            # Random perturbation (can increase or decrease)
            idx = random.randint(0, N - 1)
            old_val = current[idx]
            delta = random.randint(-10, 10)
            new_val = max(1, old_val + delta)
            if new_val == old_val:
                continue
            
            current[idx] = new_val
            score = _score(current)
            if score == 0:
                new_max = max(current)
                if new_max < best_max:
                    best_max = new_max
                    best_arr = current[:]
            else:
                current[idx] = old_val

        # Periodically restart from best found to escape local minima?
        # Or just keep going.
        if iterations % 10000 == 0:
            # Check if we are stuck at a local minimum
            # If so, maybe try a larger jump
            pass

    # Final verification
    if _score(best_arr) != 0:
        # Fallback to powers of 2
        best_arr = [1 << i for i in range(N)]
        
    assert len(best_arr) == N
    assert all(isinstance(x, int) and x > 0 for x in best_arr)
    assert all(x <= MAX_VAL for x in best_arr)
    assert len(set(best_arr)) == N
    assert _score(best_arr) == 0
    
    return best_arr
# EVOLVE-BLOCK-END
