import numpy as np
import time
import sys

def construct(q, n, radius):
    # For this specific task, q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (not expected)
        # Simple greedy
        pass
        
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # 1. Precomputation
    # Map word (tuple of n ints) to index
    # Index = sum(w[i] * q**(n-1-i))
    # For q=3, n=8, max index = 3^8 - 1 = 6560
    
    # Generate all words
    # We don't need to generate all words explicitly as tuples if we use indices.
    # But we need to generate balls.
    
    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[n-1-i] = idx % q
            idx //= q
        return w

    # Helper to convert word to index
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    # Precompute balls for all indices
    # Ball of idx: idx itself, and idx with one digit changed to other q-1 values
    balls = np.zeros((q**n, 1 + n*(q-1)), dtype=np.int32)
    
    for i in range(q**n):
        w = idx_to_word(i)
        balls[i, 0] = i
        col = 1
        for pos in range(n):
            original = w[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                w[pos] = new_val
                balls[i, col] = word_to_idx(w)
                col += 1
            w[pos] = original
            
    # 2. Greedy Set Cover
    covered = np.zeros(q**n, dtype=bool)
    code_indices = []
    
    # We need to pick candidates.
    # To speed up, we can just iterate.
    # new_covers = np.sum(~covered[balls], axis=1)
    # This creates a temporary array of size 6561*17 ~ 110k bools.
    # Summing reduces to 6561 ints.
    
    while not np.all(covered):
        if time.time() > deadline - 10:
            break
            
        # Calculate how many new words each candidate covers
        # ~covered is bool array. ~covered[balls] is (6561, 17) bool array.
        # sum along axis 1 gives int array of size 6561.
        new_covers = np.sum(~covered[balls], axis=1)
        
        # Pick the best candidate
        best_idx = np.argmax(new_covers)
        
        # If best covers 0, we are stuck (should not happen if covered is not all True)
        if new_covers[best_idx] == 0:
            break
            
        code_indices.append(best_idx)
        
        # Update covered
        covered[balls[best_idx]] = True
        
    # 3. Local Search (Removal)
    # Maintain counts of coverage
    counts = np.zeros(q**n, dtype=np.int32)
    for idx in code_indices:
        counts[balls[idx]] += 1
        
    # Try to remove codewords
    # We do multiple passes
    # To avoid O(N^2) in Python loops, we can try to vectorize the check?
    # Checking if min(counts - balls[idx]) > 0 is hard to vectorize for all idx at once.
    # But we can iterate.
    
    removed = True
    while removed:
        removed = False
        if time.time() > deadline - 5:
            break
            
        # Iterate through current code
        # Randomize order to avoid bias?
        # For simplicity, iterate in reverse or forward.
        # Let's try forward.
        
        for i in range(len(code_indices) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
                
            idx = code_indices[i]
            
            # Tentatively remove
            counts[balls[idx]] -= 1
            
            # Check if any count became 0
            if np.any(counts == 0):
                # Revert
                counts[balls[idx]] += 1
            else:
                # Accept removal
                code_indices.pop(i)
                removed = True

    # Convert indices back to words
    result = []
    for idx in code_indices:
        result.append(idx_to_word(idx))
        
    return result

construct(3, 8, 1)
