import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline to be safe
    
    # 1. Generate the initial 729-word code
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Full code: tetracode x Q_3^(n-4)
    # n=8, so n-4=4
    if n != 8:
        # Fallback for other n, though task is specific
        # Simple greedy or direct sum
        pass 

    # Generate all 729 codewords
    codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n-4):
            codewords.append(h + list(tail))
    
    # Map words to integers
    # Total words = 3^8 = 6561
    total_words = q ** n
    # Precompute neighbors for each codeword
    # neighbors[i] is a list of indices (0..6560)
    
    def word_to_idx(w):
        idx = 0
        for k in range(n):
            idx = idx * q + w[k]
        return idx

    # Precompute neighbors for all possible words? No, just for codewords.
    # But we need to update counts.
    # We can precompute the list of neighbor indices for each codeword.
    
    cw_indices = [word_to_idx(cw) for cw in codewords]
    
    neighbors_list = []
    for idx in cw_indices:
        # Convert idx back to word to find neighbors?
        # Or generate neighbors from the word representation.
        w = [0]*n
        temp = idx
        for k in range(n-1, -1, -1):
            w[k] = temp % q
            temp //= q
        # w is the codeword
        
        neigh = [idx]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val: continue
                w[pos] = new_val
                n_idx = word_to_idx(w)
                neigh.append(n_idx)
                w[pos] = orig_val
        neighbors_list.append(neigh)
        
    # Initialize coverage count
    coverage_count = np.zeros(total_words, dtype=np.int32)
    
    # Active codewords
    active = [True] * len(codewords)
    
    # Increment counts for all
    for i in range(len(codewords)):
        for n_idx in neighbors_list[i]:
            coverage_count[n_idx] += 1
            
    best_size = len(codewords)
    
    # Greedy deletion
    # We will iterate passes. In each pass, we try to remove codewords.
    # To avoid bias, we can randomize the order of indices in each pass.
    
    max_passes = 50
    for p in range(max_passes):
        if time.time() > deadline:
            break
            
        indices = list(range(len(codewords)))
        random.shuffle(indices)
        
        any_removed = False
        
        for i in indices:
            if not active[i]:
                continue
                
            # Check if we can remove i
            # Decrement counts
            for n_idx in neighbors_list[i]:
                coverage_count[n_idx] -= 1
                
            # Check min
            if np.min(coverage_count) > 0:
                # Safe to remove
                active[i] = False
                any_removed = True
                best_size -= 1
            else:
                # Restore
                for n_idx in neighbors_list[i]:
                    coverage_count[n_idx] += 1
                
        if not any_removed:
            break
            
    # Construct result
    result = [codewords[i] for i in range(len(codewords)) if active[i]]
    
    return result
