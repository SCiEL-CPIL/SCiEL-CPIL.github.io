import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 17
    MIN_DIST = 6
    TOTAL_WORDS = 1 << N
    
    # Pre-compute all bit vectors
    # Shape: (2**17, 17)
    # Using uint8
    all_bits = np.array([np.binary_repr(i, N) for i in range(TOTAL_WORDS)], dtype=np.uint8)
    all_bits = all_bits - 48  # '0' -> 0, '1' -> 1
    
    # Generate the 32-word Reed-Muller core
    def generate_core():
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    core_ints = generate_core()
    core_indices = np.array(core_ints, dtype=int)
    core_array = all_bits[core_indices].copy() # Shape (32, 17)
    
    best_size = len(core_array)
    best_array = core_array.copy()
    
    rng = np.random.default_rng(42)
    
    # Pre-allocate workspace
    workspace = np.empty((TOTAL_WORDS, N), dtype=np.uint8)
    
    while time.time() < deadline - 2.0:
        # Shuffle all indices
        shuffled_indices = rng.permutation(TOTAL_WORDS)
        
        # Start with the core
        workspace[:32] = core_array
        current_size = 32
        
        # Greedy pass
        for idx in shuffled_indices:
            if time.time() > deadline - 1.0:
                break
            
            v = all_bits[idx]
            
            # Check against current words
            # current words are workspace[:current_size]
            if current_size == 0:
                workspace[0] = v
                current_size = 1
                continue
            
            # Compute distances
            # diffs shape (current_size, 17)
            diffs = workspace[:current_size] ^ v
            dists = diffs.sum(axis=1)
            min_dist = dists.min()
            
            if min_dist >= MIN_DIST:
                if min_dist == 0:
                    continue
                workspace[current_size] = v
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_array = workspace[:current_size].copy()
            
    # Convert best_array back to integers
    powers = (1 << np.arange(N, -1, -1, dtype=np.int64))
    best_ints = (best_array * powers).sum(axis=1)
    
    return [int(x) for x in best_ints]
