import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 17
    MIN_DIST = 6
    
    # Pre-compute all vectors
    # 2^17 = 131072
    # Shape: (131072, 17)
    # Use uint8 for memory efficiency and speed
    all_indices = np.arange(2**N)
    # Create vectors
    # (i >> j) & 1
    # This can be done with broadcasting
    # i is column, j is row?
    # all_vectors[j, i] = (i >> j) & 1
    # i = all_indices (131072, 1)
    # j = np.arange(N) (1, 17)
    # all_vectors = (all_indices[:, None] >> np.arange(N)[None, :]) & 1
    all_vectors = (all_indices[:, None] >> np.arange(N)[None, :]) & 1
    all_vectors = all_vectors.astype(np.uint8)
    
    # Seed code: 32 words from RM(1, 4) extended
    # The seed in the prompt generates 32 words.
    # We can reuse the logic from the prompt to get the seed ints, then map to indices.
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # The prompt says "extended by two zero bits".
            # The prompt's words are 16-bit? No, "17-bit words".
            # The prompt code:
            # for point in range(16): ... word |= bit << point
            # This creates 16-bit words.
            # But the task says "17-bit words".
            # "The 256-word linear reference is the [18,9,6] ... shortened by one coordinate".
            # The prompt's seed: "First-order Reed-Muller code on 16 coordinates, extended by two zero bits."
            # RM(1,4) has length 16. Extended by 2 zero bits -> length 18?
            # But the task is length 17.
            # Let's look at the prompt's return value.
            # It returns `words`.
            # If I just use the prompt's words, they are 16-bit integers.
            # But the task requires 17-bit codewords.
            # "Each integer encodes one N-bit binary codeword".
            # If the prompt's words are < 2^16, they are valid 17-bit words (leading zero).
            # Distance is preserved if we add leading zeros.
            # So the seed is valid.
            seed_words.append(word)
            
    # Map seed words to indices in all_vectors
    # Since all_vectors covers all 17-bit patterns, the seed words (16-bit) are a subset.
    seed_indices = [w for w in seed_words]
    
    best_size = 0
    best_code = []
    
    # We will run multiple restarts
    while time.time() < deadline:
        # Start with seed
        current_code_indices = list(seed_indices)
        # Convert to array of vectors for fast distance check
        current_vectors = all_vectors[current_code_indices]
        
        # Shuffle all indices
        # Exclude seed indices? No, just shuffle all.
        # If a seed index is picked again, distance is 0, so it won't be added.
        # To save time, we can remove seed indices from the candidate pool.
        candidate_pool = np.setdiff1d(all_indices, current_code_indices)
        np.random.shuffle(candidate_pool)
        
        # Greedy pass
        for idx in candidate_pool:
            if time.time() > deadline - 1: break
            c_vec = all_vectors[idx]
            # Check distance
            # dists = np.sum(current_vectors != c_vec, axis=1)
            # This creates a temporary array.
            # Alternative: loop over current_vectors? No, vectorized is better.
            dists = np.sum(current_vectors != c_vec, axis=1)
            if np.min(dists) >= MIN_DIST:
                current_code_indices.append(idx)
                current_vectors = np.vstack([current_vectors, c_vec])
                
        if len(current_code_indices) > best_size:
            best_size = len(current_code_indices)
            best_code = current_code_indices[:]
            
    return best_code
