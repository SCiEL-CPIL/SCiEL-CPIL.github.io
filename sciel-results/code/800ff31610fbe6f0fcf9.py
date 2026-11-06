<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 18
    MIN_DIST = 6
    deadline = time.time() + 50.0  # Safety margin before 60s limit
    
    # Start with the Reed-Muller construction (32 words)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    word_set = set(words)
    word_list = words
    
    # Greedy random addition: try to add random codewords that maintain min distance
    # Use numpy for vectorized distance checking when possible
    
    rng = np.random.default_rng(42)
    added = 0
    
    while time.time() < deadline:
        # Generate a batch of random candidates
        batch_size = 1000
        candidates = rng.integers(0, 2**N, size=batch_size)
        
        for cand in candidates:
            if time.time() > deadline:
                break
            if cand in word_set:
                continue
            
            # Check distance against all existing words
            valid = True
            for w in word_list:
                if (cand ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                word_list.append(cand)
                word_set.add(cand)
                added += 1
    
    return word_list
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 18
    MIN_DIST = 6
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    # Start with the Reed-Muller construction (32 words)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    word_set = set(words)
    word_list = list(words)
    
    # Convert to numpy array for vectorized distance computation
    # Represent each word as a numpy array of bits
    def words_to_bits_array(ws):
        arr = np.zeros((len(ws), N), dtype=np.uint8)
        for i, w in enumerate(ws):
            for j in range(N):
                arr[i, j] = (w >> j) & 1
        return arr
    
    bits_array = words_to_bits_array(word_list)
    
    rng = np.random.default_rng(42)
    
    def min_distance_to_existing(cand_bits, existing_bits):
        """Compute minimum Hamming distance from cand_bits to all rows in existing_bits."""
        # XOR in bit space: distance = popcount(xor)
        # For binary arrays, hamming distance = sum of absolute differences
        diff = np.abs(cand_bits[None, :] - existing_bits)  # shape (M, N)
        distances = diff.sum(axis=1)  # shape (M,)
        return distances.min()
    
    # Greedy random addition with vectorized distance checking
    added = 0
    batch_size = 5000
    
    while time.time() < deadline:
        # Generate a batch of random candidates
        candidates = rng.integers(0, 2**N, size=batch_size)
        
        # Convert candidates to bits array
        cand_bits = np.zeros((batch_size, N), dtype=np.uint8)
        for i in range(batch_size):
            w = candidates[i]
            for j in range(N):
                cand_bits[i, j] = (w >> j) & 1
        
        # For each candidate, check if it's valid
        # We can vectorize this: for each candidate, compute min distance to all existing
        # But that's batch_size * len(word_list) * N operations, which could be large
        # Let's do it in chunks to manage memory
        
        chunk_size = 500
        for start in range(0, batch_size, chunk_size):
            if time.time() > deadline:
                break
            
            end = min(start + chunk_size, batch_size)
            chunk_cands = candidates[start:end]
            chunk_bits = cand_bits[start:end]
            
            # Filter out duplicates first
            valid_indices = []
            for i in range(end - start):
                if chunk_cands[i] not in word_set:
                    valid_indices.append(i)
            
            if not valid_indices:
                continue
            
            valid_cands = chunk_cands[valid_indices]
            valid_bits = chunk_bits[valid_indices]
            
            # Check each valid candidate against all existing words
            # Use vectorized approach: for each candidate, compute distances to all existing
            for vi in range(len(valid_cands)):
                if time.time() > deadline:
                    break
                
                cand = valid_cands[vi]
                cb = valid_bits[vi]
                
                # Compute distances to all existing words
                diff = np.abs(cb[None, :] - bits_array)
                distances = diff.sum(axis=1)
                
                if distances.min() >= MIN_DIST:
                    word_list.append(cand)
                    word_set.add(cand)
                    added += 1
                    # Append to bits_array
                    bits_array = np.vstack([bits_array, cb[None, :]])
    
    return word_list
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, the above still has a Python loop over candidates. Let me think of a better approach. The key insight is that for each candidate, I need to check if it's at distance >= 6 from ALL existing words. I can vectorize this by computing all pairwise distances at once for a batch of candidates against all existing words.

Let me redo this more carefully with proper vectorization:

