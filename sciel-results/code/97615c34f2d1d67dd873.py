"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    DEADLINE = 55  # Leave 5s margin for 60s limit
    
    # 1. Generate the 32 Reed-Muller RM(1,4) words
    # These are affine functions on F_2^4, embedded in 17 bits (append 000)
    rm_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Extend to 17 bits by adding 3 zero bits (positions 16, 15, 14? No, just append 0s)
            # The task says "including leading zeros". 
            # The RM code is on 16 coordinates. We extend to 17.
            # Usually, extension adds a parity bit or a zero.
            # If we append 0s, the distance between any two RM words remains 8 (or 16 if complement).
            # Min distance is 8. This is valid for MIN_DIST=6.
            rm_words.append(word)
            
    # 2. Generate all 2^17 words
    all_words = np.arange(2**N, dtype=np.uint32)
    
    # Convert to bit matrix for fast Hamming distance calculation
    # Shape: (131072, 17)
    # We need to be careful with memory and speed.
    # 131072 * 17 * 4 bytes (uint32) = ~8.9 MB. Acceptable.
    
    def to_bit_matrix(words_array):
        # words_array is 1D array of uint32
        # Returns 2D array of uint8 (or bool)
        bits = np.zeros((len(words_array), N), dtype=np.uint8)
        for i in range(N):
            bits[:, i] = (words_array >> i) & 1
        return bits

    all_bits = to_bit_matrix(all_words)
    
    # Initialize valid mask
    valid_mask = np.ones(len(all_words), dtype=bool)
    
    # Current code set
    C = list(rm_words)
    
    # Pre-compute bits for current codewords?
    # We will update valid_mask by checking distance to each new codeword.
    
    # First, invalidate candidates that are too close to the initial RM words
    for w in C:
        w_bits = np.array([(w >> i) & 1 for i in range(N)], dtype=np.uint8)
        # Compute distances
        # XOR and sum
        diffs = all_bits ^ w_bits
        dists = np.sum(diffs, axis=1)
        valid_mask &= (dists >= MIN_DIST)
        
    # Greedy expansion
    # Iterate through all words. 
    # To improve performance, we can skip invalid ones quickly.
    # We process in index order.
    
    for idx in range(len(all_words)):
        if time.time() - start_time > DEADLINE:
            break
            
        if not valid_mask[idx]:
            continue
            
        # Add this word to the code
        C.append(int(all_words[idx]))
        
        # Update valid_mask for all subsequent candidates
        # We only need to check candidates j > idx that are currently valid
        # But it's easier to just compute for all valid_mask and mask out idx+1 onwards?
        # Actually, we can just compute for all, and the mask will handle it.
        # However, computing for all 131k words every time is 256 * 131k = 33M ops.
        # This is fast in NumPy.
        
        w = all_words[idx]
        w_bits = np.array([(w >> i) & 1 for i in range(N)], dtype=np.uint8)
        diffs = all_bits ^ w_bits
        dists = np.sum(diffs, axis=1)
        
        # Only invalidate if distance < MIN_DIST
        # And only for indices > idx (since we process in order, previous ones are already fixed)
        # Actually, if we add w, it conflicts with w itself (dist 0).
        # We need to ensure we don't add the same word again.
        # valid_mask[idx] becomes False because dist(w,w)=0 < 6.
        
        new_valid = (dists >= MIN_DIST)
        # We must preserve the validity of indices < idx?
        # No, indices < idx have already been processed.
        # If they were valid, they were added to C.
        # If they were invalid, they are not in C.
        # The state of valid_mask for j < idx doesn't matter for future additions,
        # except that we shouldn't re-add them.
        # But we iterate idx from 0 to N, so we won't visit j < idx again.
        # So we can just update the whole mask.
        
        valid_mask &= new_valid
        
    return C
# EVOLVE-BLOCK-END
