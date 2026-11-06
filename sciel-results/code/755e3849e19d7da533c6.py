"""Construct a large binary code of length 18 with minimum distance 6."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 18
MIN_DIST = 6
DEADLINE_MARGIN = 5.0  # seconds to leave before the 90s kill

def _reed_muller_extended():
    """First-order Reed-Muller code on 16 coords, extended by 2 zero bits. 32 words, dist 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Extended with 2 zero bits (bits 16, 17 are 0)
            words.append(word)
    return words

def construct():
    deadline = time.time() + (90.0 - DEADLINE_MARGIN)
    rng = np.random.default_rng(42)
    
    # Start with Reed-Muller seed
    seed_words = _reed_muller_extended()
    code_arr = np.array(seed_words, dtype=np.uint32)
    code_set = set(seed_words)
    
    # Batch greedy construction
    # We generate batches of random words, filter them using vectorized distance checks,
    # and add valid ones to the code.
    
    while time.time() < deadline:
        # Adjust batch size based on code size to manage memory and time
        c = len(code_arr)
        if c < 50:
            batch_size = 5000
        elif c < 200:
            batch_size = 2000
        elif c < 500:
            batch_size = 500
        else:
            batch_size = 100
            
        # Generate random candidates
        # Using integers in [0, 2**N)
        candidates = rng.integers(0, 2**N, size=batch_size, dtype=np.uint32)
        
        # Vectorized distance check
        if c == 0:
            valid_mask = np.ones(batch_size, dtype=bool)
        else:
            # (B, 1) ^ (1, C) -> (B, C)
            xors = candidates[:, None] ^ code_arr[None, :]
            # np.bit_count works on arrays
            dists = np.bit_count(xors)
            min_dists = np.min(dists, axis=1)
            valid_mask = min_dists >= MIN_DIST
        
        valid_candidates = candidates[valid_mask]
        
        new_words = []
        for c_val in valid_candidates:
            if c_val not in code_set:
                code_set.add(c_val)
                new_words.append(c_val)
                
        if new_words:
            code_arr = np.append(code_arr, np.array(new_words, dtype=np.uint32))
            
    # Convert back to list of ints for return
    return [int(x) for x in code_arr]
# EVOLVE-BLOCK-END
