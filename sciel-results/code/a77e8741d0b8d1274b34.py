import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 57.0  # Safety margin for 60s limit
    
    # 1. Seed with the known 32-word RM(1,4) code (extended to 17 bits with a 0)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    # Convert to numpy array for vectorized operations
    words_arr = np.array(words, dtype=np.uint32)
    
    # Helper to check if a candidate word is valid (distance >= 6 from all existing)
    def is_valid(candidate):
        # Compute XOR with all existing words
        diffs = words_arr ^ candidate
        # Count bits in each diff
        # Use lookup or bit manipulation
        # For 17 bits, we can use a simple approach
        counts = np.zeros(len(diffs), dtype=np.uint8)
        for i in range(17):
            counts += (diffs >> i) & 1
        return np.all(counts >= 6)
    
    # Faster: precompute popcount for all 2^17 values? Too large.
    # Alternative: use Python's int.bit_count() in a loop but vectorize where possible
    
    # Let's use a hybrid: keep words as list of ints, use vectorized numpy for batch checks
    def is_valid_fast(candidate):
        # Convert to numpy uint32 array
        c = np.uint32(candidate)
        diffs = words_arr ^ c
        # Count set bits using a lookup table approach or bit tricks
        # For 17 bits, we can use:
        # popcount = sum of bits
        # Use numpy's unpackbits or a lookup
        # Simple approach: iterate over bits
        total = np.zeros(len(diffs), dtype=np.uint8)
        for b in range(17):
            total += ((diffs >> np.uint32(b)) & np.uint32(1)).astype(np.uint8)
        return bool(np.all(total >= 6))
    
    # Even faster: precompute a popcount lookup for 17-bit values
    # 2^17 = 131072, manageable
    popcount_lut = np.zeros(131072, dtype=np.uint8)
    for i in range(131072):
        popcount_lut[i] = bin(i).count('1')
    
    def is_valid_lut(candidate):
        c = np.uint32(candidate)
        diffs = (words_arr ^ c).astype(np.uint32)
        # Index into LUT
        counts = popcount_lut[diffs]
        return bool(np.all(counts >= 6))
    
    random.seed(42)
    
    # Phase 1: Greedy random addition
    # Try random candidates, add if valid
    attempts = 0
    while time.time() < deadline - 15.0:
        candidate = random.getrandbits(17)
        if is_valid_lut(candidate):
            words.append(candidate)
            words_arr = np.array(words, dtype=np.uint32)
        attempts += 1
        if attempts % 100000 == 0 and time.time() > start_time + 5:
            # Check if we're making progress
            if attempts > 500000 and len(words) < 40:
                break
    
    # Phase 2: Local search - try flipping 2-3 bits of existing words
    words_arr = np.array(words, dtype=np.uint32)
    n_words = len(words)
    added_in_phase2 = 0
    
    # Precompute all pairs of bit positions for 2-bit flips
    bit_pairs = [(i, j) for i in range(17) for j in range(i+1, 17)]
    
    while time.time() < deadline - 5.0 and added_in_phase2 < 200:
        # Pick a random word and try 2-bit or 3-bit flips
        base_idx = random.randint(0, n_words - 1)
        base = words[base_idx]
        
        # Try 2-bit flips
        i, j = random.choice(bit_pairs)
        candidate = base ^ (1 << i) ^ (1 << j)
        if is_valid_lut(candidate):
            words.append(candidate)
            words_arr = np.array(words, dtype=np.uint32)
            n_words += 1
            added_in_phase2 += 1
            continue
        
        # Try 3-bit flips
        i, j, k = random.sample(range(17), 3)
        candidate = base ^ (1 << i) ^ (1 << j) ^ (1 << k)
        if is_valid_lut(candidate):
            words.append(candidate)
            words_arr = np.array(words, dtype=np.uint32)
            n_words += 1
            added_in_phase2 += 1
    
    # Phase 3: Try to add more with random candidates (might find gaps)
    words_arr = np.array(words, dtype=np.uint32)
    while time.time() < deadline - 2.0:
        candidate = random.getrandbits(17)
        if is_valid_lut(candidate):
            words.append(candidate)
            words_arr = np.array(words, dtype=np.uint32)
    
    return words
