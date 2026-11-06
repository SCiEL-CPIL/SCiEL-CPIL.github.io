import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    # Start with the Reed-Muller construction (32 words)
    # RM(1,4) has length 16. We extend to 18 by adding 2 zero bits?
    # Or we can just use the 16-bit words and pad with zeros.
    # The distance is preserved.
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Pad to 18 bits (add 2 zeros at the end, i.e., high bits)
            word = word # word is already < 2^16, so it's fine for 18 bits
            words.append(word)
    
    # Convert to numpy array of bits for fast XOR and sum
    # Shape: (M, 18)
    codebook = np.zeros((len(words), N), dtype=np.uint8)
    for i, w in enumerate(words):
        for j in range(N):
            codebook[i, j] = (w >> j) & 1
            
    # List of integers for final return
    word_list = words[:]
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Generate a batch of random candidates
        # We can try to be smart: if we fail to add many, maybe we should try harder?
        # For now, just random.
        batch_size = 5000
        candidates = rng.integers(0, 2**N, size=batch_size)
        
        for cand in candidates:
            if time.time() > deadline:
                break
            
            # Check if cand is already in the set
            # A simple check against the list is O(M). 
            # We can maintain a set of integers for O(1) lookup.
            # But we also need to check distance.
            # If cand is in word_set, skip.
            # We need to maintain a set.
            # Let's add a set for O(1) membership check.
            # I'll add it in the loop below or restructure.
            
            # Let's restructure to maintain a set.
            # I'll do this in the next iteration of the code.
            
            # For now, just check distance. If it's a duplicate, distance is 0, so it will be rejected.
            # So we don't strictly need a set for correctness, just for efficiency of skipping.
            # But checking distance is O(M). Checking set is O(1).
            # If M is large, skipping duplicates is good.
            
            # Convert cand to binary vector
            cand_vec = np.zeros(N, dtype=np.uint8)
            for j in range(N):
                cand_vec[j] = (cand >> j) & 1
                
            # Compute distances
            xors = codebook ^ cand_vec
            distances = np.sum(xors, axis=1)
            min_dist = np.min(distances)
            
            if min_dist >= MIN_DIST:
                # Add to codebook
                new_row = np.vstack([codebook, cand_vec])
                codebook = new_row
                word_list.append(cand)
                
    return word_list
