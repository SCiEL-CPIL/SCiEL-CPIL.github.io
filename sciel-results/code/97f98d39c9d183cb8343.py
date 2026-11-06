"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a binary error-correcting code of length 18 and minimum distance 6.
    We use a greedy search with highly optimized distance checking using NumPy
    and a lookup table for population counts.
    """
    import numpy as np
    import time

    N = 18
    MIN_DIST = 6
    TARGET = 512
    DEADLINE = time.time() + 55

    # Lookup table for popcount of 16-bit values
    # We will process 18-bit words by splitting into 16-bit and 2-bit parts
    # Or simply use a 2^16 lookup table and handle the top 2 bits separately
    pop16 = np.zeros(1 << 16, dtype=np.uint8)
    for i in range(1, 1 << 16):
        pop16[i] = pop16[i >> 1] + (i & 1)

    def vectorized_min_distance(words_np, cand):
        """
        Calculate the minimum Hamming distance between cand and all words in words_np.
        words_np: array of uint32
        cand: uint32
        """
        diffs = words_np ^ cand
        # Split into low 16 bits and high 2 bits
        low = diffs & 0xFFFF
        high = (diffs >> 16) & 0x3
        
        # Use lookup table for low 16 bits
        low_counts = pop16[low]
        
        # High 2 bits: 0,1,2,3 -> popcount 0,1,1,2
        # Map: 0->0, 1->1, 2->1, 3->2
        high_counts = np.array([0, 1, 1, 2], dtype=np.uint8)[high]
        
        total_counts = low_counts + high_counts
        return np.min(total_counts)

    # Step 1: Start with RM(1,4) code extended to 18 bits (32 words)
    words = [0]
    for slope in range(16):
        for intercept in range(2):
            if slope == 0 and intercept == 0:
                continue
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Check distance to all existing words
            ok = True
            for w in words:
                if (word ^ w).bit_count() < MIN_DIST:
                    ok = False
                    break
            if ok:
                words.append(word)

    words_np = np.array(words, dtype=np.uint32)
    tried = np.zeros(1 << N, dtype=bool)
    tried[words] = True

    # Step 2: Greedy search
    # We iterate through candidates. To improve efficiency, we can use randomization
    # or structured ordering. Given the time limit, we'll use a simple loop with
    # optimized distance check.
    
    # To speed up, we'll try candidates in a random order to avoid bias
    # and potentially find a larger code faster.
    import random
    random.seed(42)
    
    # Create a list of all candidates
    candidates = list(range(1, 1 << N))
    random.shuffle(candidates)
    
    for cand in candidates:
        if len(words) >= TARGET:
            break
        if time.time() > DEADLINE:
            break
        if tried[cand]:
            continue

        dmin = vectorized_min_distance(words_np, cand)
        
        if dmin >= MIN_DIST:
            words.append(cand)
            words_np = np.append(words_np, cand)
            tried[cand] = True

    return words
# EVOLVE-BLOCK-END
