import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Construct the 256-word [17, 9, 6] code
    # Use a simpler approach: generate a systematic [17,9,6] code
    # We'll use a known generator matrix for the shortened extended QR code
    
    # For now, let's use a simpler method: start with a small valid code and expand
    # Actually, let's try to generate a valid [17,9,6] code properly
    
    # The [18,9,6] extended quadratic residue code over GF(17) can be constructed
    # but it's complex. Let's use a different approach:
    
    # Start with the [16,5,6] BCH code or similar, then extend
    # Actually, let's just use a greedy approach from a small seed
    
    # Seed: use all-weight-0 and some other words
    # Better: use a known small code and expand
    
    # For now, let's generate a valid code by starting with 0 and greedily adding words
    # But that won't give us 256 words quickly
    
    # Alternative: use the fact that we can construct a [17,9,6] code
    # Let's try a simpler generator: use the first 9 coordinates as information
    
    # Actually, the most reliable approach is to use a precomputed generator matrix
    # or use a known construction
    
    # Let's try: the [17,9,6] code can be obtained from the [18,9,6] extended QR code
    # by shortening. The generator matrix for the [18,9,6] code over GF(2) is:
    
    # For simplicity, let's use a different strategy:
    # 1. Generate a valid [17,9,6] code using a known method
    # 2. Expand it greedily
    
    # Here's a known generator matrix for a [17,9,6] code (one possible):
    # We'll construct it systematically
    
    # Let's use the following approach:
    # The code is linear, so we need a 9x17 generator matrix G such that
    # all nonzero codewords have weight >= 6
    
    # A systematic form: G = [I_9 | P] where P is 9x8
    
    # Let's try to find a valid P by random search
    np.random.seed(42)
    
    def check_code_valid(G):
        """Check if all nonzero codewords have weight >= 6"""
        n = G.shape[1]
        k = G.shape[0]
        for u in range(1, 1 << k):
            bits = [(u >> i) & 1 for i in range(k)]
            c = np.dot(bits, G) % 2
            if c.sum() < 6:
                return False
        return True
    
    # Try random generator matrices
    G = None
    for attempt in range(100):
        P = np.random.randint(0, 2, size=(9, 8))
        G_candidate = np.hstack([np.eye(9, dtype=int), P])
        if check_code_valid(G_candidate):
            G = G_candidate
            break
    
    if G is None:
        # Fallback: use a simpler code
        # Use the [17,4,10] code or similar
        # Actually, let's just use a small valid code
        words = [0]
        # Add some words manually
        for i in range(1, 17):
            w = 1 << i
            # Check distance
            valid = True
            for existing in words:
                if (w ^ existing).bit_count() < 6:
                    valid = False
                    break
            if valid:
                words.append(w)
        return words
    
    # Generate all 256 codewords
    codewords = set()
    for u in range(256):
        bits = [(u >> i) & 1 for i in range(9)]
        c = np.dot(bits, G) % 2
        word = 0
        for j in range(17):
            if c[j]:
                word |= (1 << j)
        codewords.add(word)
    
    if len(codewords) < 256:
        # Error in generation, fallback
        return list(codewords)
    
    # 2. Try to expand the code using Greedy Add
    code_list = list(codewords)
    
    # Convert to NumPy array for vectorized operations
    code_arr = np.array(code_list, dtype=np.uint32)
    
    # Precompute all possible candidates
    all_words = np.arange(2**17, dtype=np.uint32)
    
    # Create a mask of existing codewords
    mask = np.zeros(2**17, dtype=bool)
    mask[code_list] = True
    candidates = all_words[~mask]
    
    # Greedy expansion with vectorized distance checks
    added_count = 0
    batch_size = 1000
    
    for batch_start in range(0, len(candidates), batch_size):
        if time.time() > deadline - 15:
            break
        
        batch = candidates[batch_start:batch_start + batch_size]
        
        # Check each candidate in the batch
        for w in batch:
            if time.time() > deadline - 10:
                break
            
            # Vectorized distance check
            xored = code_arr ^ w
            # Use a lookup table for popcount
            # For 17 bits, we can use:
            dists = np.array([x.bit_count() for x in xored])
            
            if dists.min() >= 6:
                # Add w to code
                code_list.append(w)
                code_arr = np.append(code_arr, w)
                mask[w] = True
                added_count += 1
    
    # 3. Local search: try to improve by swapping codewords
    # This is optional and time-dependent
    
    return code_list
