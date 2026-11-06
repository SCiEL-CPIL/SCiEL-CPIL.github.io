import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for 60s limit
    
    N = 17
    MIN_DIST = 6
    
    # 1. Construct the [17, 9, 6] Quadratic Residue Code (256 words)
    residues = [0, 1, 2, 4, 8, 9, 13, 15, 16]
    
    gen_row = [0] * N
    for r in residues:
        gen_row[r] = 1
    
    # Pre-convert generator rows to integers
    g_ints = []
    for i in range(9):
        val = 0
        for j in range(N):
            if gen_row[(j - i) % N]:
                val |= (1 << j)
        g_ints.append(val)
    
    # Generate all 2^9 codewords
    words = []
    for mask in range(2**9):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= g_ints[i]
        words.append(w)
    
    # Verify minimum distance for linear code
    min_w = N
    for w in words:
        if w == 0:
            continue
        weight = bin(w).count('1')
        if weight < min_w:
            min_w = weight
        if min_w < MIN_DIST:
            break
    
    if min_w < MIN_DIST:
        # Fallback to Reed-Muller based greedy
        rm_words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                rm_words.append(word)
        code = list(rm_words)
        random.seed(42)
        while time.time() < deadline:
            added = False
            for _ in range(500):
                cand = random.randint(0, 2**N - 1)
                valid = all((cand ^ w).bit_count() >= MIN_DIST for w in code)
                if valid:
                    code.append(cand)
                    added = True
                    break
            if not added:
                break
        return code
    
    # We have a valid 256-word linear code. Try to extend it nonlinearly.
    code = list(words)
    
    # Convert to numpy array for vectorized distance checks
    code_arr = np.array(code, dtype=np.int32)
    
    # Precompute bit masks
    bit_masks = np.array([1 << i for i in range(N)], dtype=np.int32)
    
    random.seed(12345)
    
    # Track how many times we've failed to add a word
    fail_count = 0
    max_fails = 20000
    
    while time.time() < deadline and fail_count < max_fails:
        added = False
        
        # Generate a batch of random candidates
        batch_size = 10000
        candidates = np.random.randint(0, 2**N, size=batch_size)
        
        # Compute distances to all codewords using XOR and popcount
        # For each candidate, compute distance to each codeword
        # We can do this by broadcasting: candidates[:, None] ^ code_arr[None, :]
        # But that's 10000 x 256+ = too large. Instead, check in chunks.
        
        chunk_size = 2000
        for start in range(0, batch_size, chunk_size):
            if time.time() > deadline:
                break
            
            chunk = candidates[start:start+chunk_size]
            # Compute XOR with all codewords
            # chunk shape: (chunk_len,), code_arr shape: (code_size,)
            # We want (chunk_len, code_size)
            xor_matrix = chunk[:, None] ^ code_arr[None, :]
            # Compute popcount: sum of bits set
            # Use bit operations: for each bit position, check if set
            # popcount = sum over bits of (xor >> bit) & 1
            # This can be done with numpy:
            popcounts = np.zeros(xor_matrix.shape, dtype=np.int32)
            for b in range(N):
                popcounts += ((xor_matrix >> b) & 1).astype(np.int32)
            
            # Find candidates where all distances >= MIN_DIST
            min_dists = popcounts.min(axis=1)
            valid_indices = np.where(min_dists >= MIN_DIST)[0]
            
            if len(valid_indices) > 0:
                for idx in valid_indices:
                    cand = int(chunk[idx])
                    code.append(cand)
                    code_arr = np.append(code_arr, cand)
                    added = True
                
                if added:
                    fail_count = 0
                    break
        
        if not added:
            fail_count += 1
            
            # If we've been failing for a while, try local search:
            # Take existing words, flip bits, check
            if fail_count % 5 == 0:
                # Generate candidates by flipping 1-4 bits of random existing words
                local_candidates = []
                for _ in range(500):
                    base_idx = random.randint(0, len(code) - 1)
                    base = code[base_idx]
                    num_flips = random.randint(1, 4)
                    flips = random.sample(range(N), num_flips)
                    cand = base
                    for pos in flips:
                        cand ^= (1 << pos)
                    local_candidates.append(cand)
                
                if local_candidates:
                    local_arr = np.array(local_candidates, dtype=np.int32)
                    xor_matrix = local_arr[:, None] ^ code_arr[None, :]
                    popcounts = np.zeros(xor_matrix.shape, dtype=np.int32)
                    for b in range(N):
                        popcounts += ((xor_matrix >> b) & 1).astype(np.int32)
                    
                    min_dists = popcounts.min(axis=1)
                    valid_indices = np.where(min_dists >= MIN_DIST)[0]
                    
                    if len(valid_indices) > 0:
                        for idx in valid_indices:
                            cand = int(local_arr[idx])
                            code.append(cand)
                            code_arr = np.append(code_arr, cand)
                            added = True
                        
                        if added:
                            fail_count = 0
        
        # If we can't add any more words after many attempts, break
        if fail_count >= max_fails:
            break
    
    return code
