import time
import random
import numpy as np

def construct():
    """
    Construct a large binary code of length 17 with minimum distance 6.
    
    Strategy:
    1. Start with the shortened [18,9,6] QR code (size 256).
    2. Use a highly optimized local search with multiple strategies:
       - Random restarts with different initial orderings
       - Greedy addition of valid words
       - Local search to improve code size
    3. Use deadline guard to ensure termination within 90s.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin of 5s before 90s limit
    
    n = 17
    min_dist = 6
    max_val = 1 << n  # 2^17 = 131072
    
    # --- Step 1: Generate the seed code (Size 256) ---
    # The [17,8,6] shortened QR code
    G = [
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    ]
    
    # Convert to integers
    basis = []
    for row in G:
        val = 0
        for i, bit in enumerate(row):
            if bit:
                val |= (1 << i)
        basis.append(val)
        
    # Generate all 2^8 = 256 codewords
    seed_code = []
    for u in range(256):
        c = 0
        for k in range(8):
            if (u >> k) & 1:
                c ^= basis[k]
        seed_code.append(c)
    
    # Verify minimum distance
    min_d = 100
    for i in range(256):
        for j in range(i+1, 256):
            d = (seed_code[i] ^ seed_code[j]).bit_count()
            if d < min_d:
                min_d = d
    if min_d < 6:
        # Fallback: Random Greedy
        seed_code = [0]
        candidates = list(range(1, max_val))
        random.shuffle(candidates)
        for c in candidates:
            if time.time() > deadline:
                break
            valid = True
            for w in seed_code:
                if (c ^ w).bit_count() < min_dist:
                    valid = False
                    break
            if valid:
                seed_code.append(c)
                if len(seed_code) >= 256:
                    break
    
    # Precompute offsets for radius 5 (distance < 6)
    offsets = []
    for i in range(max_val):
        if i.bit_count() <= 5:
            offsets.append(i)
    offsets_arr = np.array(offsets, dtype=np.int32)
    
    def greedy_add(codewords, in_code, block_count, seed_words):
        """Add as many valid words as possible using greedy strategy."""
        for w in seed_words:
            if not in_code[w] and block_count[w] == 0:
                in_code[w] = True
                codewords.append(w)
                block_count[w ^ offsets_arr] += 1
    
    def local_search(codewords, in_code, block_count, max_iters=1000):
        """Local search to improve code size by swapping words."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            
            # Try to add a random valid word
            if len(codewords) == 0:
                break
            
            # Pick a random word to remove
            idx = random.randint(0, len(codewords) - 1)
            w_remove = codewords[idx]
            
            # Remove it
            in_code[w_remove] = False
            block_count[w_remove ^ offsets_arr] -= 1
            codewords.pop(idx)
            
            # Find all valid candidates
            candidates = np.where((~in_code) & (block_count == 0))[0]
            
            if len(candidates) > 1:
                # We can add more than one word, so size increases
                added = 0
                for w in candidates:
                    if time.time() > deadline - 5.0:
                        break
                    if not in_code[w] and block_count[w] == 0:
                        in_code[w] = True
                        codewords.append(w)
                        block_count[w ^ offsets_arr] += 1
                        added += 1
                
                if added > 0:
                    improved = True
            else:
                # Revert
                in_code[w_remove] = True
                block_count[w_remove ^ offsets_arr] += 1
                codewords.insert(idx, w_remove)
        
        return len(codewords)
    
    best_codewords = None
    best_size = 0
    
    # Multiple restarts with different random orderings
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with seed
        in_code = np.zeros(max_val, dtype=bool)
        block_count = np.zeros(max_val, dtype=np.int32)
        codewords = []
        
        # Add seed words
        for w in seed_code:
            in_code[w] = True
            codewords.append(w)
            block_count[w ^ offsets_arr] += 1
        
        # Greedy phase: try to add more words
        # Use a randomized greedy approach
        remaining = list(range(max_val))
        random.shuffle(remaining)
        
        added_count = 0
        for w in remaining:
            if time.time() > deadline - 10.0:
                break
            if not in_code[w] and block_count[w] == 0:
                in_code[w] = True
                codewords.append(w)
                block_count[w ^ offsets_arr] += 1
                added_count += 1
        
        # Local search phase
        size = local_search(codewords, in_code, block_count, max_iters=200)
        
        if size > best_size:
            best_size = size
            best_codewords = codewords[:]
    
    # Final attempt: if we have time, do more local search on the best solution
    if best_codewords and time.time() < deadline - 10.0:
        in_code = np.zeros(max_val, dtype=bool)
        block_count = np.zeros(max_val, dtype=np.int32)
        for w in best_codewords:
            in_code[w] = True
            block_count[w ^ offsets_arr] += 1
        
        size = local_search(best_codewords, in_code, block_count, max_iters=500)
        if size > best_size:
            best_size = size
    
    if best_codewords is None or len(best_codewords) == 0:
        return seed_code
    
    return best_codewords
