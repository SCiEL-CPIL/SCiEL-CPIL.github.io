import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for 60s limit
    
    N = 17
    MIN_DIST = 6
    
    # 1. Construct the [17, 9, 6] Quadratic Residue Code
    residues = [0, 1, 2, 4, 8, 9, 13, 15, 16]
    
    # Create the generator row vector
    gen_row = [0] * N
    for r in residues:
        gen_row[r] = 1
    
    # Pre-compute the 9 basis vectors of the code as integers
    g_ints = []
    for i in range(9):
        val = 0
        for j in range(N):
            if gen_row[(j - i) % N] == 1:
                val |= (1 << j)
        g_ints.append(val)
        
    # Generate all 2^9 codewords
    code = []
    for mask in range(2**9):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= g_ints[i]
        code.append(w)
        
    # Optional: Verify min distance of the linear code
    # min_w = N
    # for w in code:
    #    if w == 0: continue
    #    weight = bin(w).count('1')
    #    if weight < min_w: min_w = weight
    # if min_w < MIN_DIST:
    #    # Fallback to greedy if construction failed
    #    pass
    
    random.seed(42)
    
    def is_valid(cand, code, exclude_idx=-1):
        for i, w in enumerate(code):
            if i == exclude_idx:
                continue
            if (cand ^ w).bit_count() < MIN_DIST:
                return False
        return True
        
    # Search loop
    while time.time() < deadline:
        added = False
        # Try to add a new word
        # We try a few hundred candidates
        for _ in range(500):
            cand = random.randint(0, 2**N - 1)
            if is_valid(cand, code):
                code.append(cand)
                added = True
                break
                
        if added:
            continue
            
        # If we couldn't add, try to swap
        # Pick a random index to remove
        idx = random.randint(0, len(code) - 1)
        removed = code[idx]
        
        swapped = False
        # Try to find a replacement
        for _ in range(500):
            cand = random.randint(0, 2**N - 1)
            if cand == removed:
                continue
            if is_valid(cand, code, exclude_idx=idx):
                code[idx] = cand
                swapped = True
                break
        
        if not swapped:
            # Restore the removed word
            code[idx] = removed
            # If we fail to swap, we might be in a very tight local optimum.
            # We can try to remove TWO words and add ONE? Or just continue.
            # For now, just continue.
            
    return code
