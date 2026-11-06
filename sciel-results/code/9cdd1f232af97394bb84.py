import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit, keep 5s margin

    N = 18
    MIN_DIST = 6
    size = 1 << N

    # 1. GF(2^8) Arithmetic for Quadratic Residue Code Construction
    IRRED = 0b100011101
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= IRRED
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # 2. Find a primitive 17th root of unity
    alpha = None
    for i in range(2, 256):
        if gf_pow(i, 17) == 1:
            if i != 1:
                alpha = i
                break
    
    # Fallback to Reed-Muller if GF construction fails
    if alpha is None:
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # 3. Compute Minimal Polynomial g(x) for the QR code
    poly = [1]
    for j in range(8):
        root = gf_pow(alpha, 2**j)
        new_poly = [0] * (len(poly) + 1)
        for k in range(len(poly)):
            new_poly[k] ^= gf_mul(root, poly[k])
            new_poly[k+1] ^= poly[k]
        poly = new_poly
        
    # 4. Generate Initial Codewords (Linear QR Code + Parity)
    basis = []
    for i in range(9):
        vec = [0] * 17
        for j in range(9):
            vec[i+j] = poly[j]
        basis.append(vec)
        
    codewords = []
    for mask in range(512):
        word = [0] * 17
        for i in range(9):
            if mask & (1 << i):
                for j in range(17):
                    word[j] ^= basis[i][j]
        parity = sum(word) % 2
        val = 0
        for i in range(17):
            if word[i]:
                val |= (1 << i)
        if parity:
            val |= (1 << 17)
        codewords.append(val)
    
    existing_set = set(codewords)
    
    # 5. Precompute Masks for Hamming Ball Marking
    # We need to mark all words within distance < MIN_DIST (i.e., 0 to 5)
    masks = []
    for i in range(size):
        if i.bit_count() < MIN_DIST:
            masks.append(i)
    masks = np.array(masks, dtype=np.uint32)
    
    forbidden = np.zeros(size, dtype=bool)
    
    # Function to mark forbidden neighbors
    def mark_forbidden(c):
        # c is an integer
        # neighbors = (c XOR masks) % size
        # forbidden[neighbors] = True
        # Using vectorized XOR and boolean indexing
        neighbors = (c ^ masks) % size
        forbidden[neighbors] = True

    # Initialize forbidden array with existing codewords
    for c in codewords:
        mark_forbidden(c)
        
    # 6. Greedy Refinement using Forbidden Array
    # Iterate through candidates in weight order (as per current program strategy)
    # This is structurally superior to random shuffling for packing low-weight words early.
    
    # Group candidates by weight to iterate efficiently
    # Instead of storing all lists, we can just iterate 0..size-1 and check weight on the fly?
    # No, precomputing weights or iterating by weight is better for the "weight order" strategy.
    # However, iterating 0..size-1 is O(2^N) which is 262k. Checking bit_count is fast.
    # But we want low weights first.
    
    # Let's create a list of indices sorted by weight.
    # Since size is 2^18, we can do this once.
    indices = np.arange(size, dtype=np.uint32)
    weights = np.array([i.bit_count() for i in range(size)], dtype=np.uint8)
    
    # Sort indices by weight
    sorted_indices = indices[np.argsort(weights)]
    
    # Optimization: We can skip the sort if we just iterate weights 0..18 and pick indices with that weight.
    # But sorting 262k elements is fast enough.
    
    # Alternatively, just iterate 0 to size-1? No, that's not weight order.
    # Let's stick to the weight-ordered approach.
    
    # To avoid huge memory overhead of sorting, we can just loop through weight classes.
    # But generating the list for each weight class is also O(2^N).
    
    # Let's just use the sorted array.
    for c in sorted_indices:
        if time.time() > deadline:
            break
        
        c_int = int(c)
        
        # If this candidate is already in our code, it's already forbidden
        # But we might have added it in the initial QR step.
        # The forbidden array covers it.
        if not forbidden[c_int]:
            codewords.append(c_int)
            mark_forbidden(c_int)
            
    return codewords
