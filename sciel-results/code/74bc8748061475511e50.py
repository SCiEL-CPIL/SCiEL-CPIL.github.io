import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140

    terms = []

    # Helper to create a vector of zeros
    def zero_u(): return [0] * (n * m)
    def zero_v(): return [0] * (m * p)
    def zero_w(): return [0] * (p * n)

    # Inner Strassen terms for 2x2 * 2x2
    # Returns list of (u_in, v_in, w_in) where u_in, v_in, w_in are length 4
    def inner_strassen_terms():
        # Indices in 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # M1 = (A11+A22)(B11+B22)
        # u: 1,0,0,1 ; v: 1,0,0,1 ; w: 1,0,0,1
        # M2 = (A21+A22)B11
        # u: 0,0,1,1 ; v: 1,0,0,0 ; w: 0,0,1,-1
        # M3 = A11(B12-B22)
        # u: 1,0,0,0 ; v: 0,1,0,-1 ; w: 0,1,0,1
        # M4 = A22(B21-B11)
        # u: 0,0,0,1 ; v: 0,0,1,-1 ; w: 1,0,1,0
        # M5 = (A11+A12)B22
        # u: 1,1,0,0 ; v: 0,0,0,1 ; w: 0,1,-1,0
        # M6 = (A21-A11)(B11+B12)
        # u: 0,0,1,-1 ; v: 1,1,0,0 ; w: 0,0,0,1
        # M7 = (A12-A22)(B21+B22)
        # u: 0,1,0,-1 ; v: 0,0,1,1 ; w: 1,0,0,0
        
        return [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, -1, 0]),
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]

    inner_terms = inner_strassen_terms()

    # Define the 7 Strassen multiplications for 4x4 * 4x4
    # Each is defined by:
    # U_sources: list of (block_offset, sign) pairs for the first factor
    # V_sources: list of (block_offset, sign) pairs for the second factor
    # W_targets: list of (block_offset, sign) pairs for the result
    
    # Block offsets in 4x4 (0-based row, col of top-left of 2x2 block)
    # A blocks: 0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)
    # B blocks: 0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)
    # C blocks: 0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)
    
    # M1 = (A11+A22)(B11+B22)
    # A11 is block 0, A22 is block 3
    # B11 is block 0, B22 is block 3
    # C11 is block 0 (+), C22 is block 3 (+)
    strassen_defs = [
        # (U_blocks, V_blocks, W_blocks)
        # Each block is (offset, sign)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]), # M2: (A21+A22)B11 -> C21+=M2, C22-=M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]), # M3: A11(B12-B22) -> C12+=M3, C22+=M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]), # M4: A22(B21-B11) -> C11+=M4, C21+=M4
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]), # M5: (A11+A12)B22 -> C12+=M5, C11-=M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]), # M6: (A21-A11)(B11+B12) -> C22+=M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]) # M7: (A12-A22)(B21+B22) -> C11+=M7
    ]

    # Map block offset to global indices
    # Block 0: rows 0-1, cols 0-1. Indices in U (i*m+j): 0,1,4,5
    # Block 1: rows 0-1, cols 2-3. Indices in U: 2,3,6,7
    # Block 2: rows 2-3, cols 0-1. Indices in U: 8,9,12,13
    # Block 3: rows 2-3, cols 2-3. Indices in U: 10,11,14,15
    
    # Inner indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1) within the 2x2 block.
    
    def expand_u(u_in, blocks):
        u = zero_u()
        for block_off, sign in blocks:
            # Determine base index for this block
            # block_off 0 -> base 0
            # block_off 1 -> base 2
            # block_off 2 -> base 8
            # block_off 3 -> base 10
            bases = [0, 2, 8, 10]
            base = bases[block_off]
            # Inner index 0 -> +0, 1 -> +1, 2 -> +4, 3 -> +5
            offsets = [0, 1, 4, 5]
            for k in range(4):
                if u_in[k] != 0:
                    u[base + offsets[k]] += sign * u_in[k]
        return u

    def expand_v(v_in, blocks):
        v = zero_v()
        # V is indexed by (j, k). m=4, p=4 (for this part).
        # Block 0: j=0-1, k=0-1. Indices: 0,1,4,5
        # Block 1: j=0-1, k=2-3. Indices: 2,3,6,7
        # Block 2: j=2-3, k=0-1. Indices: 8,9,12,13
        # Block 3: j=2-3, k=2-3. Indices: 10,11,14,15
        bases = [0, 2, 8, 10]
        offsets = [0, 1, 4, 5]
        for block_off, sign in blocks:
            base = bases[block_off]
            for k in range(4):
                if v_in[k] != 0:
                    v[base + offsets[k]] += sign * v_in[k]
        return v

    def expand_w(w_in, blocks):
        w = zero_w()
        # W is indexed by (k, i). p=4, n=4.
        # Block 0: k=0-1, i=0-1. Indices: 0,1,4,5
        # Block 1: k=0-1, i=2-3. Indices: 2,3,6,7
        # Block 2: k=2-3, i=0-1. Indices: 8,9,12,13
        # Block 3: k=2-3, i=2-3. Indices: 10,11,14,15
        bases = [0, 2, 8, 10]
        offsets = [0, 1, 4, 5]
        for block_off, sign in blocks:
            base = bases[block_off]
            for k in range(4):
                if w_in[k] != 0:
                    w[base + offsets[k]] += sign * w_in[k]
        return w

    # Generate 49 terms for 4x4 * 4x4
    for u_blocks, v_blocks, w_blocks in strassen_defs:
        for u_in, v_in, w_in in inner_terms:
            u = expand_u(u_in, u_blocks)
            v = expand_v(v_in, v_blocks)
            w = expand_w(w_in, w_blocks)
            terms.append((u, v, w))

    # For 4x4 * 4x3 (last 3 columns of B, k=4,5,6), use schoolbook
    # k is 4, 5, 6. i is 0..3, j is 0..3.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = zero_u()
                v = zero_v()
                w = zero_w()
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                terms.append((u, v, w))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
            
    return {"U": U, "V": V, "W": W}
