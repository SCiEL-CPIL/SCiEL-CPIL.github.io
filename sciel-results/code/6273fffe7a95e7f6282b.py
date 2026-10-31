"""Matrix multiplication 4,4,4 - rank-49 recursive Strassen decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Return rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    
    U: (16, 49) - A elements
    V: (16, 49) - B elements  
    W: (16, 49) - C elements (W[k*n+i, r] convention)
    
    Uses two levels of Strassen: 4x4 -> 7 products of 2x2, each 2x2 -> 7 scalar products.
    Total rank = 49.
    """
    n = 4
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # W index: col * n + row (per problem convention W[k*n+i,r])
    def w_idx(row, col):
        return col * n + row
    
    # Inner Strassen for 2x2 block multiplication
    # a_off, b_off, w_off are the global indices for the 4 elements of each 2x2 block
    # The 2x2 block elements in row-major 4x4 are at offsets [0, 1, 4, 5] from block top-left
    def inner_strassen(a_indices, b_indices, w_indices, rank_start):
        """
        a_indices: [a00, a01, a10, a11] - global U indices for A 2x2 block elements
        b_indices: [b00, b01, b10, b11] - global V indices for B 2x2 block elements
        w_indices: [c00, c01, c10, c11] - global W indices for C 2x2 block elements
        """
        a00, a01, a10, a11 = a_indices
        b00, b01, b10, b11 = b_indices
        c00, c01, c10, c11 = w_indices
        
        r = rank_start
        
        # M1 = (A00+A11)*(B00+B11) -> C00+=1, C11+=1
        U[a00, r] += 1; U[a11, r] += 1
        V[b00, r] += 1; V[b11, r] += 1
        W[c00, r] += 1; W[c11, r] += 1
        r += 1
        
        # M2 = (A10+A11)*B00 -> C10+=1, C11-=1
        U[a10, r] += 1; U[a11, r] += 1
        V[b00, r] += 1
        W[c10, r] += 1; W[c11, r] -= 1
        r += 1
        
        # M3 = A00*(B01-B11) -> C01+=1, C11+=1
        U[a00, r] += 1
        V[b01, r] += 1; V[b11, r] -= 1
        W[c01, r] += 1; W[c11, r] += 1
        r += 1
        
        # M4 = A11*(B10-B00) -> C00+=1, C10+=1
        U[a11, r] += 1
        V[b10, r] += 1; V[b00, r] -= 1
        W[c00, r] += 1; W[c10, r] += 1
        r += 1
        
        # M5 = (A00+A01)*B11 -> C00-=1, C01+=1
        U[a00, r] += 1; U[a01, r] += 1
        V[b11, r] += 1
        W[c00, r] -= 1; W[c01, r] += 1
        r += 1
        
        # M6 = (A10-A00)*(B00+B01) -> C11+=1
        U[a10, r] += 1; U[a00, r] -= 1
        V[b00, r] += 1; V[b01, r] += 1
        W[c11, r] += 1
        r += 1
        
        # M7 = (A01-A11)*(B10+B11) -> C00+=1
        U[a01, r] += 1; U[a11, r] -= 1
        V[b10, r] += 1; V[b11, r] += 1
        W[c00, r] += 1
        r += 1
        
        return r
    
    # Outer Strassen: decompose 4x4 into 7 products of 2x2 blocks
    # Each outer term has:
    #   X = sum of coeff_A * A_block(ar, ac) for A
    #   Y = sum of coeff_B * B_block(br, bc) for B  
    #   C = sum of coeff_C * C_block(cr, cc) for C
    # Then multiply X*Y using inner Strassen
    
    def block_A_indices(br, bc):
        """Global U indices for A 2x2 block at block position (br, bc)."""
        base_r = br * 2
        base_c = bc * 2
        return [base_r*4 + base_c, base_r*4 + base_c+1, 
                (base_r+1)*4 + base_c, (base_r+1)*4 + base_c+1]
    
    def block_B_indices(br, bc):
        """Global V indices for B 2x2 block at block position (br, bc)."""
        base_r = br * 2
        base_c = bc * 2
        return [base_r*4 + base_c, base_r*4 + base_c+1, 
                (base_r+1)*4 + base_c, (base_r+1)*4 + base_c+1]
    
    def block_C_indices(cr, cc):
        """Global W indices for C 2x2 block at block position (cr, cc)."""
        base_r = cr * 2
        base_c = cc * 2
        return [w_idx(base_r, base_c), w_idx(base_r+1, base_c), 
                w_idx(base_r, base_c+1), w_idx(base_r+1, base_c+1)]
    
    # Outer Strassen terms: (X_blocks, Y_blocks, C_targets)
    # X_blocks: list of (block_pos, coeff) for A
    # Y_blocks: list of (block_pos, coeff) for B
    # C_targets: list of (block_pos, coeff) for C
    
    outer_terms = [
        # M1: X=A00+A11, Y=B00+B11, C: C00+=1, C11+=1
        ([((0,0),1), ((1,1),1)],
         [((0,0),1), ((1,1),1)],
         [((0,0),1), ((1,1),1)]),
        # M2: X=A10+A11, Y=B00, C: C10+=1, C11-=1
        ([((1,0),1), ((1,1),1)],
         [((0,0),1)],
         [((1,0),1), ((1,1),-1)]),
        # M3: X=A00, Y=B01-B11, C: C01+=1, C11+=1
        ([((0,0),1)],
         [((0,1),1), ((1,1),-1)],
         [((0,1),1), ((1,1),1)]),
        # M4: X=A11, Y=B10-B00, C: C00+=1, C10+=1
        ([((1,1),1)],
         [((1,0),1), ((0,0),-1)],
         [((0,0),1), ((1,0),1)]),
        # M5: X=A00+A01, Y=B11, C: C00-=1, C01+=1
        ([((0,0),1), ((0,1),1)],
         [((1,1),1)],
         [((0,0),-1), ((0,1),1)]),
        # M6: X=A10-A00, Y=B00+B01, C: C11+=1
        ([((1,0),1), ((0,0),-1)],
         [((0,0),1), ((0,1),1)],
         [((1,1),1)]),
        # M7: X=A01-A11, Y=B10+B11, C: C00+=1
        ([((0,1),1), ((1,1),-1)],
         [((1,0),1), ((1,1),1)],
         [((0,0),1)]),
    ]
    
    rank_offset = 0
    for X_blocks, Y_blocks, C_targets in outer_terms:
        # Build combined index lists for inner Strassen
        # For each position in the 2x2 "virtual" matrix, we need to know
        # which global elements contribute and with what coefficient.
        
        # Inner Strassen positions: 00, 01, 10, 11
        # For position (pr, pc), the contribution is sum of coeff * block_element
        # But we need to map to actual global indices.
        
        # Build a: for each of 4 positions, list of (global_idx, coeff)
        a_contrib = {0: [], 1: [], 2: [], 3: []}  # 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        for (br, bc), coeff in X_blocks:
            idxs = block_A_indices(br, bc)
            a_contrib[0].append((idxs[0], coeff))  # top-left
            a_contrib[1].append((idxs[1], coeff))  # top-right
            a_contrib[2].append((idxs[2], coeff))  # bottom-left
            a_contrib[3].append((idxs[3], coeff))  # bottom-right
        
        b_contrib = {0: [], 1: [], 2: [], 3: []}
        for (br, bc), coeff in Y_blocks:
            idxs = block_B_indices(br, bc)
            b_contrib[0].append((idxs[0], coeff))
            b_contrib[1].append((idxs[1], coeff))
            b_contrib[2].append((idxs[2], coeff))
            b_contrib[3].append((idxs[3], coeff))
        
        c_contrib = {0: [], 1: [], 2: [], 3: []}
        for (cr, cc), coeff in C_targets:
            idxs = block_C_indices(cr, cc)
            c_contrib[0].append((idxs[0], coeff))
            c_contrib[1].append((idxs[1], coeff))
            c_contrib[2].append((idxs[2], coeff))
            c_contrib[3].append((idxs[3], coeff))
        
        # Now apply inner Strassen with these contributions
        # Inner Strassen M1 = (A00+A11)*(B00+B11)
        # A00 contribution: a_contrib[0], A11: a_contrib[3]
        # B00 contribution: b_contrib[0], B11: b_contrib[3]
        # C00: c_contrib[0], C11: c_contrib[3]
        
        r = rank_offset
        
        # M1: (A00+A11)*(B00+B11)
        for (idx, coeff) in a_contrib[0]:
            U[idx, r] += coeff
        for (idx, coeff) in a_contrib[3]:
            U[idx, r] += coeff
        for (idx, coeff) in b_contrib[0]:
            V[idx, r] += coeff
        for (idx, coeff) in b_contrib[3]:
            V[idx, r] += coeff
        for (idx, coeff) in c_contrib[0]:
            W[idx, r] += coeff
        for (idx, coeff) in c_contrib[3]:
            W[idx, r] += coeff
        r += 1
        
        # M2: (A10+A11)*B00
        for (idx, coeff) in a_contrib[2]:
            U[idx, r] += coeff
        for (idx, coeff) in a_contrib[3]:
            U[idx, r] += coeff
        for (idx, coeff) in b_contrib[0]:
            V[idx, r] += coeff
        for (idx, coeff) in c_contrib[2]:
            W[idx, r] += coeff
        for (idx, coeff) in c_contrib[3]:
            W[idx, r] -= coeff
        r += 1
        
        # M3: A00*(B01-B11)
        for (idx, coeff) in a_contrib[0]:
            U[idx, r] += coeff
        for (idx, coeff) in b_contrib[1]:
            V[idx, r] += coeff
        for (idx, coeff) in b_contrib[3]:
            V[idx, r] -= coeff
        for (idx, coeff) in c_contrib[1]:
            W[idx, r] += coeff
        for (idx, coeff) in c_contrib[3]:
            W[idx, r] += coeff
        r += 1
        
        # M4: A11*(B10-B00)
        for (idx, coeff) in a_contrib[3]:
            U[idx, r] += coeff
        for (idx, coeff) in b_contrib[2]:
            V[idx, r] += coeff
        for (idx, coeff) in b_contrib[0]:
            V[idx, r] -= coeff
        for (idx, coeff) in c_contrib[0]:
            W[idx, r] += coeff
        for (idx, coeff) in c_contrib[2]:
            W[idx, r] += coeff
        r += 1
        
        # M5: (A00+A01)*B11
        for (idx, coeff) in a_contrib[0]:
            U[idx, r] += coeff
        for (idx, coeff) in a_contrib[1]:
            U[idx, r] += coeff
        for (idx, coeff) in b_contrib[3]:
            V[idx, r] += coeff
        for (idx, coeff) in c_contrib[0]:
            W[idx, r] -= coeff
        for (idx, coeff) in c_contrib[1]:
            W[idx, r] += coeff
        r += 1
        
        # M6: (A10-A00)*(B00+B01)
        for (idx, coeff) in a_contrib[2]:
            U[idx, r] += coeff
        for (idx, coeff) in a_contrib[0]:
            U[idx, r] -= coeff
        for (idx, coeff) in b_contrib[0]:
            V[idx, r] += coeff
        for (idx, coeff) in b_contrib[1]:
            V[idx, r] += coeff
        for (idx, coeff) in c_contrib[3]:
            W[idx, r] += coeff
        r += 1
        
        # M7: (A01-A11)*(B10+B11)
        for (idx, coeff) in a_contrib[1]:
            U[idx, r] += coeff
        for (idx, coeff) in a_contrib[3]:
            U[idx, r] -= coeff
        for (idx, coeff) in b_contrib[2]:
            V[idx, r] += coeff
        for (idx, coeff) in b_contrib[3]:
            V[idx, r] += coeff
        for (idx, coeff) in c_contrib[0]:
            W[idx, r] += coeff
        r += 1
        
        rank_offset = r
    
    assert rank_offset == 49, f"Expected rank 49, got {rank_offset}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
