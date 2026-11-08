"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We'll build the decomposition by:
    # 1. Decomposing the 4x4x4 part (first 4 columns of output) using recursive Strassen (rank 49)
    # 2. Adding 4 terms for the 5th column
    # Total rank should be 53
    
    # First, let's implement Strassen's algorithm for 2x2 matrices
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # For 4x4, we partition into 2x2 blocks and apply Strassen recursively
    # This gives 7^2 = 49 products
    
    # Let me define the Strassen decomposition for 2x2 as a set of (U, V, W) entries
    # Then recursively apply to get 4x4
    
    # Actually, let me just hardcode the 49-term decomposition for 4x4x4
    # This is complex, so let me use a different approach
    
    # Alternative: use the schoolbook for 4x4x4 (rank 16) and add 4 for 5th column = 20
    # But that's worse than 65
    
    # Let me try to implement recursive Strassen properly
    
    # For a 2x2 matrix multiplication C = A*B where A, B, C are 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    # Strassen's 7 products:
    # p1 = (a+d)*(e+h)
    # p2 = (c+d)*e
    # p3 = a*(f-h)
    # p4 = d*(g-e)
    # p5 = (a+b)*h
    # p6 = (c-a)*(e+f)
    # p7 = (b-d)*(g+h)
    # Then:
    # i = p1 + p4 - p5 + p7
    # j = p3 + p5
    # k = p2 + p4
    # l = p1 - p2 + p3 + p6
    
    # For 4x4, we partition A, B, C into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # Similarly for B and C
    # Then C00 = A00*B00 + A01*B10, etc.
    # Apply Strassen at the block level, giving 7 block products
    # Each block product is a 2x2 matrix multiplication, which we decompose using Strassen again
    # Total: 7 * 7 = 49 scalar products
    
    # Let me implement this step by step
    
    # First, define the 2x2 Strassen decomposition as a list of (u_indices, v_indices, w_indices, coefficients)
    # For 2x2, the indices are:
    # A: 0,1,2,3 (row-major: A00=0, A01=1, A10=2, A11=3)
    # B: 0,1,2,3
    # C: 0,1,2,3
    
    # The 7 products and their contributions to C:
    strassen_2x2 = [
        # (u_entries, v_entries, w_entries) where each is a dict from index to coefficient
        # p1 = (A00+A11)*(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # contributes to C00, C11
        # Wait, this is not quite right. Let me be more careful.
        
        # Actually, for tensor decomposition, each term is a rank-1 tensor.
        # For Strassen, each "product" is a bilinear form, but the final C is a linear combination.
        # So I need to express each element of C as a sum of products.
        
        # Let me list the 7 products and which C elements they contribute to:
        # p1 = (A00+A11)*(B00+B11) -> contributes to C00 (+1), C11 (+1)
        # p2 = (A10+A11)*B00 -> contributes to C10 (+1)
        # p3 = A00*(B01-B11) -> contributes to C01 (+1), C11 (-1)
        # p4 = A11*(B10-B00) -> contributes to C00 (+1), C10 (-1)
        # p5 = (A00+A01)*B11 -> contributes to C00 (-1), C01 (+1)
        # p6 = (A10-A00)*(B00+B01) -> contributes to C11 (+1)
        # p7 = (A01-A11)*(B10+B11) -> contributes to C01 (+1), C10 (+1)
        
        # So the 7 rank-1 terms (after expanding) are:
        # Term 1: (A00+A11)*(B00+B11) -> U: {0:1, 3:1}, V: {0:1, 3:1}, W: {0:1, 3:1}
        # Term 2: (A10+A11)*B00 -> U: {2:1, 3:1}, V: {0:1}, W: {2:1}
        # Term 3: A00*(B01-B11) -> U: {0:1}, V: {1:1, 3:-1}, W: {1:1, 3:-1}
        # Term 4: A11*(B10-B00) -> U: {3:1}, V: {2:1, 0:-1}, W: {0:1, 2:-1}
        # Term 5: (A00+A01)*B11 -> U: {0:1, 1:1}, V: {3:1}, W: {0:-1, 1:1}
        # Term 6: (A10-A00)*(B00+B01) -> U: {2:1, 0:-1}, V: {0:1, 1:1}, W: {3:1}
        # Term 7: (A01-A11)*(B10+B11) -> U: {1:1, 3:-1}, V: {2:1, 3:1}, W: {1:1, 2:1}
    ]
    
    # Now for 4x4, we have 2x2 blocks. Let me define the block indices.
    # A is 4x4, partitioned into:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # Similarly for B and C.
    
    # The 4x4 multiplication C = A*B has:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Apply Strassen at the block level:
    # Let A' = [[A00, A01], [A10, A11]], B' = [[B00, B01], [B10, B11]], C' = [[C00, C01], [C10, C11]]
    # Then C' = A'*B' using Strassen's 7 block products.
    
    # The 7 block products are:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    # Each Pi is a 2x2 matrix product. We decompose each using Strassen's 2x2 algorithm.
    
    # This is getting very complex. Let me just implement it computationally.
    
    # I'll build the U, V, W matrices by iterating over the 49 terms.
    
    # Define the 2x2 Strassen terms as above
    strassen_terms_2x2 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: -1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {1: 1, 2: 1}),
    ]
    
    # For 4x4, the block-level Strassen gives 7 block products.
    # Each block product involves sums/differences of 2x2 blocks.
    # Let me define the 7 block-level terms with their U, V, W block coefficients.
    
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11 (and similarly for B, C)
    block_strassen_terms = [
        # (u_blocks, v_blocks, w_blocks) where each is a dict from block_index to coefficient
        # P1 = (A00+A11)*(B00+B11) -> C00, C11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2 = (A10+A11)*B00 -> C10
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # P3 = A00*(B01-B11) -> C01, C11
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: -1}),
        # P4 = A11*(B10-B00) -> C00, C10
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: -1}),
        # P5 = (A00+A01)*B11 -> C00, C01
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6 = (A10-A00)*(B00+B01) -> C11
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7 = (A01-A11)*(B10+B11) -> C01, C10
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {1: 1, 2: 1}),
    ]
    
    # Now, for each block-level term, and for each 2x2 Strassen term within it,
    # we generate scalar-level terms.
    
    # The 4x4 matrix has indices:
    # A: 0..15 (row-major: A[i,j] = i*4+j)
    # B: 0..15
    # C: 0..15 (for the first 4 columns; we'll handle the 5th column separately)
    
    # Block 0 (A00): A[0:2, 0:2] -> indices 0,1,4,5
    # Block 1 (A01): A[0:2, 2:4] -> indices 2,3,6,7
    # Block 2 (A10): A[2:4, 0:2] -> indices 8,9,12,13
    # Block 3 (A11): A[2:4, 2:4] -> indices 10,11,14,15
    
    # Within each 2x2 block, the local indices are:
    # Local 0 -> global depends on block
    # For block b (0,1,2,3), the global indices are:
    # Block 0: [0, 1, 4, 5]
    # Block 1: [2, 3, 6, 7]
    # Block 2: [8, 9, 12, 13]
    # Block 3: [10, 11, 14, 15]
    
    block_to_global_A = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15],
    }
    block_to_global_B = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15],
    }
    # For C (4x4 part, first 4 columns), same as A
    block_to_global_C = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15],
    }
    
    # Now build the U, V, W matrices for the 4x4x4 part
    # U: 16 x rank, V: 16 x rank, W: 16 x rank (for 4x4 output)
    
    # Start with empty lists for the terms
    U_terms = []  # Each term is a dict from global_A_index to coefficient
    V_terms = []  # Each term is a dict from global_B_index to coefficient
    W_terms = []  # Each term is a dict from global_C_index to coefficient
    
    for block_u, block_v, block_w in block_strassen_terms:
        for local_u, local_v, local_w in strassen_terms_2x2:
            # For each block in block_u, and each local index in local_u,
            # add to the global U term
            u_global = {}
            for bu, cu in block_u.items():
                for lu, clu in local_u.items():
                    gu = block_to_global_A[bu][lu]
                    u_global[gu] = u_global.get(gu, 0) + cu * clu
            
            v_global = {}
            for bv, cv in block_v.items():
                for lv, clv in local_v.items():
                    gv = block_to_global_B[bv][lv]
                    v_global[gv] = v_global.get(gv, 0) + cv * clv
            
            w_global = {}
            for bw, cw in block_w.items():
                for lw, clw in local_w.items():
                    gw = block_to_global_C[bw][lw]
                    w_global[gw] = w_global.get(gw, 0) + cw * clw
            
            # Skip if all coefficients are zero
            if u_global and v_global and w_global:
                U_terms.append(u_global)
                V_terms.append(v_global)
                W_terms.append(w_global)
    
    num_4x4_terms = len(U_terms)
    
    # Now add 4 terms for the 5th column
    # For the 5th column: (AB)_{i,4} = sum_{j=0}^{3} A_{i,j} B_{j,4}
    # We need 4 terms, one for each j
    for j in range(4):
        u_global = {}
        for i in range(4):
            u_global[i * 4 + j] = 1  # A[i,j]
        
        v_global = {j * 5 + 4: 1}  # B[j,4]
        
        w_global = {}
        for i in range(4):
            # Output is 5x4, so C[i,4] is at position... 
            # The output factor W has shape (p*n, rank) = (5*4, rank) = (20, rank)
            # W[k*n+i, c] where k is the column of B (0..4), i is the row of A (0..3)
            # So for the 5th column (k=4), row i: W[4*4 + i, c] = W[16 + i, c]
            w_global[16 + i] = 1  # C[i,4]
        
        U_terms.append(u_global)
        V_terms.append(v_global)
        W_terms.append(w_global)
    
    total_rank = len(U_terms)
    
    # Now convert to dense matrices
    U = [[0] * total_rank for _ in range(16)]  # 4*4 = 16 rows
    V = [[0] * total_rank for _ in range(20)]  # 4*5 = 20 rows
    W = [[0] * total_rank for _ in range(20)]  # 5*4 = 20 rows
    
    for c in range(total_rank):
        for idx, val in U_terms[c].items():
            U[idx][c] = val
        for idx, val in V_terms[c].items():
            V[idx][c] = val
        for idx, val in W_terms[c].items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

