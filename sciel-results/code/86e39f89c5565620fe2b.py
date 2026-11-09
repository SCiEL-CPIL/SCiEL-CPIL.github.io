"""Rewritten program for 4x4x9 matrix multiplication with reduced rank.

Uses a hybrid approach:
- For the first 4 columns of B: use a tensor decomposition with rank 49
  (based on 7x7 = 49 structure, analogous to Strassen's 7 for 2x2x2)
- For the remaining 5 columns: schoolbook with rank 80
Total rank: 129
"""

DIMENSIONS = [4, 4, 9]

def construct():
    """Construct the factor matrices for 4x4x9 matrix multiplication.
    
    Returns:
        dict with 'U' (16 x rank), 'V' (36 x rank), 'W' (36 x rank)
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # rank 129 = 49 (for first 4 cols of B) + 80 (for last 5 cols)
    rank = 129
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 129
    V = [[0] * rank for _ in range(m * p)]  # 36 x 129
    W = [[0] * rank for _ in range(p * n)]  # 36 x 129
    
    # Part 1: Rank 49 decomposition for the 4x4x4 subproblem
    # Use a structured decomposition based on 7x7 grid
    # For each (a, b) in 7x7, create a rank-1 term
    col = 0
    for a in range(7):
        for b in range(7):
            # Determine which output indices this term affects
            # and which input indices it reads
            # Using a Kronecker-like structure:
            # u_indices from a, v_indices from b, w_indices from (a,b)
            
            # For 4x4x4: use the following mapping
            # a in 0..6 -> pair (i,j) in 2x2 with offset
            # b in 0..6 -> pair (j,k) in 2x2 with offset
            
            # Strassen-style: 7 terms correspond to the 7 non-trivial
            # rank-1 decompositions of 2x2x2
            # For 4x4x4, we use 7x7 = 49 terms where each is a
            # Kronecker product of two 2x2x2 Strassen terms
            
            # Map a to (block_i, strassen_idx)
            block_i = a // 2  # 0 or 1 (which 2x2 block in row)
            s_idx = a % 2     # 0 or 1 (which Strassen variant)
            
            # Actually, let's use a simpler valid construction:
            # Just use schoolbook for all 4x4x4 as well, but we can
            # try to find a better decomposition.
            # 
            # For now, let's use a known valid decomposition:
            # The tensor for 4x4x4 has rank at most 64 (schoolbook).
            # We claim rank 49 is achievable with a specific construction.
            #
            # Let me use a different approach: use the fact that
            # 4x4x4 can be decomposed using 49 terms via the following:
            # Think of 4x4x4 as a 2x2 block matrix where each block is 2x2.
            # Use 7 multiplications at the block level (Strassen for 2x2x2),
            # and each block multiplication is 2x2x2 with rank 7.
            # Total: 7 * 7 = 49.
            
            # Strassen's 7 formulas for 2x2x2:
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)(B11)
            # M3 = (A11)(B12-B22)
            # M4 = (A22)(B21-B11)
            # M5 = (A11+A12)(B22)
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # For the outer 2x2 block Strassen, we have 7 block-level terms.
            # For the inner 2x2x2, we have 7 scalar-level terms.
            # Total 49 terms.
            
            # Outer block index: oa in 0..6 (which of the 7 Strassen terms)
            # Inner block index: ib in 0..6 (which of the 7 Strassen terms)
            
            # Let's reindex: a = outer, b = inner
            oa = a
            ib = b
            
            # Define the Strassen terms for 2x2x2
            # Each term is (u_coeff, v_coeff, w_output)
            # u_coeff: which A elements with what signs
            # v_coeff: which B elements with what signs
            
            strassen_A = [
                # (row, col, coeff) for A elements
                [(0,0,1), (1,1,1)],  # M1: A11+A22
                [(1,0,1), (1,1,1)],  # M2: A21+A22
                [(0,0,1)],           # M3: A11
                [(1,1,1)],           # M4: A22
                [(0,0,1), (0,1,1)],  # M5: A11+A12
                [(1,0,1), (0,0,-1)], # M6: A21-A11
                [(0,1,1), (1,1,-1)], # M7: A12-A22
            ]
            
            strassen_B = [
                [(0,0,1), (1,1,1)],  # M1: B11+B22
                [(0,0,1)],           # M2: B11
                [(0,1,1), (1,1,-1)], # M3: B12-B22
                [(1,0,1), (0,0,-1)], # M4: B21-B11
                [(1,1,1)],           # M5: B22
                [(0,0,1), (0,1,1)],  # M6: B11+B12
                [(1,0,1), (1,1,1)],  # M7: B21+B22
            ]
            
            strassen_W = [
                [(0,0,1), (1,1,-1), (0,1,1), (1,0,1)],  # C11: M1+M4-M5+M7
                [(0,1,1), (1,1,1)],                     # C12: M3+M5
                [(1,0,1), (1,1,1)],                     # C21: M2+M4
                [(0,0,1), (1,0,-1), (0,1,1), (1,1,1)], # C22: M1-M2+M3+M6
            ]
            # Wait, let me double-check Strassen output formulas:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            strassen_W = [
                [(0,0,1), (1,1,-1), (0,1,-1), (1,0,1)],  # C11: M1+M4-M5+M7
                [(0,1,1), (1,1,1)],                       # C12: M3+M5
                [(1,0,1), (1,1,1)],                       # C21: M2+M4
                [(0,0,1), (1,0,-1), (0,1,1), (1,1,1)],   # C22: M1-M2+M3+M6
            ]
            
            # Now for the outer Strassen (block level):
            # A is 4x4, divided into 2x2 blocks: A00, A01, A10, A11
            # B is 4x4, divided into 2x2 blocks: B00, B01, B10, B11
            # C is 4x4, divided into 2x2 blocks: C00, C01, C10, C11
            
            # Outer Strassen terms operate on blocks:
            # OM1 = (A00+A11)(B00+B11)
            # OM2 = (A10+A11)(B00)
            # OM3 = (A00)(B01-B11)
            # OM4 = (A11)(B10-B00)
            # OM5 = (A00+A01)(B11)
            # OM6 = (A10-A00)(B00+B01)
            # OM7 = (A01-A11)(B10+B11)
            
            # Output blocks:
            # C00 = OM1 + OM4 - OM5 + OM7
            # C01 = OM3 + OM5
            # C10 = OM2 + OM4
            # C11 = OM1 - OM2 + OM3 + OM6
            
            outer_strassen_A = [
                [(0,0,1), (1,1,1)],  # OM1
                [(1,0,1), (1,1,1)],  # OM2
                [(0,0,1)],           # OM3
                [(1,1,1)],           # OM4
                [(0,0,1), (0,1,1)],  # OM5
                [(1,0,1), (0,0,-1)], # OM6
                [(0,1,1), (1,1,-1)], # OM7
            ]
            
            outer_strassen_B = [
                [(0,0,1), (1,1,1)],  # OM1
                [(0,0,1)],           # OM2
                [(0,1,1), (1,1,-1)], # OM3
                [(1,0,1), (0,0,-1)], # OM4
                [(1,1,1)],           # OM5
                [(0,0,1), (0,1,1)],  # OM6
                [(1,0,1), (1,1,1)],  # OM7
            ]
            
            outer_strassen_W = [
                [(0,0,1), (1,1,-1), (0,1,-1), (1,0,1)],  # C00
                [(0,1,1), (1,1,1)],                       # C01
                [(1,0,1), (1,1,1)],                       # C10
                [(0,0,1), (1,0,-1), (0,1,1), (1,1,1)],   # C11
            ]
            
            # For this term (oa, ib):
            # oa determines which outer Strassen term
            # ib determines which inner Strassen term
            
            # U part: combine outer and inner A coefficients
            for (ob_i, ob_j, o_coeff) in outer_strassen_A[oa]:
                # ob_i, ob_j are block indices (0 or 1)
                # Inner A coefficients from strassen_A[ib]
                for (ib_i, ib_j, i_coeff) in strassen_A[ib]:
                    # Convert block + inner to actual index
                    row = ob_i * 2 + ib_i
                    col_a = ob_j * 2 + ib_j
                    coeff = o_coeff * i_coeff
                    U[row * m + col_a][col] += coeff
            
            # V part: combine outer and inner B coefficients
            for (ob_j, ob_k, o_coeff) in outer_strassen_B[oa]:
                # ob_j, ob_k are block indices for B (cols and rows of B)
                # Inner B coefficients from strassen_B[ib]
                for (ib_j, ib_k, i_coeff) in strassen_B[ib]:
                    # B is 4x4: row index is j (0..3), col index is k (0..3)
                    # But in our notation, V[j*p+k] where j is row of B, k is col
                    # For the 4x4x4 part, p=4
                    row_b = ob_j * 2 + ib_j
                    col_b = ob_k * 2 + ib_k
                    coeff = o_coeff * i_coeff
                    V[row_b * 4 + col_b][col] += coeff
            
            # W part: combine outer and inner W coefficients
            for (oc_i, oc_j, o_coeff) in outer_strassen_W[oa]:
                # oc_i, oc_j are block indices for C output
                # Inner W coefficients from strassen_W[ib]
                for (ib_i, ib_j, i_coeff) in strassen_W[ib]:
                    # W[k*n+i] where k is col of B (0..3), i is row of A (0..3)
                    # C output: row = i (0..3), col = k (0..3)
                    # In W indexing: W[k*n + i]
                    row_c = oc_i * 2 + ib_i
                    col_c = oc_j * 2 + ib_j
                    coeff = o_coeff * i_coeff
                    W[col_c * n + row_c][col] += coeff
            
            col += 1
    
    # Part 2: Schoolbook for the remaining 5 columns of B (cols 4-8)
    for k in range(4, 9):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify we used all 129 columns
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
