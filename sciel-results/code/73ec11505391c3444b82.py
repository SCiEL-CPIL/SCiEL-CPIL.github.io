"""Strassen-based rank-189 decomposition for 6x6 matrix multiplication."""
DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 7 * 27  # 189

    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 7 multiplications for 2x2 block structure
    # Each step: (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (block_row, block_col, coeff)
    # B_blocks: list of (block_row, block_col, coeff)
    # C_blocks: list of (block_row, block_col, coeff)
    
    strassen_steps = [
        # M1 = (A00 + A11)(B00 + B11)
        ([(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (0, 1, -1), (1, 0, 1), (1, 1, 1)]),
        
        # M2 = (A10 + A11)B00
        ([(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)],
         [(1, 0, 1), (1, 1, 1)]),
        
        # M3 = A00(B01 - B11)
        ([(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)],
         [(0, 1, 1), (1, 1, 1)]),
        
        # M4 = A11(B10 - B00)
        ([(1, 1, 1)],
         [(1, 0, 1), (0, 0, -1)],
         [(0, 0, -1), (1, 0, 1)]),
        
        # M5 = (A00 + A01)B11
        ([(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)],
         [(0, 0, -1), (0, 1, 1)]),
        
        # M6 = (A10 - A00)(B00 + B01)
        ([(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)]),
        
        # M7 = (A01 - A11)(B10 + B11)
        ([(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)],
         [(0, 1, 1), (1, 1, 1)]),
    ]

    col = 0
    for A_blocks, B_blocks, C_blocks in strassen_steps:
        # For each local position in the 3x3 block product
        for r in range(3):  # row within block
            for c in range(3):  # col within block (inner dimension)
                for k in range(3):  # row within B block / output col
                
                    # Collect all contributions to this (r,c,k) position
                    # from all pairs of (A_block, B_block) that match on the inner index
                    
                    # We need to sum over all valid (bx, by) for A and (by, bz) for B
                    # where the inner index 'by' matches
                    
                    # For a fixed (r, c, k), the term is:
                    # sum over bx, by, bz: coeff_A(bx,by) * coeff_B(by,bz) * A[bx*3+r, by*3+c] * B[by*3+c, bz*3+k]
                    # This contributes to C blocks (bx, bz) with coefficient from C_blocks
                    
                    # We need to create one rank-1 term per (r,c,k) for each Strassen step
                    # But we need to handle the linear combinations properly
                    
                    # For this specific (r, c, k), we create one column in U, V, W
                    # U gets the coefficients for A entries
                    # V gets the coefficients for B entries  
                    # W gets the coefficients for C entries
                    
                    # But wait - this is a rank-1 term, so we need:
                    # U[i,j] * V[j,k] * W[k,i] = sum of products
                    
                    # Actually, for each (r,c,k) in each Strassen step, we have ONE rank-1 term
                    # where:
                    # U has non-zero entries at positions corresponding to A blocks in A_blocks
                    # V has non-zero entries at positions corresponding to B blocks in B_blocks
                    # W has non-zero entries at positions corresponding to C blocks in C_blocks
                    
                    # The specific values:
                    # For A block (bx, by) with coeff ca: U[bx*3+r, by*3+c] += ca
                    # For B block (by, bz) with coeff cb: V[by*3+c, bz*3+k] += cb
                    # For C block (bx, bz) with coeff cc: W[bz*3+k, bx*3+r] += cc
                    
                    # But this only works if the inner index 'by' is consistent!
                    # In Strassen, the blocks are aligned so that A's col block = B's row block
                    
                    # Let me reconsider: for a valid Strassen term, we need pairs where
                    # A block (bx, by) pairs with B block (by, bz) - same 'by'
                    
                    # So for this (r,c,k), we need to find all valid (bx,by) and (by,bz) pairs
                    # and accumulate the coefficients
                    
                    # However, this creates a problem: a single rank-1 term can only have
                    # a fixed set of non-zero entries. If multiple (bx,by) pairs contribute,
                    # we need separate rank-1 terms for each.
                    
                    # Let me restructure: iterate over each valid (bx, by, bz) combination
                    
                    for bx, by_a, ca in A_blocks:
                        for by_b, bz, cb in B_blocks:
                            if by_a != by_b:
                                continue
                            
                            # This is a valid pair
                            # A index: (bx*3+r, by_a*3+c)
                            # B index: (by_b*3+c, bz*3+k)
                            # C index: (bz*3+k, bx*3+r)
                            
                            # But we also need the C coefficient
                            # The C block (bx, bz) should be in C_blocks with some coefficient
                            cc = 0
                            for cbx, cbz, ccoeff in C_blocks:
                                if cbx == bx and cbz == bz:
                                    cc = ccoeff
                                    break
                            
                            if cc == 0:
                                continue
                            
                            # Now create this rank-1 term
                            i_a = bx * 3 + r
                            j_a = by_a * 3 + c
                            k_b = by_b * 3 + c
                            l_b = bz * 3 + k
                            i_c = bx * 3 + r
                            k_c = bz * 3 + k  # W is indexed as (k, i)
                            
                            # W index: (k_c, i_c) = (bz*3+k, bx*3+r)
                            w_idx = k_c * n + i_c
                            
                            U[i_a * m + j_a][col] = ca
                            V[k_b * p + l_b][col] = cb
                            W[w_idx][col] = cc
                            
                            col += 1

    # Verify we used all columns
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
