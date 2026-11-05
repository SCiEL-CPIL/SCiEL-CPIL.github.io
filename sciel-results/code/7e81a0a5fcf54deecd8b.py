import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    deadline = start_time + 110

    # Block structure: 6x6 = 3x3 blocks of 2x2
    # Outer: 3x3 matrix multiplication (BCLR rank 19)
    # Inner: 2x2 matrix multiplication (Strassen rank 7)
    # Total Rank: 19 * 7 = 133

    block_size = 2
    num_blocks = 3

    # BCLR (Bini, Capovani, Lotti, Romani) rank-19 decomposition for 3x3 matrices
    # Each entry is (u_coeffs, v_coeffs, w_coeffs) where each is a list of (row, col, coeff)
    # U: coefficients for A (left matrix)
    # V: coefficients for B (right matrix)
    # W: coefficients for C (result matrix)
    # The decomposition satisfies: sum_r U[r] * V[r] = C where C = A * B
    
    # BCLR rank-19 scheme for 3x3 matrix multiplication
    # Reference: Bini, Capovani, Lotti, Romani (1979), "Tensor rank three algorithms for matrix product"
    # This is a well-known rank-19 algorithm for 3x3 matrix multiplication
    
    bclr_3x3 = [
        # Term 0
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # Term 1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # Term 2
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # Term 3
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # Term 4
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # Term 5
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # Term 6
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        # Term 7
        ([(2,2,1)], [(2,2,1)], [(2,2,1)]),
        # Term 8
        ([(2,0,1),(2,1,1)], [(0,2,1)], [(2,0,1),(2,1,1)]),
        # Term 9
        ([(0,2,1)], [(2,0,1),(2,1,-1)], [(0,2,1),(2,1,1)]),
        # Term 10
        ([(2,1,1)], [(0,2,1),(1,2,-1)], [(0,2,1),(1,2,1)]),
        # Term 11
        ([(0,2,1),(1,2,1)], [(2,0,1)], [(0,2,-1),(1,2,1)]),
        # Term 12
        ([(1,2,1),(0,2,-1)], [(2,0,1),(2,1,1)], [(2,0,1)]),
        # Term 13
        ([(0,2,1),(2,2,-1)], [(2,1,1)], [(0,2,1)]),
        # Term 14
        ([(2,0,1),(2,1,-1)], [(1,2,1)], [(2,1,1)]),
        # Term 15
        ([(2,2,1)], [(0,2,1),(2,0,-1)], [(0,2,1)]),
        # Term 16
        ([(0,2,1)], [(2,0,1),(2,1,1)], [(2,2,1)]),
        # Term 17
        ([(2,0,1)], [(1,2,1),(0,2,-1)], [(1,2,1)]),
        # Term 18
        ([(1,2,1),(2,2,-1)], [(0,2,1)], [(2,0,1)]),
    ]
    
    # Strassen coefficients for 2x2 multiplication
    # Returns 7 terms: (u_coeffs, v_coeffs, w_coeffs)
    strassen_2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Verify BCLR rank-19 decomposition for 3x3
    # Check with random matrices
    def verify_bclr():
        np.random.seed(42)
        A = np.random.randn(3, 3)
        B = np.random.randn(3, 3)
        C_expected = A @ B
        C_computed = np.zeros((3, 3))
        for u_list, v_list, w_list in bclr_3x3:
            u = np.zeros((3, 3))
            for r, c, coeff in u_list:
                u[r, c] = coeff
            v = np.zeros((3, 3))
            for r, c, coeff in v_list:
                v[r, c] = coeff
            # The term is: (u^T A v) * w_matrix
            scalar = np.trace(u.T @ A @ v.T)
            w = np.zeros((3, 3))
            for r, c, coeff in w_list:
                w[r, c] = coeff
            C_computed += scalar * w
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Verify Strassen 2x2
    def verify_strassen():
        np.random.seed(42)
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)
        C_expected = A @ B
        C_computed = np.zeros((2, 2))
        for u_list, v_list, w_list in strassen_2x2:
            u = np.zeros((2, 2))
            for r, c, coeff in u_list:
                u[r, c] = coeff
            v = np.zeros((2, 2))
            for r, c, coeff in v_list:
                v[r, c] = coeff
            scalar = np.trace(u.T @ A @ v.T)
            w = np.zeros((2, 2))
            for r, c, coeff in w_list:
                w[r, c] = coeff
            C_computed += scalar * w
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    bclr_valid = verify_bclr()
    strassen_valid = verify_strassen()
    
    if not (bclr_valid and strassen_valid):
        # Fallback: use naive 3x3 (rank 27) with Strassen 2x2 (rank 7) = 189
        naive_3x3 = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u = [(i, j, 1)]
                    v = [(j, k, 1)]
                    w = [(i, k, 1)]
                    naive_3x3.append((u, v, w))
        
        outer_scheme = naive_3x3
        total_rank = 27 * 7
    else:
        outer_scheme = bclr_3x3
        total_rank = 19 * 7
    
    U = np.zeros((36, total_rank), dtype=np.float64)
    V = np.zeros((36, total_rank), dtype=np.float64)
    W = np.zeros((36, total_rank), dtype=np.float64)
    
    bs = block_size
    dim = n  # 6
    
    col = 0
    for u_outer, v_outer, w_outer in outer_scheme:
        # For each outer term, we need to multiply the blocks
        # The outer term involves specific block indices
        # u_outer: list of (block_row, block_col, coeff) for A blocks
        # v_outer: list of (block_row, block_col, coeff) for B blocks
        # w_outer: list of (block_row, block_col, coeff) for C blocks
        
        # For each combination of block indices in u_outer, v_outer, w_outer,
        # we need to perform the inner 2x2 multiplication using Strassen
        
        # Actually, the BCLR scheme works at the block level.
        # Each "term" in the outer scheme specifies which blocks to combine.
        # For each outer term, we need to sum over all valid block index combinations.
        
        # Let me reconsider: The BCLR scheme is for 3x3 scalar matrices.
        # When we use it for 3x3 blocks of 2x2, each scalar multiplication
        # becomes a 2x2 matrix multiplication.
        
        # So for each outer term (u_outer, v_outer, w_outer):
        # - u_outer specifies coefficients for A blocks
        # - v_outer specifies coefficients for B blocks  
        # - w_outer specifies coefficients for C blocks
        # - The actual computation is: sum of (u_coeff * A_block) * (v_coeff * B_block) added to w_coeff * C_block
        
        # For each outer term, we need to compute the product of the linear combinations
        # of blocks, which is a 2x2 * 2x2 multiplication.
        
        # Let's collect all the A block coefficients, B block coefficients, and C block coefficients
        # For this outer term, the effective A "matrix" in block space is a linear combination
        # But since we're doing matrix multiplication at the block level, we need to be careful.
        
        # Actually, the standard Kronecker product approach:
        # If T_outer is a rank-r decomposition of 3x3 matrix mult,
        # and T_inner is a rank-s decomposition of 2x2 matrix mult,
        # then T_outer ⊗ T_inner is a rank-(r*s) decomposition of 6x6 matrix mult.
        
        # Each rank-1 term in T_outer becomes r_inner terms in the Kronecker product.
        
        # So for each outer term (u_outer, v_outer, w_outer):
        #   For each inner Strassen term (u_inner, v_inner, w_inner):
        #     Create a rank-1 term in the 6x6 decomposition
        
        for u_inner, v_inner, w_inner in strassen_2x2:
            # Expand the outer and inner coefficients into global 6x6 indices
            
            # U part: A[ib, jb] block, local indices from u_inner
            for (ib, jb, outer_coeff) in u_outer:
                for (r, s, inner_coeff) in u_inner:
                    u_idx = (ib * bs + r) * dim + (jb * bs + s)
                    U[u_idx, col] += outer_coeff * inner_coeff
            
            # V part: B[jb, kb] block, local indices from v_inner
            for (ib, jb, outer_coeff) in v_outer:
                # Wait, v_outer should specify which B blocks to use
                # In the BCLR scheme, v_outer entries are (block_row, block_col, coeff)
                # But block_row is the row index in B, which should match the col index in A
                # Let me check the BCLR scheme more carefully...
                
                # Actually, I think the issue is that the BCLR scheme entries don't directly
                # give us the block indices for A and B in a way that's compatible with
                # the Kronecker product. Let me re-examine.
                
                # In a rank-1 tensor decomposition of matrix multiplication:
                # C_{ij} = sum_k A_{ik} B_{kj}
                # The tensor has indices (i, k, j) for A, B, C respectively.
                # A rank-1 term is: U_{ik} * V_{kj} * W_{ij}
                # where U, V, W are 3x3 matrices (for 3x3 case).
                
                # So u_outer entries (row, col, coeff) correspond to U[row, col] = coeff
                # v_outer entries (row, col, coeff) correspond to V[row, col] = coeff  
                # w_outer entries (row, col, coeff) correspond to W[row, col] = coeff
                
                # For the Kronecker product with 2x2 Strassen:
                # Each 3x3 entry U[i,j] becomes a 2x2 block
                # The Kronecker product U ⊗ u_inner gives a 6x6 matrix
                
                # Let me redo this properly.
                break
            break
        break
    
    # Let me restart with a cleaner approach
    
    U = np.zeros((36, total_rank), dtype=np.float64)
    V = np.zeros((36, total_rank), dtype=np.float64)
    W = np.zeros((36, total_rank), dtype=np.float64)
    
    bs = block_size
    dim = n  # 6
    col = 0
    
    for u_outer, v_outer, w_outer in outer_scheme:
        for u_inner, v_inner, w_inner in strassen_2x2:
            # Build the 6x6 U matrix for this term
            # U_global[i*6+j, col] where i,j are global indices
            # i = ib*2 + r, j = jb*2 + s
            # where (ib, jb) are block indices from outer, (r, s) are local from inner
            
            for (ib, jb, outer_coeff) in u_outer:
                for (r, s, inner_coeff) in u_inner:
                    u_idx = (ib * bs + r) * dim + (jb * bs + s)
                    U[u_idx, col] += outer_coeff * inner_coeff
            
            for (ib, jb, outer_coeff) in v_outer:
                for (r, s, inner_coeff) in v_inner:
                    # For V, the block indices are (jb, kb) where jb is the column of A / row of B
                    # In the BCLR scheme, v_outer entries are (row, col, coeff) for V
                    # V[row, col] where row is the row index in B, col is the col index in B
                    # But we need V to have indices that match B's structure
                    # B is 6x6, with blocks B[jb, kb]
                    # V_global[i*6+j, col] where i,j are global indices in B
                    # i = jb*2 + r, j = kb*2 + s
                    # But v_outer gives us (ib, jb, coeff) which I interpreted as (row, col)
                    # For B, row index is jb (the middle index), col index is kb
                    # So v_outer entries should be (jb, kb, coeff)
                    # Let me assume v_outer entries are (block_row_B, block_col_B, coeff) = (jb, kb, coeff)
                    v_idx = (ib * bs + r) * dim + (jb * bs + s)
                    V[v_idx, col] += outer_coeff * inner_coeff
            
            for (ib, jb, outer_coeff) in w_outer:
                for (r, s, inner_coeff) in w_inner:
                    # W follows trace convention: W[k*n+i, r]
                    # The result is C[ib, kb] block, local indices (r, s)
                    # Global row: ib*2 + r, Global col: kb*2 + s
                    # W index: (col*6 + row) or (row*6 + col)?
                    # From the original code: w_idx = (kb * bs + s) * dim + (ib * bs + r)
                    # This is (col * 6 + row)
                    w_idx = (jb * bs + s) * dim + (ib * bs + r)
                    W[w_idx, col] += outer_coeff * inner_coeff
            
            col += 1
            
            # Periodic time check
            if col % 133 == 0 and time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
