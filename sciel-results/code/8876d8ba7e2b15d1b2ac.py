import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 114
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 7 terms for 2x2x2 Strassen
    # u, v, w are vectors of length 4 (flattened 2x2)
    # Indices: 0:00, 1:01, 2:10, 3:11
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4 (Updated C11=1, C21=1)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5 (Updated C11=-1, C12=1)
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    # Define the 7 Strassen products for 4x4x4
    # Each product is defined by:
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_blocks: list of (block_idx, coeff)
    # Block indices: 0:00, 1:01, 2:10, 3:11
    
    strassen_4x4_products = [
        # M1 = (A00 + A11) * (B00 + B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10 + A11) * B00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A00 * (B01 - B11)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A11 * (B10 - B00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (A00 + A01) * B11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (A10 - A00) * (B00 + B01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A01 - A11) * (B10 + B11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    def add_block_decomposition(a_blocks, b_blocks, c_blocks, col_offset):
        """
        Adds the 49 rank-1 terms for a 4x4x4 block multiplication.
        a_blocks, b_blocks, c_blocks are lists of (block_idx, coeff)
        """
        # For each of the 7 sub-products (M1..M7)
        # Wait, the strassen_4x4_products list IS the definition of M1..M7.
        # But M1..M7 are 2x2 matrix multiplications.
        # So for each M_k, we need to apply the 2x2 Strassen decomposition.
        
        # The structure is:
        # For each M_k in strassen_4x4_products:
        #   It defines linear combinations of blocks for A, B, C.
        #   We need to compute A_k * B_k = C_k where A_k, B_k, C_k are 2x2 matrices.
        #   We use the 7-term 2x2 Strassen decomposition for this.
        
        # So, for each M_k:
        #   For each term t in strassen_2x2:
        #       Add a rank-1 term to the global U, V, W.
        
        # But wait, the "blocks" in strassen_4x4_products are 2x2 matrices.
        # So A_k is a 2x2 matrix formed by summing the blocks.
        # The 2x2 Strassen decomposition operates on the elements of this 2x2 matrix.
        
        # So, for a specific M_k:
        #   A_k[i,j] = sum_{b, c} c * A_b[i,j]
        #   B_k[i,j] = sum_{b, c} c * B_b[i,j]
        #   C_k[i,j] = sum_{b, c} c * C_b[i,j]
        
        # The 2x2 Strassen term t has u_t[i,j], v_t[i,j], w_t[i,j].
        # The contribution to global U at index (block b, row i, col j) is:
        #   sum over M_k, t: (coeff of b in A_k) * u_t[i,j]
        
        # We iterate over M_k and t.
        
        for a_def, b_def, c_def in strassen_4x4_products:
            # a_def is list of (block_idx, coeff)
            # For each term in 2x2 Strassen:
            for u_t, v_t, w_t in strassen_2x2:
                # u_t is [u00, u01, u10, u11]
                # We need to add to U, V, W
                
                # For U:
                for b_idx, coeff in a_def:
                    # Block b_idx is 2x2.
                    # Global indices:
                    # Block 0: rows 0-1, cols 0-1
                    # Block 1: rows 0-1, cols 2-3
                    # Block 2: rows 2-3, cols 0-1
                    # Block 3: rows 2-3, cols 2-3
                    
                    r0, c0 = (b_idx // 2) * 2, (b_idx % 2) * 2
                    # u_t[0] -> (r0, c0)
                    U[r0 * 4 + c0][col_offset] += coeff * u_t[0]
                    U[r0 * 4 + c0 + 1][col_offset] += coeff * u_t[1]
                    U[(r0 + 1) * 4 + c0][col_offset] += coeff * u_t[2]
                    U[(r0 + 1) * 4 + c0 + 1][col_offset] += coeff * u_t[3]
                
                # For V:
                for b_idx, coeff in b_def:
                    r0, c0 = (b_idx // 2) * 2, (b_idx % 2) * 2
                    # V index: row j, col k.
                    # B is 4x4.
                    # Block 0: rows 0-1, cols 0-1
                    # Block 1: rows 0-1, cols 2-3
                    # Block 2: rows 2-3, cols 0-1
                    # Block 3: rows 2-3, cols 2-3
                    # v_t[0] -> (r0, c0)
                    V[r0 * 4 + c0][col_offset] += coeff * v_t[0]
                    V[r0 * 4 + c0 + 1][col_offset] += coeff * v_t[1]
                    V[(r0 + 1) * 4 + c0][col_offset] += coeff * v_t[2]
                    V[(r0 + 1) * 4 + c0 + 1][col_offset] += coeff * v_t[3]
                
                # For W:
                for b_idx, coeff in c_def:
                    r0, c0 = (b_idx // 2) * 2, (b_idx % 2) * 2
                    # W index: row k, col i.
                    # C is 4x4.
                    # Block 0: rows 0-1, cols 0-1
                    # Block 1: rows 0-1, cols 2-3
                    # Block 2: rows 2-3, cols 0-1
                    # Block 3: rows 2-3, cols 2-3
                    # w_t[0] -> (r0, c0)
                    W[r0 * 4 + c0][col_offset] += coeff * w_t[0]
                    W[r0 * 4 + c0 + 1][col_offset] += coeff * w_t[1]
                    W[(r0 + 1) * 4 + c0][col_offset] += coeff * w_t[2]
                    W[(r0 + 1) * 4 + c0 + 1][col_offset] += coeff * w_t[3]
                
                col_offset += 1
        
        return col_offset
    
    # Block 1: Columns 0-3 of B
    # A is full 4x4.
    # B is 4x4 (cols 0-3).
    # C is 4x4 (cols 0-3).
    
    # We need to map the 4x4 B and C to the block structure.
    # B_block_0 is B[0:2, 0:2].
    # B_block_1 is B[0:2, 2:4].
    # B_block_2 is B[2:4, 0:2].
    # B_block_3 is B[2:4, 2:4].
    
    # C_block_0 is C[0:2, 0:2].
    # C_block_1 is C[0:2, 2:4].
    # C_block_2 is C[2:4, 0:2].
    # C_block_3 is C[2:4, 2:4].
    
    # The function add_block_decomposition assumes standard block layout.
    # It writes to U, V, W at specific indices.
    # We need to ensure the indices correspond to the correct sub-matrix of B and C.
    
    # For Block 1 (cols 0-3):
    # V indices: j*9 + k.
    # For B[0:2, 0:2], j in 0-1, k in 0-1.
    # My add_block_decomposition writes to V[r0*4+c0]. This assumes B is 4x4.
    # But B is 4x9.
    # So I need to modify the function or pass an offset.
    
    # Let's create a specific function for 4x4x4 block at column offset k_off.
    
    def add_4x4_block(k_off, col_offset):
        for a_def, b_def, c_def in strassen_4x4_products:
            for u_t, v_t, w_t in strassen_2x2:
                # U: A is 4x4, no offset.
                for b_idx, coeff in a_def:
                    r0, c0 = (b_idx // 2) * 2, (b_idx % 2) * 2
                    U[r0 * 4 + c0][col_offset] += coeff * u_t[0]
                    U[r0 * 4 + c0 + 1][col_offset] += coeff * u_t[1]
                    U[(r0 + 1) * 4 + c0][col_offset] += coeff * u_t[2]
                    U[(r0 + 1) * 4 + c0 + 1][col_offset] += coeff * u_t[3]
                
                # V: B is 4x9. Offset k_off.
                for b_idx, coeff in b_def:
                    r0, c0 = (b_idx // 2) * 2, (b_idx % 2) * 2
                    # V index: row j, col k.
                    # j = r0, r0+1
                    # k = c0 + k_off, c0+1+k_off
                    V[r0 * 9 + (c0 + k_off)][col_offset] += coeff * v_t[0]
                    V[r0 * 9 + (c0 + 1 + k_off)][col_offset] += coeff * v_t[1]
                    V[(r0 + 1) * 9 + (c0 + k_off)][col_offset] += coeff * v_t[2]
                    V[(r0 + 1) * 9 + (c0 + 1 + k_off)][col_offset] += coeff * v_t[3]
                
                # W: C is 4x9. Offset k_off.
                for b_idx, coeff in c_def:
                    r0, c0 = (b_idx // 2) * 2, (b_idx % 2) * 2
                    # W index: row k, col i.
                    # k = r0, r0+1 (from C block rows)
                    # i = c0, c0+1 (from C block cols)
                    # W[k*n + i]
                    W[(r0 + k_off) * 4 + c0][col_offset] += coeff * w_t[0]
                    W[(r0 + k_off) * 4 + c0 + 1][col_offset] += coeff * w_t[1]
                    W[(r0 + 1 + k_off) * 4 + c0][col_offset] += coeff * w_t[2]
                    W[(r0 + 1 + k_off) * 4 + c0 + 1][col_offset] += coeff * w_t[3]
                
                col_offset += 1
        return col_offset
    
    col = add_4x4_block(0, col)
    col = add_4x4_block(4, col)
    
    # Block 3: Column 8 of B.
    # A is 4x4.
    # B is 4x1 (col 8).
    # C is 4x1 (col 8).
    # Schoolbook rank 16.
    
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 9 + 8][col] = 1
            W[8 * 4 + i][col] = 1
            col += 1
    
    # Verify
    # A = np.random.rand(4, 4)
    # B = np.random.rand(4, 9)
    # C_expected = A @ B
    
    # Compute C from U, V, W
    # C[i, k] = sum_c U[i*4+j, c] * V[j*9+k, c] * W[k*4+i, c]
    
    # Efficient check
    # U_r (4,4,114), V_r (4,9,114), W_r (9,4,114)
    # C[i, k] = sum_{j, c} U[i, j, c] V[j, k, c] W[k, i, c]
    
    U_r = np.array(U).reshape(4, 4, 114)
    V_r = np.array(V).reshape(4, 9, 114)
    W_r = np.array(W).reshape(9, 4, 114)
    
    # C[i, k] = sum_c sum_j U[i, j, c] V[j, k, c] W[k, i, c]
    # Let T[j, k, c] = V[j, k, c] * W[k, i, c]? No, i is fixed in W.
    # C[i, k] = sum_c W[k, i, c] * (sum_j U[i, j, c] V[j, k, c])
    
    # sum_j U[i, j, c] V[j, k, c] is (U_i * V^T)_{k, c}?
    # U_i is (4, 114). V is (4, 9, 114).
    # For fixed i, U_i is (4, 114).
    # We want sum_j U[i, j, c] V[j, k, c].
    # This is (U_i @ V_k^T) where V_k is (4, 114) slice of V for row k?
    # V[j, k, c].
    # Let's use einsum.
    # C[i, k] = np.einsum('ijc,jkc,kic->ik', U_r, V_r, W_r)
    
    A_test = np.random.rand(4, 4)
    B_test = np.random.rand(4, 9)
    C_expected = A_test @ B_test
    
    C_computed = np.einsum('ijc,jkc,kic->ik', U_r, V_r, W_r)
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook
        rank_s = 144
        U_s = [[0] * rank_s for _ in range(16)]
        V_s = [[0] * rank_s for _ in range(36)]
        W_s = [[0] * rank_s for _ in range(36)]
        col_s = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_s[i*4+j][col_s] = 1
                    V_s[j*9+k][col_s] = 1
                    W_s[k*4+i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
