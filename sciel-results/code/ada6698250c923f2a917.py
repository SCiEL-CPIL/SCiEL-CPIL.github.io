import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def _get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U[0, 0] = 1; U[3, 0] = 1 # A00, A11 (indices 0, 3)
    V[0, 0] = 1; V[3, 0] = 1 # B00, B11 (indices 0, 3)
    W[0, 0] = 1; W[3, 0] = 1 # C00 (k=0,i=0 -> 0), C11 (k=1,i=1 -> 3)
    
    # Term 2: (A10+A11)B00 -> C10 += 1, C11 -= 1
    U[2, 1] = 1; U[3, 1] = 1 # A10 (idx 2), A11 (idx 3)
    V[0, 1] = 1             # B00
    W[2, 1] = 1; W[3, 1] = -1 # C10 (k=1,i=0 -> 2), C11 (k=1,i=1 -> 3)
    
    # Term 3: A00(B01-B11) -> C01 += 1, C11 += 1
    U[0, 2] = 1             # A00
    V[1, 2] = 1; V[3, 2] = -1 # B01 (idx 1), B11 (idx 3)
    W[1, 2] = 1; W[3, 2] = 1  # C01 (k=0,i=1 -> 1), C11 (k=1,i=1 -> 3)
    
    # Term 4: A11(B10-B00) -> C00 += 1, C10 += 1
    U[3, 3] = 1             # A11
    V[2, 3] = 1; V[0, 3] = -1 # B10 (idx 2), B00 (idx 0)
    W[0, 3] = 1; W[2, 3] = 1  # C00 (k=0,i=0 -> 0), C10 (k=1,i=0 -> 2)
    
    # Term 5: (A00+A01)B11 -> C00 -= 1, C01 += 1
    U[0, 4] = 1; U[1, 4] = 1 # A00, A01 (idx 1)
    V[3, 4] = 1             # B11
    W[0, 4] = -1; W[1, 4] = 1 # C00, C01
    
    # Term 6: (A10-A00)(B00+B01) -> C11 += 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1             # C11
    
    # Term 7: (A01-A11)(B10+B11) -> C00 += 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1             # C00
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Part 2: 4x4x2 (Rank 28)
    total_rank = 77
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    U2, V2, W2 = _get_strassen_2x2x2()
    
    # Strassen 4x4x4 steps
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Helper to add a 2x2x2 block at offset
    def add_block(offset, u_block, v_block, w_block, k_offset=0):
        # u_block is 4x7, v_block 4x7, w_block 4x7
        # We need to map local indices to global
        # This is tricky because the blocks are not just simple offsets in all dimensions
        # We will manually construct the 4x4x4 factors using the 2x2x2 factors
        
        pass

    # Manual construction for 4x4x4
    # Indices for A (4x4): 0..15
    # Indices for B (4x4): 0..15
    # Indices for C (4x4): 0..15 (k*4+i)
    
    # We will create a list of (U_vec, V_vec, W_vec)
    terms = []
    
    def add_2x2_term(u_coeffs, v_coeffs, w_coeffs, i_off, j_off, k_off, i_len=2, j_len=2, k_len=2):
        # u_coeffs: dict (i,j) -> val
        # This is getting complex. Let's just generate the 49 terms explicitly.
        pass

    # Alternative: Use the property that Strassen 4x4x4 is 7 copies of 2x2x2.
    # We can generate the 7 "super-terms" and then expand each.
    
    # Let's define the 7 linear forms for A, B and the C coefficients for 4x4x4
    # A blocks: A00 (0,0), A01 (0,2), A10 (2,0), A11 (2,2)
    
    # M1: (A00+A11)(B00+B11)
    # A coeffs: A00: 1, A11: 1
    # B coeffs: B00: 1, B11: 1
    # C coeffs: C00: 1, C11: 1 (in terms of 2x2 blocks? No, C is 4x4)
    # C00 (top-left 2x2) += M1. C11 (bottom-right 2x2) += M1.
    
    # We will iterate over the 7 M_k's and for each, iterate over the 7 local terms.
    
    strassen_4_steps = [
        # (A_blocks, B_blocks, C_blocks, C_signs)
        # A_blocks: list of (block_idx, coeff)
        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
        ([ (0,1), (3,1) ], [ (0,1), (3,1) ], [ (0,1), (3,1) ]), # M1
        ([ (2,1), (3,1) ], [ (0,1) ], [ (2,1), (3,-1) ]),         # M2
        ([ (0,1) ], [ (1,1), (3,-1) ], [ (1,1), (3,1) ]),         # M3
        ([ (3,1) ], [ (2,1), (0,-1) ], [ (0,1), (2,1) ]),         # M4
        ([ (0,1), (1,1) ], [ (3,1) ], [ (0,-1), (1,1) ]),         # M5
        ([ (2,1), (0,-1) ], [ (0,1), (1,1) ], [ (3,1) ]),         # M6
        ([ (1,1), (3,-1) ], [ (2,1), (3,1) ], [ (0,1) ]),         # M7
    ]
    
    col = 0
    for a_blocks, b_blocks, c_blocks in strassen_4_steps:
        # Construct U, V, W for this M_k (which is a 2x2x2 multiplication)
        # We need to sum the contributions from the blocks
        U_mk = np.zeros((4, 7))
        V_mk = np.zeros((4, 7))
        W_mk = np.zeros((4, 7))
        
        for (b_idx, coeff) in a_blocks:
            # b_idx 0: A00 (i 0-1, j 0-1)
            # b_idx 1: A01 (i 0-1, j 2-3)
            # b_idx 2: A10 (i 2-3, j 0-1)
            # b_idx 3: A11 (i 2-3, j 2-3)
            i_off = (b_idx // 2) * 2
            j_off = (b_idx % 2) * 2
            for i in range(2):
                for j in range(2):
                    local_i = i
                    local_j = j
                    # Add to U_mk at global index
                    gi = i_off + i
                    gj = j_off + j
                    U_mk[gi*2+gj, :] += coeff * U2[local_i*2+local_j, :]
                    
        for (b_idx, coeff) in b_blocks:
            i_off = (b_idx // 2) * 2
            j_off = (b_idx % 2) * 2
            for i in range(2):
                for j in range(2):
                    local_i = i
                    local_j = j
                    gi = i_off + i
                    gj = j_off + j
                    V_mk[gi*2+gj, :] += coeff * V2[local_i*2+local_j, :]
                    
        for (b_idx, coeff) in c_blocks:
            i_off = (b_idx // 2) * 2
            j_off = (b_idx % 2) * 2
            for i in range(2):
                for j in range(2):
                    local_i = i
                    local_j = j
                    gi = i_off + i
                    gj = j_off + j
                    W_mk[gj*2+gi, :] += coeff * W2[local_j*2+local_i, :]
                    
        U[:, col:col+7] = U_mk
        V[:, col:col+7] = V_mk
        W[:, col:col+7] = W_mk
        col += 7
        
    # Now Part 2: 4x4x2
    # A is 4x4. B_R is 4x2.
    # C_R is 4x2.
    # C_R[0:2, :] = A[0:2, :] B_R[0:2, :] + A[0:2, 2:4] B_R[2:4, :]
    # C_R[2:4, :] = A[2:4, :] B_R[0:2, :] + A[2:4, 2:4] B_R[2:4, :]
    
    # 4 terms:
    # T1: A00 B1R -> C1R (top half)
    # T2: A01 B2R -> C1R
    # T3: A10 B1R -> C2R (bottom half)
    # T4: A11 B2R -> C2R
    
    # B1R is B[0:2, 4:6] (cols 4,5)
    # B2R is B[2:4, 4:6]
    
    # For each term, we use 2x2x2 Strassen.
    # A block is 2x2. B block is 2x2. C block is 2x2.
    
    terms_4x4x2 = [
        # (A_block_idx, B_block_idx, C_block_idx)
        # A_idx: 0=A00, 1=A01, 2=A10, 3=A11
        # B_idx: 0=B1R (rows 0-1), 1=B2R (rows 2-3)
        # C_idx: 0=C1R (rows 0-1), 1=C2R (rows 2-3)
        (0, 0, 0),
        (1, 1, 0),
        (2, 0, 1),
        (3, 1, 1),
    ]
    
    for a_idx, b_idx, c_idx in terms_4x4x2:
        i_off = (a_idx // 2) * 2
        j_off = (a_idx % 2) * 2
        
        # B block:
        # b_idx 0: rows 0-1, cols 4-5
        # b_idx 1: rows 2-3, cols 4-5
        bi_off = b_idx * 2
        bj_off = 4 # cols 4,5
        
        # C block:
        # c_idx 0: rows 0-1, cols 4-5
        # c_idx 1: rows 2-3, cols 4-5
        ci_off = c_idx * 2
        cj_off = 4
        
        U_mk = np.zeros((4, 7))
        V_mk = np.zeros((4, 7))
        W_mk = np.zeros((4, 7))
        
        # A part
        for i in range(2):
            for j in range(2):
                gi = i_off + i
                gj = j_off + j
                U_mk[gi*2+gj, :] = U2[i*2+j, :]
                
        # B part
        for i in range(2):
            for j in range(2):
                gi = bi_off + i
                gj = bj_off + j
                V_mk[gi*2+gj, :] = V2[i*2+j, :]
                
        # C part
        for i in range(2):
            for j in range(2):
                gi = ci_off + i
                gj = cj_off + j
                W_mk[gj*2+gi, :] = W2[j*2+i, :]
                
        U[:, col:col+7] = U_mk
        V[:, col:col+7] = V_mk
        W[:, col:col+7] = W_mk
        col += 7
        
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                rounded = round(val * 2) / 2
                r.append(rounded)
            result.append(r)
        return result
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
