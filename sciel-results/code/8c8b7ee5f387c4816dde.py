"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # A is 4x4, B is 4x5.
    # Partition A into 2x2 blocks of size 2x2.
    # Partition B into 2 row-blocks of size 2, and 2 col-blocks: size 2 and 3.
    # This leads to 4 blocks of 2x2x2 (rank 7) and 4 blocks of 2x2x3 (rank 11).
    # Total rank = 4*7 + 4*11 = 28 + 44 = 72.
    
    rank_222 = 7
    rank_223 = 11
    total_rank = 4 * rank_222 + 4 * rank_223
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_222(i_b, j_b, k_b, offset):
        # i_b, j_b: block indices for A (0..1)
        # k_b: block index for B (0..1, but for 222 it's 0 or 1 depending on which part of B)
        # Actually, for 2x2x2, B is 2x2.
        # We need to map local indices to global.
        
        # Strassen coefficients for 2x2x2
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # U: coeff of A_ij in M_k
        # V: coeff of B_kl in M_k
        # W: coeff of M_k in C_ij
        
        # A local indices: (i_l, j_l) -> global i = 2*i_b + i_l, j = 2*j_b + j_l
        # B local indices: (j_l, k_l) -> global j = 2*j_b + j_l, k = ...
        # C local indices: (k_l, i_l) -> global k = ..., i = 2*i_b + i_l
        
        # For 2x2x2, B is 2x2.
        # We need to know which 2x2 block of B this is.
        # In our partition, B has col blocks of size 2 and 3.
        # The 2x2x2 blocks use the first 2 cols of B (k in 0,1).
        # So k_l maps to global k = k_l.
        
        U_block = np.zeros((4, 7))
        V_block = np.zeros((4, 7))
        W_block = np.zeros((4, 7))
        
        # M1
        U_block[0,0] = 1; U_block[3,0] = 1 # a, d
        V_block[0,0] = 1; V_block[3,0] = 1 # e, h
        W_block[0,0] = 1; W_block[3,0] = 1 # C11, C22
        # M2
        U_block[2,1] = 1; U_block[3,1] = 1 # c, d
        V_block[0,1] = 1 # e
        W_block[2,1] = 1 # C21
        # M3
        U_block[0,2] = 1 # a
        V_block[1,2] = 1; V_block[3,2] = -1 # f, h
        W_block[1,2] = 1; W_block[3,2] = 1 # C12, C22
        # M4
        U_block[3,3] = 1 # d
        V_block[2,3] = 1; V_block[0,3] = -1 # g, e
        W_block[0,3] = 1; W_block[2,3] = 1 # C11, C21
        # M5
        U_block[0,4] = 1; U_block[1,4] = 1 # a, b
        V_block[3,4] = 1 # h
        W_block[0,4] = -1; W_block[1,4] = 1 # C11, C12
        # M6
        U_block[2,5] = 1; U_block[0,5] = -1 # c, a
        V_block[0,5] = 1; V_block[1,5] = 1 # e, f
        W_block[3,5] = 1 # C22
        # M7
        U_block[1,6] = 1; U_block[3,6] = -1 # b, d
        V_block[2,6] = 1; V_block[3,6] = 1 # g, h
        W_block[0,6] = 1; W_block[1,6] = 1 # C11, C12
        
        # Map to global indices
        for l_i in range(2):
            for l_j in range(2):
                g_i = 2*i_b + l_i
                g_j = 2*j_b + l_j
                u_idx = g_i*m + g_j
                for k in range(7):
                    U[u_idx, offset+k] = U_block[l_i*2+l_j, k]
        
        for l_j in range(2):
            for l_k in range(2):
                g_j = 2*j_b + l_j
                g_k = l_k # Since this is the first 2 cols of B
                v_idx = g_j*p + g_k
                for k in range(7):
                    V[v_idx, offset+k] = V_block[l_j*2+l_k, k]
        
        for l_k in range(2):
            for l_i in range(2):
                g_k = l_k
                g_i = 2*i_b + l_i
                w_idx = g_k*n + g_i
                for k in range(7):
                    W[w_idx, offset+k] = W_block[l_k*2+l_i, k]
        
        return offset + 7

    # Helper to add a 2x2x3 block (rank 11 = 7 + 4)
    def add_strassen_naive_223(i_b, j_b, offset):
        # This block uses A_{i_b, j_b} (2x2) and B_{j_b, 2..4} (2x3).
        # We decompose this into 2x2x2 (rank 7) and 2x2x1 (rank 4).
        # The 2x2x2 part uses B cols 2,3.
        # The 2x2x1 part uses B col 4.
        
        # Part 1: 2x2x2 for B cols 2,3
        # B local indices: l_k in 0,1 -> global k = 2 + l_k
        # We can reuse the Strassen logic but with shifted k.
        
        U_block = np.zeros((4, 7))
        V_block = np.zeros((4, 7))
        W_block = np.zeros((4, 7))
        
        # Same Strassen coefficients as before
        U_block[0,0] = 1; U_block[3,0] = 1
        V_block[0,0] = 1; V_block[3,0] = 1
        W_block[0,0] = 1; W_block[3,0] = 1
        U_block[2,1] = 1; U_block[3,1] = 1
        V_block[0,1] = 1
        W_block[2,1] = 1
        U_block[0,2] = 1
        V_block[1,2] = 1; V_block[3,2] = -1
        W_block[1,2] = 1; W_block[3,2] = 1
        U_block[3,3] = 1
        V_block[2,3] = 1; V_block[0,3] = -1
        W_block[0,3] = 1; W_block[2,3] = 1
        U_block[0,4] = 1; U_block[1,4] = 1
        V_block[3,4] = 1
        W_block[0,4] = -1; W_block[1,4] = 1
        U_block[2,5] = 1; U_block[0,5] = -1
        V_block[0,5] = 1; V_block[1,5] = 1
        W_block[3,5] = 1
        U_block[1,6] = 1; U_block[3,6] = -1
        V_block[2,6] = 1; V_block[3,6] = 1
        W_block[0,6] = 1; W_block[1,6] = 1
        
        for l_i in range(2):
            for l_j in range(2):
                g_i = 2*i_b + l_i
                g_j = 2*j_b + l_j
                u_idx = g_i*m + g_j
                for k in range(7):
                    U[u_idx, offset+k] = U_block[l_i*2+l_j, k]
        
        for l_j in range(2):
            for l_k in range(2):
                g_j = 2*j_b + l_j
                g_k = 2 + l_k # Shifted by 2
                v_idx = g_j*p + g_k
                for k in range(7):
                    V[v_idx, offset+k] = V_block[l_j*2+l_k, k]
        
        for l_k in range(2):
            for l_i in range(2):
                g_k = 2 + l_k
                g_i = 2*i_b + l_i
                w_idx = g_k*n + g_i
                for k in range(7):
                    W[w_idx, offset+k] = W_block[l_k*2+l_i, k]
        
        offset += 7
        
        # Part 2: 2x2x1 for B col 4
        # B local index: l_k = 0 -> global k = 4
        # Naive multiplication:
        # C11 = a*g + b*h
        # C21 = c*g + d*h
        # Here g is B[0,4], h is B[1,4].
        # M8 = a*B[0,4]
        # M9 = b*B[1,4]
        # M10 = c*B[0,4]
        # M11 = d*B[1,4]
        
        U_block2 = np.zeros((4, 4))
        V_block2 = np.zeros((4, 4))
        W_block2 = np.zeros((4, 4))
        
        # M8
        U_block2[0,0] = 1
        V_block2[0,0] = 1 # B[0,4]
        W_block2[0,0] = 1 # C11
        # M9
        U_block2[1,1] = 1
        V_block2[1,1] = 1 # B[1,4]
        W_block2[0,1] = 1 # C11
        # M10
        U_block2[2,2] = 1
        V_block2[0,2] = 1 # B[0,4]
        W_block2[1,2] = 1 # C21
        # M11
        U_block2[3,3] = 1
        V_block2[1,3] = 1 # B[1,4]
        W_block2[1,3] = 1 # C21
        
        for l_i in range(2):
            for l_j in range(2):
                g_i = 2*i_b + l_i
                g_j = 2*j_b + l_j
                u_idx = g_i*m + g_j
                for k in range(4):
                    U[u_idx, offset+k] = U_block2[l_i*2+l_j, k]
        
        for l_j in range(2):
            # l_k is always 0 for the 2x2x1 part
            g_j = 2*j_b + l_j
            g_k = 4
            v_idx = g_j*p + g_k
            for k in range(4):
                V[v_idx, offset+k] = V_block2[l_j*2+0, k]
        
        for l_k in range(2):
            # l_k is always 0 for the 2x2x1 part
            g_k = 4
            for l_i in range(2):
                g_i = 2*i_b + l_i
                w_idx = g_k*n + g_i
                for k in range(4):
                    W[w_idx, offset+k] = W_block2[l_k*2+l_i, k]
        
        return offset + 4

    # Add the 4 blocks of 2x2x2
    col_offset = add_strassen_222(0, 0, 0, col_offset) # A00 B00
    col_offset = add_strassen_222(0, 1, 0, col_offset) # A01 B10
    col_offset = add_strassen_222(1, 0, 0, col_offset) # A10 B00
    col_offset = add_strassen_222(1, 1, 0, col_offset) # A11 B10
    
    # Add the 4 blocks of 2x2x3
    col_offset = add_strassen_naive_223(0, 0, col_offset) # A00 B01
    col_offset = add_strassen_naive_223(0, 1, col_offset) # A01 B11
    col_offset = add_strassen_naive_223(1, 0, col_offset) # A10 B01
    col_offset = add_strassen_naive_223(1, 1, col_offset) # A11 B11
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
