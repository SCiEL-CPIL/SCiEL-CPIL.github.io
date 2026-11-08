"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # 1. Construct Strassen Decomposition for 2x2 matrices
    # Indices for U: 0..3 (i*2+j), V: 0..3 (j*2+k), W: 0..3 (k*2+i)
    # Strassen 7 terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    strassen_U = [
        [1, 0, 0, 0, 1, 0, 0], # A00
        [0, 0, 0, 0, 0, 1, 0], # A01
        [0, 1, 0, 0, 0, -1, 0],# A10
        [1, 1, 0, 1, 0, 0, -1] # A11
    ]
    strassen_V = [
        [1, 1, 0, 0, 0, 1, 1], # B00
        [0, 0, 1, -1, 0, 0, 0],# B01
        [0, 0, 1, 0, 0, 1, 0], # B10
        [1, 0, -1, -1, 1, 0, 1]# B11
    ]
    strassen_W = [
        [1, 0, 0, 1],  # C00 (M1, M4, M5, M7) -> signs: +, -, +, +? 
        # Let's re-verify C mapping.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # W row index is k*2+i.
        # k=0, i=0 -> C00. M1(1), M4(-1), M5(1), M7(1). Wait, C00 = M1+M4-M5+M7. So coeffs: [1, 0, 0, 1, -1, 0, 1]
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01 (M3, M5) -> [0, 0, 1, 0, 1, 0, 0]
        [0, 1, 0, 1, 0, 0, 0],  # C10 (M2, M4) -> [0, 1, 0, 1, 0, 0, 0]
        [1, -1, 1, 0, 0, 1, 0]  # C11 (M1, M2, M3, M6) -> [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # 2. Construct Block Decomposition for 6x6 matrices
    # Partition 6x6 into 3x3 blocks of size 2x2.
    # Block indices I, J, K in 0..2.
    # Total Rank = 3 * 3 * 3 * 7 = 189.
    
    total_rank = 189
    U = np.zeros((36, total_rank), dtype=np.int8)
    V = np.zeros((36, total_rank), dtype=np.int8)
    W = np.zeros((36, total_rank), dtype=np.int8)
    
    col_offset = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                # For each block multiplication (A[I,J] * B[J,K] -> C[I,K])
                # We use 7 Strassen terms.
                for s in range(7):
                    # U index: Block I, J. Row within block: 0..3.
                    # Global row: I*4 + (local_row)
                    # V index: Block J, K. Col within block: 0..3.
                    # Global col: J*4 + (local_col)
                    # W index: Block I, K. Row within block: 0..3.
                    # Global row: K*4 + (local_row) ? 
                    # Wait, W is output C. C is 6x6.
                    # C index (i, k). Block I, K.
                    # W dimension is p*n = 36.
                    # W row index: k*6 + i.
                    # Block I, K corresponds to i in [I*2, I*2+1], k in [K*2, K*2+1].
                    # Local row in W block: l_w. Global W row: K*4 + l_w? No.
                    # W row is k*n + i.
                    # If block is I, K:
                    # i = I*2 + r_i (r_i in 0,1)
                    # k = K*2 + r_k (r_k in 0,1)
                    # W row = k*6 + i = (K*2 + r_k)*6 + (I*2 + r_i) = 12*K + 6*r_k + 2*I + r_i.
                    
                    # U row: i*m + j.
                    # i = I*2 + r_i
                    # j = J*2 + r_j
                    # U row = (I*2 + r_i)*6 + (J*2 + r_j) = 12*I + 6*r_i + 2*J + r_j.
                    
                    # V col: j*p + k.
                    # j = J*2 + r_j
                    # k = K*2 + r_k
                    # V col = (J*2 + r_j)*6 + (K*2 + r_k) = 12*J + 6*r_j + 2*K + r_k.
                    
                    # Strassen local indices:
                    # U_local = r_i*2 + r_j (0..3)
                    # V_local = r_j*2 + r_k (0..3)
                    # W_local = r_k*2 + r_i (0..3)
                    
                    # Loop over local indices
                    for r_i in range(2):
                        for r_j in range(2):
                            u_local = r_i * 2 + r_j
                            u_val = strassen_U[u_local][s]
                            if u_val != 0:
                                u_row = 12*I + 6*r_i + 2*J + r_j
                                U[u_row, col_offset] = u_val
                            
                            # We need to find r_k such that V_local and W_local match?
                            # No, V and W depend on r_k.
                            # The term s involves specific combinations of A and B entries.
                            # The decomposition is sum_s U_s * V_s * W_s.
                            # For a fixed s, U_s is a vector over (i,j), V_s over (j,k), W_s over (k,i).
                            # So for fixed s, we iterate over all local i,j,k.
                            
    # The nested loop above was slightly messy. Let's rebuild cleanly.
    
    U = np.zeros((36, total_rank), dtype=np.int8)
    V = np.zeros((36, total_rank), dtype=np.int8)
    W = np.zeros((36, total_rank), dtype=np.int8)
    
    col_offset = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for s in range(7):
                    # Iterate over all local indices in the 2x2 blocks
                    for r_i in range(2):
                        for r_j in range(2):
                            for r_k in range(2):
                                # Local indices
                                u_local = r_i * 2 + r_j
                                v_local = r_j * 2 + r_k
                                w_local = r_k * 2 + r_i
                                
                                u_val = strassen_U[u_local][s]
                                v_val = strassen_V[v_local][s]
                                w_val = strassen_W[w_local][s]
                                
                                # Global indices
                                u_row = 12*I + 6*r_i + 2*J + r_j
                                v_col = 12*J + 6*r_j + 2*K + r_k
                                w_row = 12*K + 6*r_k + 2*I + r_i
                                
                                if u_val != 0:
                                    U[u_row, col_offset] = u_val
                                if v_val != 0:
                                    V[v_col, col_offset] = v_val
                                if w_val != 0:
                                    W[w_row, col_offset] = w_val
                    col_offset += 1
                    
    # Verify correctness quickly?
    # The task requires integer/half-integer entries. int8 is fine.
    # Return lists of lists as per interface? "factor matrices of shapes... with integer/half-integer entries"
    # The seed returns lists of lists. I will return lists of lists to be safe, or numpy arrays if accepted.
    # "Interface: construct() -> {'U', 'V', 'W'}: factor matrices... "
    # Seed uses lists. I'll convert to lists of lists.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
