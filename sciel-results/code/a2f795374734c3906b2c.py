import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Constructs the factor matrices for a rank-28 decomposition of matrix multiplication.
    Dimensions: n x m times m x p.
    Returns a dictionary with keys 'U', 'V', 'W'.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    
    This implementation uses a block-based approach:
    - Decompose the 2x3 matrix A into a 2x2 block (cols 0,1) and a 2x1 block (col 2).
    - Decompose the 3x5 matrix B into 2x2 blocks (rows 0,1; cols 0,1 and cols 2,3) and a 2x1 block (rows 0,1; col 4),
      plus a 1x5 block (row 2).
    
    Rank breakdown:
    - Block 1 (A[0:2,0:2] * B[0:2,0:2]): Strassen rank 7.
    - Block 2 (A[0:2,0:2] * B[0:2,2:4]): Strassen rank 7.
    - Block 3 (A[0:2,0:2] * B[0:2,4:5]): Schoolbook rank 4 (2*2*1).
    - Block 4 (A[0:2,2:3] * B[2:3,0:5]): Schoolbook rank 10 (2*1*5).
    Total Rank = 7 + 7 + 4 + 10 = 28.
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add Strassen block for 2x2 * 2x2
    def add_strassen_block(rank_offset, j_offset, k_offset):
        # A vars
        a_idx = 0 * m + j_offset
        b_idx = 0 * m + (j_offset + 1)
        c_idx = 1 * m + j_offset
        d_idx = 1 * m + (j_offset + 1)
        
        # B vars
        e_idx = j_offset * p + k_offset
        f_idx = j_offset * p + (k_offset + 1)
        g_idx = (j_offset + 1) * p + k_offset
        h_idx = (j_offset + 1) * p + (k_offset + 1)
        
        # Ranks
        r0 = rank_offset
        r1 = rank_offset + 1
        r2 = rank_offset + 2
        r3 = rank_offset + 3
        r4 = rank_offset + 4
        r5 = rank_offset + 5
        r6 = rank_offset + 6
        
        # U (L_A)
        U[a_idx, r0] += 1; U[d_idx, r0] += 1
        U[c_idx, r1] += 1; U[d_idx, r1] += 1
        U[a_idx, r2] += 1
        U[d_idx, r3] += 1
        U[a_idx, r4] += 1; U[b_idx, r4] += 1
        U[c_idx, r5] += 1; U[a_idx, r5] -= 1
        U[b_idx, r6] += 1; U[d_idx, r6] -= 1
        
        # V (L_B)
        V[e_idx, r0] += 1; V[h_idx, r0] += 1
        V[e_idx, r1] += 1
        V[f_idx, r2] += 1; V[h_idx, r2] -= 1
        V[g_idx, r3] += 1; V[e_idx, r3] -= 1
        V[h_idx, r4] += 1
        V[e_idx, r5] += 1; V[f_idx, r5] += 1
        V[g_idx, r6] += 1; V[h_idx, r6] += 1
        
        # W (Coefficients)
        # C_00 (i=0, k=k_offset) -> idx k_offset*n + 0
        W[k_offset * n + 0, r0] += 1
        W[k_offset * n + 0, r3] += 1
        W[k_offset * n + 0, r4] -= 1
        W[k_offset * n + 0, r6] += 1
        
        # C_01 (i=0, k=k_offset+1) -> idx (k_offset+1)*n + 0
        W[(k_offset + 1) * n + 0, r2] += 1
        W[(k_offset + 1) * n + 0, r4] += 1
        
        # C_10 (i=1, k=k_offset) -> idx k_offset*n + 1
        W[k_offset * n + 1, r1] += 1
        W[k_offset * n + 1, r3] += 1
        
        # C_11 (i=1, k=k_offset+1) -> idx (k_offset+1)*n + 1
        W[(k_offset + 1) * n + 1, r0] += 1
        W[(k_offset + 1) * n + 1, r1] -= 1
        W[(k_offset + 1) * n + 1, r2] += 1
        W[(k_offset + 1) * n + 1, r5] += 1

    # Block 1: j=0,1, k=0,1 (2x2 * 2x2)
    add_strassen_block(0, 0, 0)
    
    # Block 2: j=0,1, k=2,3 (2x2 * 2x2)
    add_strassen_block(7, 0, 2)
    
    # Block 3: j=0,1, k=4 (2x2 * 2x1)
    r0 = 14
    r1 = 15
    r2 = 16
    r3 = 17
    
    U[0, r0] += 1 # a
    U[1, r1] += 1 # b
    U[3, r2] += 1 # c
    U[4, r3] += 1 # d
    
    V[4, r0] += 1 # e
    V[9, r1] += 1 # g
    V[4, r2] += 1 # e
    V[9, r3] += 1 # g
    
    # C_04 (i=0, k=4) -> idx 4*2 + 0 = 8
    W[8, r0] += 1
    W[8, r1] += 1
    
    # C_14 (i=1, k=4) -> idx 4*2 + 1 = 9
    W[9, r2] += 1
    W[9, r3] += 1
    
    # Block 4: j=2, k=0..4 (2x1 * 1x5)
    rank_offset = 18
    for k in range(5):
        r_u0 = rank_offset
        r_u1 = rank_offset + 1
        rank_offset += 2
        
        U[2, r_u0] += 1 # u0 = A_02
        U[5, r_u1] += 1 # u1 = A_12
        
        V[10 + k, r_u0] += 1 # v_k = B_2k
        V[10 + k, r_u1] += 1 # v_k = B_2k
        
        # C_0k (i=0, k=k) -> idx k*2 + 0
        W[k * n + 0, r_u0] += 1
        # C_1k (i=1, k=k) -> idx k*2 + 1
        W[k * n + 1, r_u1] += 1
        
    # Convert numpy arrays to lists of lists to match the expected interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
