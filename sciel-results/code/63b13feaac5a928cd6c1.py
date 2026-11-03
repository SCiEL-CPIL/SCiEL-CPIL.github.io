"""Program for finding a rank-minimal exact tensor decomposition for 2x4*4x5 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-38 tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    Strategy: Split the m=4 dimension into two blocks of size 2.
    The problem decomposes into two independent 2x2 * 2x5 multiplications.
    For each 2x2 * 2x5 block, we use a rank-19 decomposition:
      - Strassen (rank 7) for the first 2x2 sub-block (columns 0,1 of B)
      - Schoolbook (rank 12) for the remaining 2x3 sub-block (columns 2,3,4 of B)
    Total rank = 19 + 19 = 38.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 38
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # We'll build the decomposition block by block.
    # Block 1: j in {0, 1}
    # Block 2: j in {2, 3}
    
    # For each block, we need rank-19 factors for 2x2 * 2x5.
    # Let's define a helper to generate rank-19 factors for 2x2 * 2x5.
    
    def rank19_2x2_2x5():
        """
        Returns (U_block, V_block, W_block) for 2x2 * 2x5 multiplication.
        U_block: shape (4, 19)  -- rows indexed by (i, j_local) where i in [0,1], j_local in [0,1]
        V_block: shape (10, 19) -- rows indexed by (j_local, k) where j_local in [0,1], k in [0,4]
        W_block: shape (10, 19) -- rows indexed by (k, i) where k in [0,4], i in [0,1]
        
        The decomposition uses:
        - 7 terms from Strassen for the 2x2 * 2x2 part (k=0,1)
        - 12 terms from schoolbook for the 2x2 * 2x3 part (k=2,3,4)
        """
        U_b = np.zeros((4, 19))
        V_b = np.zeros((10, 19))
        W_b = np.zeros((10, 19))
        
        # --- Strassen part for k in {0, 1} ---
        # A is 2x2: A[i, j_local]
        # B is 2x2: B[j_local, k]
        # C is 2x2: C[i, k] = sum_j A[i,j] B[j,k]
        #
        # Strassen's 7 multiplications:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B12
        # M3 = A11 * (B21 - B11)
        # M4 = A22 * (B12 - B22)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # U rows: (i, j_local) -> index i*2 + j_local
        # V rows: (j_local, k) -> index j_local*2 + k
        # W rows: (k, i) -> index k*2 + i
        
        # Term 0: M1 = (A11 + A22) * (B11 + B22)
        # A11: i=0, j=0 -> U[0, 0] = 1
        # A22: i=1, j=1 -> U[3, 0] = 1
        # B11: j=0, k=0 -> V[0, 0] = 1
        # B22: j=1, k=1 -> V[3, 0] = 1
        # W: contributes to C11 (k=0,i=0) and C22 (k=1,i=1)
        # C11 = M1 + ... -> W[0, 0] = 1
        # C22 = M1 - ... -> W[3, 0] = 1
        U_b[0, 0] = 1; U_b[3, 0] = 1
        V_b[0, 0] = 1; V_b[3, 0] = 1
        W_b[0, 0] = 1; W_b[3, 0] = 1
        
        # Term 1: M2 = (A21 + A22) * B12
        # A21: i=1, j=0 -> U[2, 1] = 1
        # A22: i=1, j=1 -> U[3, 1] = 1
        # B12: j=0, k=1 -> V[1, 1] = 1
        # W: C21 = M2 + ... -> W[2, 1] = 1 (k=1, i=0? No, C21 means i=1, k=0? 
        # Wait, C[i,k]. C21 is i=1, k=1? No, standard indexing C[i,j] where i is row, j is col.
        # Let's use C[i, k]. C21 means i=1, k=1? No, C21 usually means row 2, col 1 (1-indexed).
        # In 0-indexed: C[1, 0]? No, let's stick to C[i, k].
        # C[i, k] = sum_j A[i, j] B[j, k].
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
        # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
        #
        # Strassen:
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        #
        # So:
        # W index is k*2 + i.
        # C[0,0]: k=0, i=0 -> W[0]
        # C[0,1]: k=1, i=0 -> W[2]
        # C[1,0]: k=0, i=1 -> W[1]
        # C[1,1]: k=1, i=1 -> W[3]
        
        # Let's redo W assignments.
        W_b[0, 0] = 1   # C[0,0] gets +M1
        W_b[1, 0] = 1   # C[1,1] gets +M1
        
        W_b[1, 1] = 1   # C[1,0] gets +M2
        
        W_b[2, 2] = 1   # C[0,1] gets +M3
        
        W_b[0, 3] = 1   # C[0,0] gets +M4
        W_b[1, 3] = 1   # C[1,0] gets +M4
        
        W_b[2, 4] = -1  # C[0,1] gets -M5
        W_b[0, 4] = -1  # C[0,0] gets -M5
        
        W_b[3, 5] = 1   # C[1,1] gets +M6
        
        W_b[0, 6] = 1   # C[0,0] gets +M7
        
        # Now fill U and V for terms 1-6
        # Term 1: M2 = (A21 + A22) * B12
        U_b[2, 1] = 1; U_b[3, 1] = 1
        V_b[1, 1] = 1
        
        # Term 2: M3 = A11 * (B21 - B11)
        U_b[0, 2] = 1
        V_b[2, 2] = 1; V_b[0, 2] = -1
        
        # Term 3: M4 = A22 * (B12 - B22)
        U_b[3, 3] = 1
        V_b[1, 3] = 1; V_b[3, 3] = -1
        
        # Term 4: M5 = (A11 + A12) * B22
        U_b[0, 4] = 1; U_b[1, 4] = 1
        V_b[3, 4] = 1
        
        # Term 5: M6 = (A21 - A11) * (B11 + B12)
        U_b[2, 5] = 1; U_b[0, 5] = -1
        V_b[0, 5] = 1; V_b[1, 5] = 1
        
        # Term 6: M7 = (A12 - A22) * (B21 + B22)
        U_b[1, 6] = 1; U_b[3, 6] = -1
        V_b[2, 6] = 1; V_b[3, 6] = 1
        
        # --- Schoolbook part for k in {2, 3, 4} ---
        # 2x2 * 2x3 schoolbook: 12 terms.
        # C[i, k] = A[i, 0] * B[0, k] + A[i, 1] * B[1, k]
        # For each (i, j, k), one term.
        # i in {0,1}, j in {0,1}, k in {2,3,4}
        
        term_idx = 7
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    k_global = k + 2  # k is 2,3,4
                    # U row: i*2 + j
                    U_b[i*2 + j, term_idx] = 1
                    # V row: j*2 + k_global
                    V_b[j*2 + k_global, term_idx] = 1
                    # W row: k_global*2 + i
                    W_b[k_global*2 + i, term_idx] = 1
                    term_idx += 1
        
        return U_b, V_b, W_b
    
    # Generate rank-19 factors for one block
    U_b, V_b, W_b = rank19_2x2_2x5()
    
    # Place Block 1 (j in {0,1}) into the global matrices
    # Global U row index: i*m + j = i*4 + j
    # Block U row index: i*2 + j_local
    # j_local = j (since j is 0,1)
    for i in range(2):
        for j_local in range(2):
            global_u_row = i * 4 + j_local
            block_u_row = i * 2 + j_local
            U[global_u_row, 0:19] = U_b[block_u_row, :]
    
    # Global V row index: j*p + k = j*5 + k
    # Block V row index: j_local*2 + k
    for j_local in range(2):
        for k in range(5):
            global_v_row = j_local * 5 + k
            block_v_row = j_local * 2 + k
            V[global_v_row, 0:19] = V_b[block_v_row, :]
    
    # Global W row index: k*n + i = k*2 + i
    # Block W row index: k*2 + i
    for k in range(5):
        for i in range(2):
            global_w_row = k * 2 + i
            block_w_row = k * 2 + i
            W[global_w_row, 0:19] = W_b[block_w_row, :]
    
    # Place Block 2 (j in {2,3}) into the global matrices
    # j_local = j - 2
    for i in range(2):
        for j_local in range(2):
            j = j_local + 2
            global_u_row = i * 4 + j
            block_u_row = i * 2 + j_local
            U[global_u_row, 19:38] = U_b[block_u_row, :]
    
    for j_local in range(2):
        j = j_local + 2
        for k in range(5):
            global_v_row = j * 5 + k
            block_v_row = j_local * 2 + k
            V[global_v_row, 19:38] = V_b[block_v_row, :]
    
    for k in range(5):
        for i in range(2):
            global_w_row = k * 2 + i
            block_w_row = k * 2 + i
            W[global_w_row, 19:38] = W_b[block_w_row, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
