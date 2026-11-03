import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    rank = 28
    
    # Build factor matrices as numpy arrays for efficient construction
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # Inputs: A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # Products:
    # P1 = (a00 + a11) * (b00 + b11)
    # P2 = (a10 + a11) * b00
    # P3 = a00 * (b01 - b11)
    # P4 = a11 * (b10 - b00)
    # P5 = (a00 + a01) * b11
    # P6 = (a10 - a00) * (b00 + b01)
    # P7 = (a01 - a11) * (b10 + b11)
    #
    # Output C = [[P1 - P4 + P5 + P6, P3 + P5],
    #             [P2 + P4,            P1 - P2 + P3 + P7]]
    
    # Strassen terms: (U_coeffs, V_coeffs, W_coeffs)
    # Each is a dict mapping local index -> coefficient
    # Local indices for A block: 0=a00, 1=a01, 2=a10, 3=a11
    # Local indices for B block: 0=b00, 1=b01, 2=b10, 3=b11
    # Local indices for C block: 0=c00, 1=c01, 2=c10, 3=c11
    
    strassen_terms = [
        # P1: (a00+a11)(b00+b11) contributes to c00(+1), c11(-1), c10(-1)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 2: -1, 3: -1}),
        # P2: (a10+a11)b00 contributes to c10(+1), c11(-1)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3: a00(b01-b11) contributes to c01(+1), c11(+1)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # P4: a11(b10-b00) contributes to c00(-1), c10(+1)
        ({3: 1}, {2: 1, 0: -1}, {0: -1, 2: 1}),
        # P5: (a00+a01)b11 contributes to c00(+1), c01(+1)
        ({0: 1, 1: 1}, {3: 1}, {0: 1, 1: 1}),
        # P6: (a10-a00)(b00+b01) contributes to c00(+1)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {0: 1}),
        # P7: (a01-a11)(b10+b11) contributes to c11(+1)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {3: 1}),
    ]
    
    # Define 4 blocks of the 2x4 * 4x4 multiplication
    # Each block is a 2x2 * 2x2 subproblem
    # A is 2x4: rows 0-1, cols 0-3
    # B is 4x4: rows 0-3, cols 0-3
    # C is 4x2: rows 0-3, cols 0-1
    # 
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2]  (top-left of A * top-left of B)
    # C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4]  (top-left of A * top-right of B)
    # C[2:4, 0:2] = A[0:2, 2:4] * B[2:4, 0:2]  (bottom-left of A * bottom-left of B)
    # C[2:4, 2:4] = A[0:2, 2:4] * B[2:4, 2:4]  (bottom-left of A * bottom-right of B)
    #
    # Wait, let me reconsider the block structure.
    # A is 2x4, B is 4x4, C = A*B is 2x4.
    # 
    # Actually, let me re-examine. The output C is n x p = 2 x 4.
    # So C has 2 rows and 4 columns.
    #
    # We can decompose this as:
    # C[:, 0:2] = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :]
    # C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    #
    # That gives 4 subproblems, each 2x2 * 2x2.
    #
    # Subproblem 1: A_L (cols 0,1) * B_TL (rows 0,1; cols 0,1) -> C_L_top (rows 0,1; cols 0,1)
    #   But wait, C is 2x4, so C[0:2, 0:2] is a 2x2 block.
    #   A[0:2, 0:2] is 2x2, B[0:2, 0:2] is 2x2. Product is 2x2. Correct.
    #
    # Subproblem 2: A_R (cols 2,3) * B_BL (rows 2,3; cols 0,1) -> C_L_bot (rows 0,1; cols 0,1)
    #   A[0:2, 2:4] is 2x2, B[2:4, 0:2] is 2x2. Product is 2x2.
    #   This adds to C[0:2, 0:2].
    #
    # Subproblem 3: A_L (cols 0,1) * B_TR (rows 0,1; cols 2,3) -> C_R_top (rows 0,1; cols 2,3)
    #   A[0:2, 0:2] is 2x2, B[0:2, 2:4] is 2x2. Product is 2x2.
    #   This goes to C[0:2, 2:4].
    #
    # Subproblem 4: A_R (cols 2,3) * B_BR (rows 2,3; cols 2,3) -> C_R_bot (rows 0,1; cols 2,3)
    #   A[0:2, 2:4] is 2x2, B[2:4, 2:4] is 2x2. Product is 2x2.
    #   This adds to C[0:2, 2:4].
    
    # Block definitions:
    # (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
    # C_col_offset is the column offset in the output C (which is 2x4, so cols 0-3)
    # But wait, C is indexed as W[k*n + i, r] where k is column, i is row.
    # So W has shape (p*n, rank) = (4*2, rank) = (8, rank).
    # W[k*n + i, r] corresponds to C[i, k].
    
    blocks = [
        # (a_col_off, b_row_off, b_col_off, c_col_off)
        (0, 0, 0, 0),  # Block 1: A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        (2, 2, 0, 0),  # Block 2: A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
        (0, 0, 2, 2),  # Block 3: A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
        (2, 2, 2, 2),  # Block 4: A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    ]
    
    col_idx = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # Fill U
            # Local A indices: 0->(0,a_off), 1->(0,a_off+1), 2->(1,a_off), 3->(1,a_off+1)
            # Global U index = row * m + col
            for local_idx, val in u_coeffs.items():
                if local_idx == 0:
                    g_idx = 0 * m + a_off
                elif local_idx == 1:
                    g_idx = 0 * m + a_off + 1
                elif local_idx == 2:
                    g_idx = 1 * m + a_off
                elif local_idx == 3:
                    g_idx = 1 * m + a_off + 1
                U[g_idx, col_idx] = val
            
            # Fill V
            # Local B indices: 0->(b_row_off, b_col_off), 1->(b_row_off, b_col_off+1),
            #                 2->(b_row_off+1, b_col_off), 3->(b_row_off+1, b_col_off+1)
            # Global V index = row * p + col
            for local_idx, val in v_coeffs.items():
                if local_idx == 0:
                    g_idx = b_row_off * p + b_col_off
                elif local_idx == 1:
                    g_idx = b_row_off * p + b_col_off + 1
                elif local_idx == 2:
                    g_idx = (b_row_off + 1) * p + b_col_off
                elif local_idx == 3:
                    g_idx = (b_row_off + 1) * p + b_col_off + 1
                V[g_idx, col_idx] = val
            
            # Fill W
            # Local C indices: 0->(0, c_off), 1->(0, c_off+1), 2->(1, c_off), 3->(1, c_off+1)
            # Global W index = col * n + row (trace convention: W[k*n+i, r] for C[i, k])
            for local_idx, val in w_coeffs.items():
                if local_idx == 0:
                    g_idx = c_off * n + 0
                elif local_idx == 1:
                    g_idx = (c_off + 1) * n + 0
                elif local_idx == 2:
                    g_idx = c_off * n + 1
                elif local_idx == 3:
                    g_idx = (c_off + 1) * n + 1
                W[g_idx, col_idx] = val
            
            col_idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Convert to lists of lists (the expected format)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure all entries are integers (they should be, since all coefficients are ±1)
    for i in range(len(U_list)):
        for j in range(len(U_list[i])):
            U_list[i][j] = int(round(U_list[i][j]))
    for i in range(len(V_list)):
        for j in range(len(V_list[i])):
            V_list[i][j] = int(round(V_list[i][j]))
    for i in range(len(W_list)):
        for j in range(len(W_list[i])):
            W_list[i][j] = int(round(W_list[i][j]))
    
    return {"U": U_list, "V": V_list, "W": W_list}
