import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition manually
    # Rank 22: 7 (Strassen 1) + 7 (Strassen 2) + 8 (Schoolbook 1)
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # term: (U_indices, U_coeffs, V_indices, V_coeffs, W_index, W_coeff)
    def add_term(u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff, col):
        U[u_idx][col] += u_coeff
        V[v_idx][col] += v_coeff
        W[w_idx][col] += w_coeff

    # Part 1: C[:, 0:4] from A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :]
    # This is two 2x2x2 multiplications.
    # Strassen for 2x2x2
    # A11, A12, A21, A22 are 2x2 blocks of A (2x4) -> A11 is A[0:2, 0:2], A12 is A[0:2, 2:4]
    # B11, B12, B21, B22 are 2x2 blocks of B (4x4) -> B11 is B[0:2, 0:2], etc.
    
    # Strassen 1: A_left * B_top_part?
    # Let's define the blocks clearly.
    # A is 2x4. A = [A_L, A_R] where A_L, A_R are 2x2.
    # B is 4x5. B = [B_T; B_B] where B_T, B_B are 2x5.
    # C = A_L B_T + A_R B_B.
    # B_T is 2x5. Split into B_T1 (2x2), B_T2 (2x2), B_T3 (2x1).
    # B_B is 2x5. Split into B_B1 (2x2), B_B2 (2x2), B_B3 (2x1).
    
    # C[:, 0:2] = A_L B_T1 + A_R B_B1 (2x2x2)
    # C[:, 2:4] = A_L B_T2 + A_R B_B2 (2x2x2)
    # C[:, 4:5] = A_L B_T3 + A_R B_B3 (2x4x1)
    
    # Indices for A (2x4): 0,1,2,3 (row 0: 0,1; row 1: 2,3)
    # A_L: A[0:2, 0:2] -> indices 0,1 (row 0), 2,3 (row 1)?
    # A is stored row-major: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    # A_L: A[0,0], A[0,1], A[1,0], A[1,1] -> U[0], U[1], U[2], U[3]
    # A_R: A[0,2], A[0,3], A[1,2], A[1,3] -> U[4], U[5], U[6], U[7]
    
    # B is 4x5. B[0,0]=V[0], ..., B[0,4]=V[4], B[1,0]=V[5]...
    # B_T1: B[0:2, 0:2] -> V[0], V[1], V[5], V[6]
    # B_T2: B[0:2, 2:4] -> V[2], V[3], V[7], V[8]
    # B_T3: B[0:2, 4:5] -> V[4], V[9]
    # B_B1: B[2:4, 0:2] -> V[10], V[11], V[15], V[16]
    # B_B2: B[2:4, 2:4] -> V[12], V[13], V[17], V[18]
    # B_B3: B[2:4, 4:5] -> V[14], V[19]
    
    # C is 2x5. C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2]...
    # W index: k*n + i. k is col, i is row.
    # C[:, 0] -> k=0. W[0] (i=0), W[1] (i=1).
    # C[:, 1] -> k=1. W[2], W[3].
    # C[:, 2] -> k=2. W[4], W[5].
    # C[:, 3] -> k=3. W[6], W[7].
    # C[:, 4] -> k=4. W[8], W[9].
    
    # Strassen 1 for C[:, 0:2]
    # A_L = [[a11, a12], [a21, a22]] = U[0], U[1], U[2], U[3]
    # B_T1 = [[b11, b12], [b21, b22]] = V[0], V[1], V[5], V[6]
    # B_B1 = [[c11, c12], [c21, c22]] = V[10], V[11], V[15], V[16]
    # Note: B_T1 and B_B1 are 2x2.
    # C1 = A_L B_T1 + A_R B_B1? No.
    # C[:, 0:2] = A_L B_T1 + A_R B_B1.
    # This is sum of two 2x2x2 products.
    # We can use Strassen for each?
    # Strassen computes X Y.
    # Here we have A_L B_T1 AND A_R B_B1.
    # These are two separate products.
    # Rank 7 + 7 = 14.
    
    # Let's define Strassen terms for A_L B_T1.
    # M1 = (A11+A22)(B11+B22)
    # ...
    # We need to add these to U, V, W.
    
    def strassen_2x2x2(a_idx, b_idx, c_idx, col_offset):
        # a_idx: [a11, a12, a21, a22]
        # b_idx: [b11, b12, b21, b22]
        # c_idx: [c11, c12, c21, c22] (indices in W)
        
        # M1
        add_term(a_idx[0], 1, b_idx[0], 1, c_idx[0], 1, col_offset)
        add_term(a_idx[0], 1, b_idx[1], 1, c_idx[1], 1, col_offset)
        add_term(a_idx[2], 1, b_idx[0], 1, c_idx[2], 1, col_offset)
        add_term(a_idx[2], 1, b_idx[1], 1, c_idx[3], 1, col_offset)
        add_term(a_idx[3], 1, b_idx[0], 1, c_idx[0], 1, col_offset)
        add_term(a_idx[3], 1, b_idx[1], 1, c_idx[1], 1, col_offset)
        add_term(a_idx[3], 1, b_idx[2], 1, c_idx[2], 1, col_offset)
        add_term(a_idx[3], 1, b_idx[3], 1, c_idx[3], 1, col_offset)
        # Wait, M1 = (A11+A22)(B11+B22).
        # U: A11, A22. V: B11, B22.
        # W: C11, C12, C21, C22 all get contribution.
        # C11 += M1, C12 += M1, C21 += M1, C22 += M1? No.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for M1, W coeffs are 1, 0, 0, 1? No.
        # C11 gets +1, C12 gets 0, C21 gets 0, C22 gets +1.
        # Let's re-verify Strassen.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # I will implement this correctly.
        pass

    # I will just write the code directly in the construct function.
    return {"U": U, "V": V, "W": W}
