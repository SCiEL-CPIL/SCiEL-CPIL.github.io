import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin

    # 1. Define Strassen decomposition for 2x2x2
    # U: 2x2x7, V: 2x2x7, W: 2x2x7
    # Indices: U[i, j, r], V[j, k, r], W[k, i, r]
    
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    W2[0, 1, 0] = -1; W2[1, 0, 0] = 1
    # Wait, W is for output C.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's fill W based on the reconstruction formulas.
    # W[k, i, r] is the coefficient of M_r in C[i, k].
    # C[i, k] = sum_r W[k, i, r] M_r.
    
    # M1 (r=0)
    W2[0, 0, 0] = 1 # C00
    W2[1, 1, 0] = 1 # C11
    # M2 (r=1)
    W2[0, 1, 1] = 1 # C10
    W2[1, 1, 1] = -1 # C11
    # M3 (r=2)
    W2[1, 0, 2] = 1 # C01
    W2[1, 1, 2] = 1 # C11
    # M4 (r=3)
    W2[0, 0, 3] = 1 # C00
    W2[0, 1, 3] = 1 # C10
    # M5 (r=4)
    W2[1, 0, 4] = 1 # C01
    W2[0, 0, 4] = -1 # C00
    # M6 (r=5)
    W2[1, 1, 5] = 1 # C11
    # M7 (r=6)
    W2[0, 0, 6] = 1 # C00

    # U and V for each M_r
    # M1: A00+A11, B00+B11
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    
    # M2: A10+A11, B00
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    
    # M3: A00, B01-B11
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    
    # M4: A11, B10-B00
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    
    # M5: A00+A01, B11
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    
    # M6: A10-A00, B00+B01
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    
    # M7: A01-A11, B10+B11
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1

    # 2. Construct 4x4x4 decomposition using block Strassen
    # Rank 49
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))
    
    # Mapping from local 2x2 indices to global 4x4 indices
    # Block 00: rows 0,1 cols 0,1 -> global (0,0), (0,1), (1,0), (1,1)
    # Block 01: rows 0,1 cols 2,3 -> global (0,2), (0,3), (1,2), (1,3)
    # Block 10: rows 2,3 cols 0,1 -> global (2,0), (2,1), (3,0), (3,1)
    # Block 11: rows 2,3 cols 2,3 -> global (2,2), (2,3), (3,2), (3,3)
    
    # For each of the 7 multiplications M_k (k=0..6)
    # We need to define the linear combinations of A and B blocks.
    # A_k is a 2x2 matrix of coefficients for A blocks.
    # B_k is a 2x2 matrix of coefficients for B blocks.
    
    # M1: A00+A11, B00+B11
    # A_k[i,j] is the coefficient of Block (i,j) in A_k.
    # A_k is 2x2. A_k[0,0]=1, A_k[1,1]=1.
    
    A_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2 (A10+A11)
        [[1, 0], [0, 0]], # M3 (A00)
        [[0, 0], [0, 1]], # M4 (A11)
        [[1, 1], [0, 0]], # M5 (A00+A01)
        [[-1, 0], [1, 0]],# M6 (A10-A00)
        [[0, 1], [0, -1]] # M7 (A01-A11)
    ]
    
    B_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2 (B00)
        [[0, 1], [0, -1]],# M3 (B01-B11)
        [[-1, 0], [1, 0]],# M4 (B10-B00)
        [[0, 0], [0, 1]], # M5 (B11)
        [[1, 1], [0, 0]], # M6 (B00+B01)
        [[0, 1], [0, 1]]  # M7 (B10+B11)
    ]
    
    # W_coeffs for 4x4 reconstruction
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # W4[k, i, r] where r is global rank index.
    # r = 7*k + r_local.
    
    W_block_coeffs = [
        [[1, 0], [0, 1]], # M1 contributes to C00, C11
        [[0, 0], [1, -1]],# M2 contributes to C10, C11
        [[0, 1], [1, 0]], # M3 contributes to C01, C11
        [[1, 0], [1, 0]], # M4 contributes to C00, C10
        [[-1, 1], [0, 0]],# M5 contributes to C00, C01
        [[0, 0], [0, 1]], # M6 contributes to C11
        [[1, 0], [0, 0]]  # M7 contributes to C00
    ]
    
    for k in range(7):
        for r_local in range(7):
            r_global = 7 * k + r_local
            
            # U4
            for i_local in range(2):
                for j_local in range(2):
                    u_val = U2[i_local, j_local, r_local]
                    if u_val != 0:
                        # A_k[i_local, j_local] is the coefficient for Block (i_local, j_local)
                        coeff = A_coeffs[k][i_local][j_local]
                        if coeff != 0:
                            # Global indices for Block (i_local, j_local)
                            # Row block i_local: rows 2*i_local, 2*i_local+1
                            # Col block j_local: cols 2*j_local, 2*j_local+1
                            for di in range(2):
                                for dj in range(2):
                                    I = 2 * i_local + di
                                    J = 2 * j_local + dj
                                    U4[I, J, r_global] += u_val * coeff
                                    
            # V4
            for j_local in range(2):
                for k_local in range(2):
                    v_val = V2[j_local, k_local, r_local]
                    if v_val != 0:
                        coeff = B_coeffs[k][j_local][k_local]
                        if coeff != 0:
                            for dj in range(2):
                                for dk in range(2):
                                    J = 2 * j_local + dj
                                    K = 2 * k_local + dk
                                    V4[J, K, r_global] += v_val * coeff
                                    
            # W4
            for k_local in range(2):
                for i_local in range(2):
                    w_val = W2[k_local, i_local, r_local]
                    if w_val != 0:
                        coeff = W_block_coeffs[k][i_local][k_local]
                        if coeff != 0:
                            for dk in range(2):
                                for di in range(2):
                                    K = 2 * k_local + dk
                                    I = 2 * i_local + di
                                    W4[K, I, r_global] += w_val * coeff

    # 3. Construct 4x4x8 decomposition
    # Rank 98
    U8 = np.zeros((4, 4, 98))
    V8 = np.zeros((4, 8, 98))
    W8 = np.zeros((8, 4, 98))
    
    # First 49 ranks for B_L (cols 0-3)
    for r in range(49):
        U8[:, :, r] = U4[:, :, r]
        V8[:, 0:4, r] = V4[:, :, r]
        W8[0:4, :, r] = W4[:, :, r]
        
    # Next 49 ranks for B_R (cols 4-7)
    for r in range(49):
        r_global = 49 + r
        U8[:, :, r_global] = U4[:, :, r]
        V8[:, 4:8, r_global] = V4[:, :, r]
        W8[4:8, :, r_global] = W4[:, :, r]
        
    # Convert to lists
    U_list = U8.tolist()
    V_list = V8.tolist()
    W_list = W8.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    construct()
