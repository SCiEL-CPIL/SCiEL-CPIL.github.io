import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for M_{2,2,2}
    # U: 4x7, V: 4x7, W: 4x7
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ], dtype=float)
    
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ], dtype=float)
    
    W_s = np.array([
        [1, 0, 0, -1, 1, 0, -1], # C00 (c=0)
        [0, 1, 0, 1, -1, 1, 0],  # C10 (c=1)
        [0, 0, 1, 0, 0, 0, 1],   # C01 (c=2)
        [1, -1, -1, 0, 0, -1, 0] # C11 (c=3)
    ], dtype=float)
    
    # Construct M_{2,2,8} decomposition (Rank 28)
    # A is 2x2, B is 2x8, C is 8x2 (transposed)
    # U_m: 4x28, V_m: 16x28, W_m: 16x28
    U_m = np.zeros((4, 28), dtype=float)
    V_m = np.zeros((16, 28), dtype=float)
    W_m = np.zeros((16, 28), dtype=float)
    
    for q in range(4):
        # q corresponds to block of 2 columns in B, 2 rows in C
        # B indices: j in {0,1}, k in {2q, 2q+1}
        # V index: j*8 + k
        # W index: k*2 + i (since C is 8x2, row k, col i)
        # Wait, W index in M_{2,2,8} is c = k*2 + i.
        # k is row of C (which is col of B). i is col of C (which is row of A).
        
        for t in range(7):
            col = q * 7 + t
            
            # U part: A is 2x2. Index a = i*2 + j.
            # In M_{2,2,8}, A is the same 2x2 matrix.
            U_m[:, col] = U_s[:, t]
            
            # V part: B is 2x8. Index b = j*8 + k.
            # j in {0,1}. k in {2q, 2q+1}.
            # V_s index v_idx = j*2 + k_local.
            # k_local in {0,1} corresponds to k = 2q + k_local.
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * q + k_local
                    b_idx = j * 8 + k_global
                    v_idx = j * 2 + k_local
                    V_m[b_idx, col] = V_s[v_idx, t]
            
            # W part: C is 8x2. Index c = k*2 + i.
            # k in {2q, 2q+1}. i in {0,1}.
            # W_s index w_idx = k_local*2 + i.
            for k_local in range(2):
                k_global = 2 * q + k_local
                for i in range(2):
                    c_idx = k_global * 2 + i
                    w_idx = k_local * 2 + i
                    W_m[c_idx, col] = W_s[w_idx, t]
                    
    # Construct M_{2,4,8} decomposition (Rank 56)
    # A is 2x4. B is 4x8. C is 8x2.
    # Split A into A_L (cols 0,1) and A_R (cols 2,3).
    # Split B into B_T (rows 0,1) and B_B (rows 2,3).
    # C = A_L B_T + A_R B_B.
    
    U_full = np.zeros((8, 56), dtype=float) # n*m = 2*4 = 8
    V_full = np.zeros((32, 56), dtype=float) # m*p = 4*8 = 32
    W_full = np.zeros((16, 56), dtype=float) # p*n = 8*2 = 16
    
    # Part 1: A_L B_T
    # A_L indices: a = i*4 + j, j in {0,1}.
    # B_T indices: b = j*8 + k, j in {0,1}.
    # C indices: c = k*2 + i.
    
    for t in range(28):
        col = t
        # U
        for i in range(2):
            for j in range(2):
                a_idx = i * 4 + j
                a_local = i * 2 + j
                U_full[a_idx, col] = U_m[a_local, t]
                
        # V
        for j in range(2):
            for k in range(8):
                b_idx = j * 8 + k
                b_local = j * 8 + k
                V_full[b_idx, col] = V_m[b_local, t]
                
        # W
        for k in range(8):
            for i in range(2):
                c_idx = k * 2 + i
                c_local = k * 2 + i
                W_full[c_idx, col] = W_m[c_local, t]
                
    # Part 2: A_R B_B
    # A_R indices: a = i*4 + j, j in {2,3}.
    # B_B indices: b = j*8 + k, j in {2,3}.
    # C indices: c = k*2 + i.
    
    for t in range(28):
        col = 28 + t
        # U
        for i in range(2):
            for j in range(2):
                j_global = j + 2
                a_idx = i * 4 + j_global
                a_local = i * 2 + j
                U_full[a_idx, col] = U_m[a_local, t]
                
        # V
        for j in range(2):
            j_global = j + 2
            for k in range(8):
                b_idx = j_global * 8 + k
                b_local = j * 8 + k
                V_full[b_idx, col] = V_m[b_local, t]
                
        # W
        for k in range(8):
            for i in range(2):
                c_idx = k * 2 + i
                c_local = k * 2 + i
                W_full[c_idx, col] = W_m[c_local, t]

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
