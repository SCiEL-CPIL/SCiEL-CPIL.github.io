import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    
    # Rank 56 decomposition using 8 blocks of Strassen (rank 7)
    num_blocks = p // 2  # 8 blocks
    R = num_blocks * 7   # 56
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Strassen coefficients for one 2x2 block
    # A indices: 0(A00), 1(A01), 2(A10), 3(A11)
    # B local: 0(B00), 1(B01), 2(B10), 3(B11)
    # C local: 0(C00), 1(C01), 2(C10), 3(C11)
    
    # U_coeffs[s][a_idx] = coeff
    U_coeffs = [
        [1, 0, 0, 1],      # M1
        [0, 0, 1, 1],      # M2
        [1, 0, 0, 0],      # M3
        [0, 0, 0, 1],      # M4
        [1, 1, 0, 0],      # M5
        [-1, 0, 1, 0],     # M6
        [0, 1, 0, -1]      # M7
    ]
    
    # V_coeffs[s][b_local_idx] = coeff
    V_coeffs = [
        [1, 0, 0, 1],      # M1
        [1, 0, 0, 0],      # M2
        [0, 1, 0, -1],     # M3
        [0, 0, 1, 0],      # M4
        [0, 0, 0, 1],      # M5
        [1, 1, 0, 0],      # M6
        [0, 0, 1, 1]       # M7
    ]
    
    # W_contribs[s][c_local_idx] = coeff
    # C local indices: 0(C00), 1(C01), 2(C10), 3(C11)
    W_contribs = [
        [1, 0, 0, 1],      # M1 -> C00, C11
        [0, 0, 1, -1],     # M2 -> C10, C11
        [0, 1, 0, 1],      # M3 -> C01, C11
        [1, 0, 1, 0],      # M4 -> C00, C10
        [-1, 1, 0, 0],     # M5 -> C00, C01
        [0, 0, 0, 1],      # M6 -> C11
        [1, 0, 0, 0]       # M7 -> C00
    ]
    
    for q in range(num_blocks):
        # Global indices for B in block q
        # B is 2x16. Row-major flatten: j*p + k.
        # Block q covers columns 2q and 2q+1.
        b_00 = 0 * p + 2 * q       # B[0, 2q]
        b_01 = 0 * p + (2 * q + 1) # B[0, 2q+1]
        b_10 = 1 * p + 2 * q       # B[1, 2q]
        b_11 = 1 * p + (2 * q + 1) # B[1, 2q+1]
        
        # Global indices for C in block q
        # C is 2x16. Convention: k*n + i.
        # Block q covers columns 2q and 2q+1.
        c_00 = (2 * q) * n + 0     # C[0, 2q]
        c_01 = (2 * q + 1) * n + 0 # C[0, 2q+1]
        c_10 = (2 * q) * n + 1     # C[1, 2q]
        c_11 = (2 * q + 1) * n + 1 # C[1, 2q+1]
        
        b_map = [b_00, b_01, b_10, b_11]
        c_map = [c_00, c_01, c_10, c_11]
        
        for s in range(7):
            r = q * 7 + s
            
            # Fill U
            for a_idx in range(4):
                coeff = U_coeffs[s][a_idx]
                if coeff != 0:
                    U[a_idx, r] = coeff
                    
            # Fill V
            for local_idx in range(4):
                coeff = V_coeffs[s][local_idx]
                if coeff != 0:
                    V[b_map[local_idx], r] = coeff
                    
            # Fill W
            for local_out_idx in range(4):
                coeff = W_contribs[s][local_out_idx]
                if coeff != 0:
                    W[c_map[local_out_idx], r] = coeff

    # Verification
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, size=(n, m))
        B_test = np.random.randint(-5, 5, size=(m, p))
        C_test = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        UA = U.T @ A_flat
        VB = V.T @ B_flat
        terms = UA * VB
        C_decomp = W @ terms
        
        # C_test is 2x16. W convention is k*n + i.
        # C_test.T is 16x2. Flatten row-major: index k*2 + i.
        C_test_flat = C_test.T.flatten()
        
        if not np.allclose(C_decomp, C_test_flat, rtol=1e-9, atol=1e-9):
            # Fallback to schoolbook if Strassen fails
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        U[u_idx, column] = 1
                        V[v_idx, column] = 1
                        W[w_idx, column] = 1
                        column += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
