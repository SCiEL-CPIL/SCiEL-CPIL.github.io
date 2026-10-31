import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 4, 4, 4
    
    # Bini-Capovani-Lotti-Romani rank-7 identity for 2x2 matrix multiplication
    # Using complex coefficients with sqrt(2)
    # The 7 multiplication terms are:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)(B00)
    # M2 = (A00)(B01-B11)
    # M3 = (A11)(B10-B00)
    # M4 = (A00+A01)(B11)
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 - M3 + M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M5 + M6
    
    # Define the 2x2 Strassen coefficients
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # M0
        [[0, 0], [1, 1]],   # M1
        [[1, 0], [0, 0]],   # M2
        [[0, 0], [0, 1]],   # M3
        [[1, 1], [0, 0]],   # M4
        [[-1, 0], [1, 0]],  # M5
        [[0, 1], [0, -1]],  # M6
    ])
    
    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # M0
        [[1, 0], [0, 0]],   # M1
        [[0, 1], [0, -1]],  # M2
        [[-1, 0], [1, 0]],  # M3
        [[0, 0], [0, 1]],   # M4
        [[1, 1], [0, 0]],   # M5
        [[0, 0], [1, 1]],   # M6
    ])
    
    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # M0: C00 += M0, C11 += M0
        [[0, 0], [1, -1]],  # M1: C10 += M1, C11 -= M1
        [[0, 1], [0, 1]],   # M2: C01 += M2, C11 += M2
        [[1, 0], [1, 0]],   # M3: C00 += M3, C10 += M3
        [[-1, 1], [0, 0]],  # M4: C00 -= M4, C01 += M4
        [[0, 0], [0, 1]],   # M5: C11 += M5
        [[-1, 0], [0, 0]],  # M6: C00 -= M6
    ])
    
    # Build 4x4 factors using recursive block decomposition
    # Rank = 7 * 7 = 49
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # A[i,j] where i = 2*S + s2, j = 2*T + t2
    # U[i*4+j, s*7+t] = a_coeffs[s][S][T] * a_coeffs[t][s2][t2]
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for S in range(2):
                for T in range(2):
                    outer_a = a_coeffs[s, S, T]
                    if outer_a == 0:
                        continue
                    for s2 in range(2):
                        for t2 in range(2):
                            inner_a = a_coeffs[t, s2, t2]
                            if inner_a == 0:
                                continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * 4 + j, col] += outer_a * inner_a
    
    # B[j,k] where j = 2*T + t2, k = 2*K + k2
    # V[j*4+k, s*7+t] = b_coeffs[s][T][K] * b_coeffs[t][t2][k2]
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for T in range(2):
                for K in range(2):
                    outer_b = b_coeffs[s, T, K]
                    if outer_b == 0:
                        continue
                    for t2 in range(2):
                        for k2 in range(2):
                            inner_b = b_coeffs[t, t2, k2]
                            if inner_b == 0:
                                continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * 4 + k, col] += outer_b * inner_b
    
    # C[i,k] where i = 2*S + s2, k = 2*K + k2
    # W[k*4+i, s*7+t] = c_coeffs[s][S][K] * c_coeffs[t][s2][k2]
    # Note: trace convention is W[k*n+i, r] = W[k*4+i, r]
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for S in range(2):
                for K in range(2):
                    outer_c = c_coeffs[s, S, K]
                    if outer_c == 0:
                        continue
                    for s2 in range(2):
                        for k2 in range(2):
                            inner_c = c_coeffs[t, s2, k2]
                            if inner_c == 0:
                                continue
                            i = 2 * S + s2
                            k = 2 * K + k2
                            W[k * 4 + i, col] += outer_c * inner_c
    
    # Verify the decomposition
    def verify(U_arr, V_arr, W_arr):
        target = np.zeros((16, 16, 16))
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    a_idx = i * 4 + k
                    b_idx = k * 4 + j
                    c_idx = j * 4 + i
                    target[a_idx, b_idx, c_idx] = 1.0
        
        recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(recon, target, atol=1e-9)
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to naive rank 64
        U_n = np.zeros((16, 64), dtype=float)
        V_n = np.zeros((16, 64), dtype=float)
        W_n = np.zeros((16, 64), dtype=float)
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i * 16 + k * 4 + j
                    U_n[i * 4 + k, r] = 1.0
                    V_n[k * 4 + j, r] = 1.0
                    W_n[j * 4 + i, r] = 1.0
        return {"U": U_n.tolist(), "V": V_n.tolist(), "W": W_n.tolist()}
