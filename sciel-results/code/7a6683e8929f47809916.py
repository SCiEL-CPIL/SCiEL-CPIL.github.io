import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0
    
    # Rank 28 decomposition using Strassen's algorithm on 2x2 blocks.
    # A (2x4) is split into A_L (2x2, cols 0,1) and A_R (2x2, cols 2,3).
    # B (4x4) is split into B_TL (2x2, rows 0,1 cols 0,1), B_TR (2x2, rows 0,1 cols 2,3),
    #             B_BL (2x2, rows 2,3 cols 0,1), B_BR (2x2, rows 2,3 cols 2,3).
    # C (2x4) is split into C_L (2x2, cols 0,1) and C_R (2x2, cols 2,3).
    # C_L = A_L B_TL + A_R B_BL
    # C_R = A_L B_TR + A_R B_BR
    
    # Strassen terms for 2x2 matrix multiplication X*Y
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Each term is a tuple (u_coeffs, v_coeffs, w_coeffs)
    # Coeffs are lists of (local_index, value)
    
    strassen = [
        # M1 = (A00 + A11) * (B00 + B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10 + A11) * B00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A00 * (B10 - B11)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A11 * (B01 - B00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (A00 + A01) * B11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (A10 - A00) * (B00 + B01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A01 - A11) * (B10 + B11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Global index mappings
    # A (2x4):
    # A_L (cols 0,1): indices 0, 1, 4, 5 (row-major: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5)
    # A_R (cols 2,3): indices 2, 3, 6, 7
    u_map_L = [0, 1, 4, 5]
    u_map_R = [2, 3, 6, 7]
    
    # B (4x4):
    # B_TL (rows 0,1, cols 0,1): indices 0, 1, 4, 5
    # B_TR (rows 0,1, cols 2,3): indices 2, 3, 6, 7
    # B_BL (rows 2,3, cols 0,1): indices 8, 9, 12, 13
    # B_BR (rows 2,3, cols 2,3): indices 10, 11, 14, 15
    v_map_TL = [0, 1, 4, 5]
    v_map_TR = [2, 3, 6, 7]
    v_map_BL = [8, 9, 12, 13]
    v_map_BR = [10, 11, 14, 15]
    
    # C (2x4): W[k*n + i] = W[k*2 + i]
    # C_L (cols 0,1): 
    #   Local (r,c) where r in 0,1 and c in 0,1
    #   Global col = c. Global row = r.
    #   W index = global_col * n + global_row = c * 2 + r
    #   Loc 0 (0,0) -> 0*2+0 = 0
    #   Loc 1 (0,1) -> 1*2+0 = 2
    #   Loc 2 (1,0) -> 0*2+1 = 1
    #   Loc 3 (1,1) -> 1*2+1 = 3
    w_map_L = [0, 2, 1, 3]
    
    # C_R (cols 2,3):
    #   Global col = 2 + c. Global row = r.
    #   W index = (2+c) * 2 + r
    #   Loc 0 (0,0) -> 2*2+0 = 4
    #   Loc 1 (0,1) -> 3*2+0 = 6
    #   Loc 2 (1,0) -> 2*2+1 = 5
    #   Loc 3 (1,1) -> 3*2+1 = 7
    w_map_R = [4, 6, 5, 7]
    
    col = 0
    
    # 1. A_L * B_TL -> C_L
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U[u_map_L[idx], col] += val
        for idx, val in v_t:
            V[v_map_TL[idx], col] += val
        for idx, val in w_t:
            W[w_map_L[idx], col] += val
        col += 1
        
    # 2. A_R * B_BL -> C_L
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U[u_map_R[idx], col] += val
        for idx, val in v_t:
            V[v_map_BL[idx], col] += val
        for idx, val in w_t:
            W[w_map_L[idx], col] += val
        col += 1
            
    # 3. A_L * B_TR -> C_R
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U[u_map_L[idx], col] += val
        for idx, val in v_t:
            V[v_map_TR[idx], col] += val
        for idx, val in w_t:
            W[w_map_R[idx], col] += val
        col += 1
            
    # 4. A_R * B_BR -> C_R
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U[u_map_R[idx], col] += val
        for idx, val in v_t:
            V[v_map_BR[idx], col] += val
        for idx, val in w_t:
            W[w_map_R[idx], col] += val
        col += 1
        
    if time.time() - start_time > time_limit:
        raise TimeoutError("Construction took too long")
        
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Test with a few random matrices
    for seed in [42, 123, 999]:
        np.random.seed(seed)
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U_arr[ii * m + jj, cc] * V_arr[jj * p + kk, cc] * W_arr[kk * n + ii, cc]
                C_computed[ii, kk] = acc
        
        if not np.allclose(C_expected, C_computed, atol=1e-10):
            # Fallback: return naive rank-32 decomposition
            U_naive = np.zeros((n * m, n * m * p), dtype=int)
            V_naive = np.zeros((m * p, n * m * p), dtype=int)
            W_naive = np.zeros((p * n, n * m * p), dtype=int)
            col = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_naive[ii * m + jj, col] = 1
                        V_naive[jj * p + kk, col] = 1
                        W_naive[kk * n + ii, col] = 1
                        col += 1
            return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices to ensure correctness
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    if not np.allclose(C, Cf):
        print("Verification Failed")
        print(C)
        print(Cf)
    else:
        print("Verification Passed")
    print("Rank:", U.shape[1])
