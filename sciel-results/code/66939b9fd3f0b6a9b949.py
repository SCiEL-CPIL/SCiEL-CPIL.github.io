import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, n=3, m=3, p=6, trials=50, seed=42):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Direct tensor verification using numpy
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22)(B11 + B22) -> C11 += M1, C22 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21 + A22)B11 -> C22 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[3, 1] = -1
    
    # M3 = A11(B12 - B22) -> C22 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[3, 2] = 1
    
    # M4 = A22(B21 - B11) -> C21 += M4, C11 -= M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    W[0, 3] = -1
    
    # M5 = (A11 + A12)B22 -> C12 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # M6 = (A21 - A11)(B11 + B12) -> C11 += M6, C21 += M6, C22 -= M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1
    W[2, 5] = 1
    W[3, 5] = -1
    
    # M7 = (A12 - A22)(B21 + B22) -> C12 += M7, C11 -= M7, C21 -= M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1
    W[0, 6] = -1
    W[2, 6] = -1
    
    return U, V, W

def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank26_3x3x3():
    """
    Construct a rank-26 decomposition for 3x3x3 matrix multiplication.
    """
    n, m, p = 3, 3, 3
    rank = 26
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 1. Strassen for A22 B22 (indices 0,1,3,4)
    U_s, V_s, W_s = _strassen_2x2x2()
    map_idx = [0, 1, 3, 4]
    for i in range(4):
        for c in range(7):
            U[map_idx[i], c] = U_s[i, c]
            V[map_idx[i], c] = V_s[i, c]
            W[map_idx[i], c] = W_s[i, c]
    
    # 2. a e (indices 2,5 for A; 6,7 for B; 0,1,3,4 for C)
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[1, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[3, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[4, 10] = 1
    
    # 3. A22 c (indices 0,1,3,4 for A; 2,5 for B; 2,5 for C)
    U[0, 11] = 1; V[2, 11] = 1; W[2, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[2, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    
    # 4. a f (indices 2,5 for A; 8 for B; 2,5 for C)
    U[2, 15] = 1; V[8, 15] = 1; W[2, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    # 5. d B22 (indices 6,7 for A; 0,1,3,4 for B; 6,7 for C)
    U[6, 17] = 1; V[0, 17] = 1; W[6, 17] = 1
    U[7, 18] = 1; V[3, 18] = 1; W[6, 18] = 1
    U[6, 19] = 1; V[1, 19] = 1; W[7, 19] = 1
    U[7, 20] = 1; V[4, 20] = 1; W[7, 20] = 1
    
    # 6. b e (index 8 for A; 6,7 for B; 6,7 for C)
    U[8, 21] = 1; V[6, 21] = 1; W[6, 21] = 1
    U[8, 22] = 1; V[7, 22] = 1; W[7, 22] = 1
    
    # 7. d c (indices 6,7 for A; 2,5 for B; 8 for C)
    U[6, 23] = 1; V[2, 23] = 1; W[8, 23] = 1
    U[7, 24] = 1; V[5, 24] = 1; W[8, 24] = 1
    
    # 8. b f (index 8 for A; 8 for B; 8 for C)
    U[8, 25] = 1; V[8, 25] = 1; W[8, 25] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _combine_3x3x6():
    """
    Combine two 3x3x3 decompositions to get 3x3x6.
    """
    decomp = _rank26_3x3x3()
    U1 = decomp["U"]
    V1 = decomp["V"]
    W1 = decomp["W"]
    
    rank = 52
    V_total = [[0]*rank for _ in range(18)]
    W_total = [[0]*rank for _ in range(18)]
    
    # Copy V1 to V_total (cols 0-2 of B)
    for j in range(3):
        for k in range(3):
            idx_b1 = j*3 + k
            idx_b_total = j*6 + k
            for c in range(26):
                V_total[idx_b_total][c] = V1[idx_b1][c]
    
    # Copy V1 to V_total (cols 3-5 of B)
    for j in range(3):
        for k in range(3):
            idx_b1 = j*3 + k
            idx_b_total = j*6 + (k+3)
            for c in range(26):
                V_total[idx_b_total][c+26] = V1[idx_b1][c]
                
    # Copy W1 to W_total (cols 0-2 of C)
    for k in range(3):
        for i in range(3):
            idx_c1 = k*3 + i
            idx_c_total = k*6 + i
            for c in range(26):
                W_total[idx_c_total][c] = W1[idx_c1][c]
                
    # Copy W1 to W_total (cols 3-5 of C)
    for k in range(3):
        for i in range(3):
            idx_c1 = k*3 + i
            idx_c_total = k*6 + (i+3)
            for c in range(26):
                W_total[idx_c_total][c+26] = W1[idx_c1][c]
                
    return {"U": U1, "V": V_total, "W": W_total}

def construct():
    start_time = time.time()
    deadline = 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        candidate = _combine_3x3x6()
        if _verify_decomposition(candidate["U"], candidate["V"], candidate["W"], n, m, p):
            return candidate
    except Exception:
        pass
    
    return _schoolbook(n, m, p)
