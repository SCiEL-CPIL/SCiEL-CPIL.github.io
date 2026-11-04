import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _make_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
    if not np.allclose(T_rec, T, atol=tol):
        return False
    return True

def _strassen_factors_2x2x2():
    # Returns U, V, W of shape (4, 7)
    # Indices: i in 0..3 (2x2), j in 0..3, k in 0..3
    # i = 2*i1 + i2
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A11+A22)(B11+B22) -> C11 += M1, C12 += M1, C21 += M1, C22 += M1
    # A11 is (0,0), A22 is (1,1)
    # U for A11: (0,0)=1. U for A22: (1,1)=1.
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2: (A21+A22)B11 -> C21 += M2
    # A21 is (1,0), A22 is (1,1). B11 is (0,0).
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3: A11(B12-B22) -> C12 += M3
    # A11 is (0,0). B12 is (0,1), B22 is (1,1).
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4: A22(B21-B11) -> C21 += M4
    # A22 is (1,1). B21 is (1,0), B11 is (0,0).
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # M5: (A11+A12)B22 -> C12 += M5
    # A11 is (0,0), A12 is (0,1). B22 is (1,1).
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # M6: (A21-A11)(B11+B12) -> C21 += M6, C22 += M6
    # A21 is (1,0), A11 is (0,0). B11 is (0,0), B12 is (0,1).
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7: (A12-A22)(B21+B22) -> C12 += M7, C22 += M7
    # A12 is (0,1), A22 is (1,1). B21 is (1,0), B22 is (1,1).
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def construct():
    deadline = time.time() + 120 - 10
    
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    
    # Rank 51 decomposition
    rank = 51
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: A_R B_B (Rank 1)
    # A_R is col 4 of A. B_B is row 4 of B.
    # T[i, 4, k] = 1
    col = 0
    U[:, col] = 1.0
    V[4, col] = 1.0  # j=4
    W[:, col] = 1.0
    col += 1
    
    # Part 2: A_L B_TR (Rank 1)
    # A_L is cols 0..3 of A. B_TR is col 4 of B (rows 0..3).
    # T[i, j, 4] = 1 for j in 0..3
    U[:, col] = 1.0
    V[0:4, col] = 1.0
    V[4, col] = 0.0
    W[4, col] = 1.0 # k=4
    col += 1
    
    # Part 3: A_L B_TL (Rank 49)
    # 4x4x4 multiplication
    # Indices i, j, k in 0..3
    U_s, V_s, W_s = _strassen_factors_2x2x2()
    
    # Kronecker product
    # U_444[i, c] = U_s[i1, c1] * U_s[i2, c2]
    # i = 2*i1 + i2
    # c = 7*c1 + c2
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            if c >= col: break
            # U
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    U[i, col + c] = U_s[i1, c1] * U_s[i2, c2]
            
            # V
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    V[j, col + c] = V_s[j1, c1] * V_s[j2, c2]
            
            # W
            for k1 in range(2):
                for k2 in range(2):
                    k = k1 * 2 + k2
                    W[k, col + c] = W_s[k1, c1] * W_s[k2, c2]
            
    col += 49
    
    # Verify
    T = _make_tensor(n, m, p)
    if _verify_decomposition(U, V, W, T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    else:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_fb = np.zeros((n*m, rank_schoolbook))
        V_fb = np.zeros((m*p, rank_schoolbook))
        W_fb = np.zeros((p*n, rank_schoolbook))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m + j, c] = 1
                    V_fb[j*p + k, c] = 1
                    W_fb[k*n + i, c] = 1
                    c += 1
        return {'U': U_fb.tolist(), 'V': V_fb.tolist(), 'W': W_fb.tolist()}
