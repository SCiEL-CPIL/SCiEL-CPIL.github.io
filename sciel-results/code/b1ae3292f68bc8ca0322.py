import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # Output: C[i,k] = sum_r W[k*2+i, r] * M_r
    # C00 = M1 + M4 - M5 - M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
    # C01 = M3 + M5
    W[1, 2] = 1; W[1, 4] = 1
    # C10 = M2 + M4
    W[2, 1] = 1; W[2, 3] = 1
    # C11 = M1 - M2 + M3 + M7
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_block_decomposition(n, m, p):
    """Build a rank-51 decomposition for 3x3 * 3x6 using block Strassen."""
    U_list = []
    V_list = []
    W_list = []
    
    U_s, V_s, W_s = _strassen_2x2()
    
    for block in range(3):
        b_start = 2 * block
        
        # Strassen for A[0:2,0:2] * B[0:2, b_start:b_start+2]
        for r in range(7):
            U_term = np.zeros(9)
            V_term = np.zeros(18)
            W_term = np.zeros(18)
            
            for af in range(4):
                ai, aj = af // 2, af % 2
                U_term[ai * 3 + aj] = U_s[af, r]
            
            for bf in range(4):
                bj, bk = bf // 2, bf % 2
                V_term[bj * 6 + b_start + bk] = V_s[bf, r]
            
            for cf in range(4):
                ci, ck = cf // 2, cf % 2
                W_term[(b_start + ck) * 3 + ci] = W_s[cf, r]
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
        
        # A[0:2,2] * B[2, b_start:b_start+2]: 2x1 * 1x2 -> 4 terms
        for i in range(2):
            for k in range(2):
                U_term = np.zeros(9)
                V_term = np.zeros(18)
                W_term = np.zeros(18)
                gk = b_start + k
                U_term[i * 3 + 2] = 1.0
                V_term[2 * 6 + gk] = 1.0
                W_term[gk * 3 + i] = 1.0
                U_list.append(U_term)
                V_list.append(V_term)
                W_list.append(W_term)
        
        # A[2,0:2] * B[0:2, b_start:b_start+2]: 1x2 * 2x2 -> 4 terms
        for jj in range(2):
            for k in range(2):
                U_term = np.zeros(9)
                V_term = np.zeros(18)
                W_term = np.zeros(18)
                gk = b_start + k
                U_term[2 * 3 + jj] = 1.0
                V_term[jj * 6 + gk] = 1.0
                W_term[gk * 3 + 2] = 1.0
                U_list.append(U_term)
                V_list.append(V_term)
                W_list.append(W_term)
        
        # A[2,2] * B[2, b_start:b_start+2]: 1x1 * 1x2 -> 2 terms
        for k in range(2):
            U_term = np.zeros(9)
            V_term = np.zeros(18)
            W_term = np.zeros(18)
            gk = b_start + k
            U_term[2 * 3 + 2] = 1.0
            V_term[2 * 6 + gk] = 1.0
            W_term[gk * 3 + 2] = 1.0
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    R = U.shape[1]
    
    # Check all non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    
    # Check all zero entries (full check, not sampled)
    for a in range(a_dim):
        for b in range(b_dim):
            for c in range(c_dim):
                i1, j1 = a // m, a % m
                j2, k1 = b // p, b % p
                k2, i2 = c // n, c % n
                if i1 == i2 and j1 == j2 and k1 == k2:
                    continue
                s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
                if abs(s) > tol:
                    return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Primary: block Strassen decomposition (rank 51)
    try:
        U, V, W = _build_block_decomposition(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook (rank 54)
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
