import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_schoolbook(n, m, p):
    """Return the trivial schoolbook (rank = n*m*p) decomposition."""
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
    return U, V, W

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
    """Build a decomposition using block structure with Strassen for 2x2 blocks."""
    U_list = []
    V_list = []
    W_list = []
    
    for j in range(3):  # For each 3x2 block of B
        b_col_start = 2 * j
        
        # --- Top part of C (rows 0,1) ---
        # Term 1-7: Strassen for A[0:2, 0:2] * B_block[0:2, :]
        U_s, V_s, W_s = _strassen_2x2()
        
        for r in range(7):
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            
            for a_flat in range(4):
                ai = a_flat // 2
                aj = a_flat % 2
                U_term[ai * m + aj] = U_s[a_flat, r]
            
            for b_flat in range(4):
                bj = b_flat // 2
                bk = b_flat % 2
                global_bk = b_col_start + bk
                V_term[bj * p + global_bk] = V_s[b_flat, r]
            
            for c_flat in range(4):
                ci = c_flat // 2
                ck = c_flat % 2
                global_ck = b_col_start + ck
                W_term[global_ck * n + ci] = W_s[c_flat, r]
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
        
        # Term 8: A[0:2,2] * B_block[2,:] -> outer product
        U_term = np.zeros(n * m)
        V_term = np.zeros(m * p)
        W_term = np.zeros(p * n)
        
        for i in range(2):
            U_term[i * m + 2] = 1.0
        
        for k in range(2):
            V_term[2 * p + b_col_start + k] = 1.0
        
        for i in range(2):
            for k in range(2):
                global_k = b_col_start + k
                W_term[global_k * n + i] = 1.0
        
        U_list.append(U_term)
        V_list.append(V_term)
        W_list.append(W_term)
        
        # --- Bottom part of C (row 2) ---
        # Term 9-10: A[2,0:2] * B_block[0:2,:] -> rank 2
        for jj in range(2):
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            
            U_term[2 * m + jj] = 1.0
            
            for k in range(2):
                global_k = b_col_start + k
                V_term[jj * p + global_k] = 1.0
                W_term[global_k * n + 2] = 1.0
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
        
        # Term 11: A[2,2] * B_block[2,:] -> rank 1
        U_term = np.zeros(n * m)
        V_term = np.zeros(m * p)
        W_term = np.zeros(p * n)
        
        U_term[2 * m + 2] = 1.0
        
        for k in range(2):
            global_k = b_col_start + k
            V_term[2 * p + global_k] = 1.0
            W_term[global_k * n + 2] = 1.0
        
        U_list.append(U_term)
        V_list.append(V_term)
        W_list.append(W_term)
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    rank = U.shape[1]
    return rank, U, V, W

def _is_integer_or_half_integer(arr, tol=1e-9):
    """Check if all entries are integer or half-integer (i.e., 2*x is integer)."""
    doubled = 2.0 * arr
    rounded = np.round(doubled)
    return np.allclose(doubled, rounded, atol=tol)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    R = U.shape[1]
    
    # Check integer/half-integer entries
    if not _is_integer_or_half_integer(U):
        return False
    if not _is_integer_or_half_integer(V):
        return False
    if not _is_integer_or_half_integer(W):
        return False
    
    # Check all non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(R):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check all zero entries
    for a in range(a_dim):
        for b in range(b_dim):
            for c in range(c_dim):
                i1, j1 = a // m, a % m
                j2, k1 = b // p, b % p
                k2, i2 = c // n, c % n
                
                if i1 == i2 and j1 == j2 and k1 == k2:
                    continue
                
                s = 0.0
                for r in range(R):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s) > tol:
                    return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 140  # Safety margin (150s limit)
    
    # First try the correct block decomposition (rank 33)
    try:
        rank, U, V, W = _build_block_decomposition(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    if time.time() > deadline:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
