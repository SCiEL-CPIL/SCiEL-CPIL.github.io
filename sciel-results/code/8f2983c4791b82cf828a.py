import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _get_strassen_2x2_local():
    """
    Returns U, V, W for 2x2 Strassen decomposition.
    U: (4, 7) for A[0,0], A[0,1], A[1,0], A[1,1]
    V: (4, 7) for B[0,0], B[0,1], B[1,0], B[1,1]
    W: (4, 7) for C[0,0], C[0,1], C[1,0], C[1,1] (Note: W index is k*n+i, so for 2x2, k*2+i)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00 - A11)(B01 + B11)
    U[0, 0] = 1.0; U[3, 0] = -1.0
    V[1, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0 # C00
    W[3, 0] = 1.0 # C11 (index 1*2+1=3)
    
    # M2 = (A00 + A11)B11
    U[0, 1] = 1.0; U[3, 1] = 1.0
    V[3, 1] = 1.0
    W[2, 1] = 1.0 # C10 (index 0*2+1=2)
    W[3, 1] = -1.0 # C11
    
    # M3 = A00(B00 - B11)
    U[0, 2] = 1.0
    V[0, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0 # C01 (index 1*2+0=1)
    W[3, 2] = 1.0 # C11
    
    # M4 = A11(B00 + B11)
    U[3, 3] = 1.0
    V[0, 3] = 1.0; V[3, 3] = 1.0
    W[0, 3] = 1.0 # C00
    W[2, 3] = 1.0 # C10
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0 # C00
    W[1, 4] = 1.0 # C01
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0 # C11
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0 # C00
    W[3, 6] = 1.0 # C11
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    """
    r = U.shape[1]
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    L = A_flat @ U
    M = B_flat @ V
    
    C_flat = W @ (L * M)
    C_computed = C_flat.reshape(p, n).T
    
    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 110
    
    # Define the 4 Strassen blocks
    # Each block is (i_start, j_start, k_start)
    blocks = [
        (0, 0, 0),
        (0, 2, 2),
        (2, 0, 0),
        (2, 2, 2)
    ]
    
    # Create a set of (i, j, k) tuples covered by Strassen blocks
    strassen_terms = set()
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk
                    if i < n and j < m and k < p:
                        strassen_terms.add((i, j, k))
    
    # Initialize U, V, W
    # We don't know the final rank yet, so let's use a list of columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    # 1. Add Schoolbook terms for all (i,j,k) NOT in strassen_terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_terms:
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)
                    
                    U_col[i * m + j] = 1.0
                    V_col[j * p + k] = 1.0
                    W_col[k * n + i] = 1.0
                    
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
    
    # 2. Add Strassen terms for each block
    U_local, V_local, W_local = _get_strassen_2x2_local()
    
    for i0, j0, k0 in blocks:
        for c in range(7):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            
            # Map local indices to global
            for li in range(2):
                for lj in range(2):
                    val = U_local[li * 2 + lj, c]
                    if val != 0:
                        gi = i0 + li
                        gj = j0 + lj
                        U_col[gi * m + gj] = val
            
            for lj in range(2):
                for lk in range(2):
                    val = V_local[lj * 2 + lk, c]
                    if val != 0:
                        gj = j0 + lj
                        gk = k0 + lk
                        V_col[gj * p + gk] = val
            
            for lk in range(2):
                for li in range(2):
                    val = W_local[lk * 2 + li, c]
                    if val != 0:
                        gk = k0 + lk
                        gi = i0 + li
                        W_col[gk * n + gi] = val
            
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
    
    # Convert to numpy arrays
    U = np.array(U_cols).T # (n*m, rank)
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    rank = U.shape[1]
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        U_s = np.zeros((n * m, n * m * p), dtype=float)
        V_s = np.zeros((m * p, n * m * p), dtype=float)
        W_s = np.zeros((p * n, n * m * p), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1.0
                    V_s[j * p + k, col] = 1.0
                    W_s[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

# EVOLVE-BLOCK-END
