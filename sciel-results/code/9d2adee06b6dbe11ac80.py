"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3 rows of A, m=4 cols of A/rows of B, p=4 cols of B
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (12, rank) with W[k*n+i]
    
    deadline = time.time() + 140  # safety margin
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try Strassen-based decomposition
    strassen_result = strassen_decomp(n, m, p)
    if strassen_result is not None:
        srank = len(strassen_result['U'][0])
        if srank < best_rank:
            best_result = strassen_result
            best_rank = srank
    
    # Try to find lower rank via ALS
    for rank in range(best_rank - 1, 0, -1):
        if time.time() > deadline:
            break
        result = try_rank(n, m, p, rank)
        if result is not None:
            best_result = result
            break
    
    return best_result

def strassen_decomp(n, m, p):
    """
    Decompose 3x4 * 4x4 using Strassen on 2x2 blocks.
    
    Split A (3x4) into top 2x4 and bottom 1x4.
    Split B (4x4) into top 2x4 and bottom 2x4.
    
    C_top (2x4) = A_top (2x4) * B (4x4)
    C_bot (1x4) = A_bot (1x4) * B (4x4)
    
    For C_top: split into left 2x2 and right 2x2.
    A_top = [A11 A12] where A11, A12 are 2x2
    B = [B11 B12; B21 B22] where each is 2x2
    
    C_top_left = A11*B11 + A12*B21 (2x2)
    C_top_right = A11*B12 + A12*B22 (2x2)
    
    Use Strassen for each 2x2 product (rank 7 each).
    C_bot: schoolbook (1*4*4 = 16 terms)
    
    Total: 7 + 7 + 16 = 30
    """
    U_list = []
    V_list = []
    W_list = []
    
    col = 0
    
    # Strassen for 2x2 * 2x2 -> 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def add_strassen_2x2(i_base, j_base, k_base, i_rows=2, j_cols=2, k_cols=2):
        """
        Add Strassen decomposition for A[i_base:i_base+i_rows, j_base:j_base+j_cols] * 
        B[j_base:j_base+j_cols, k_base:k_base+k_cols] -> C[i_base:i_base+i_rows, k_base:k_base+k_cols]
        Only works for 2x2 * 2x2 -> 2x2
        """
        nonlocal col
        if i_rows != 2 or j_cols != 2 or k_cols != 2:
            return
        
        # A is 2x2 at rows i_base, i_base+1; cols j_base, j_base+1
        # B is 2x2 at rows j_base, j_base+1; cols k_base, k_base+1
        # C is 2x2 at rows i_base, i_base+1; cols k_base, k_base+1
        
        # A11 = A[i_base, j_base], A12 = A[i_base, j_base+1]
        # A21 = A[i_base+1, j_base], A22 = A[i_base+1, j_base+1]
        # B11 = B[j_base, k_base], B12 = B[j_base, k_base+1]
        # B21 = B[j_base+1, k_base], B22 = B[j_base+1, k_base+1]
        # C11 = C[i_base, k_base], C12 = C[i_base, k_base+1]
        # C21 = C[i_base+1, k_base], C22 = C[i_base+1, k_base+1]
        
        a11 = (i_base, j_base)
        a12 = (i_base, j_base+1)
        a21 = (i_base+1, j_base)
        a22 = (i_base+1, j_base+1)
        
        b11 = (j_base, k_base)
        b12 = (j_base, k_base+1)
        b21 = (j_base+1, k_base)
        b22 = (j_base+1, k_base+1)
        
        c11 = (i_base, k_base)
        c12 = (i_base, k_base+1)
        c21 = (i_base+1, k_base)
        c22 = (i_base+1, k_base+1)
        
        # M1 = (A11+A22)(B11+B22)
        U_vals = [a11, a22]
        V_vals = [b11, b22]
        W_vals = [c11, c22]
        signs_U = [1, 1]
        signs_V = [1, 1]
        signs_W = [1, 1]
        U_list.append((U_vals, signs_U))
        V_list.append((V_vals, signs_V))
        W_list.append((W_vals, signs_W))
        col += 1
        
        # M2 = (A21+A22)B11
        U_vals = [a21, a22]
        V_vals = [b11]
        W_vals = [c21, c22]
        signs_U = [1, 1]
        signs_V = [1]
        signs_W = [1, -1]
        U_list.append((U_vals, signs_U))
        V_list.append((V_vals, signs_V))
        W_list.append((W_vals, signs_W))
        col += 1
        
        # M3 = A11(B21-B22)
        U_vals = [a11]
        V_vals = [b21, b22]
        W_vals = [c12, c22]
        signs_U = [1]
        signs_V = [1, -1]
        signs_W = [1, 1]
        U_list.append((U_vals, signs_U))
        V_list.append((V_vals, signs_V))
        W_list.append((W_vals, signs_W))
        col += 1
        
        # M4 = A22(B12-B11)
        U_vals = [a22]
        V_vals = [b12, b11]
        W_vals = [c11, c21]
        signs_U = [1]
        signs_V = [1, -1]
        signs_W = [1, 1]
        U_list.append((U_vals, signs_U))
        V_list.append((V_vals, signs_V))
        W_list.append((W_vals, signs_W))
        col += 1
        
        # M5 = (A11+A12)B22
        U_vals = [a11, a12]
        V_vals = [b22]
        W_vals = [c11, c12]
        signs_U = [1, 1]
        signs_V = [1]
        signs_W = [-1, 1]
        U_list.append((U_vals, signs_U))
        V_list.append((V_vals, signs_V))
        W_list.append((W_vals, signs_W))
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U_vals = [a21, a11]
        V_vals = [b11, b12]
        W_vals = [c22]
        signs_U = [1, -1]
        signs_V = [1, 1]
        signs_W = [1]
        U_list.append((U_vals, signs_U))
        V_list.append((V_vals, signs_V))
        W_list.append((W_vals, signs_W))
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U_vals = [a12, a22]
        V_vals = [b21, b22]
        W_vals = [c11]
        signs_U = [1, -1]
        signs_V = [1, 1]
        signs_W = [1]
        U_list.append((U_vals, signs_U))
        V_list.append((V_vals, signs_V))
        W_list.append((W_vals, signs_W))
        col += 1
    
    # For C_top (rows 0,1 of output): A_top (2x4) * B (4x4)
    # Split into left 2x2 and right 2x2
    # C_top_left = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    # C_top_right = A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2:4] * B[2:4, 2:4]
    
    # Each 2x2 * 2x2 product uses Strassen (7 terms)
    # Total for C_top: 4 Strassen decompositions = 28 terms
    
    # Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_2x2(0, 0, 0)  # 7 terms, col 0-6
    
    # Strassen for A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    add_strassen_2x2(0, 2, 0)  # 7 terms, col 7-13
    
    # Strassen for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen_2x2(0, 0, 2)  # 7 terms, col 14-20
    
    # Strassen for A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    add_strassen_2x2(0, 2, 2)  # 7 terms, col 21-27
    
    # For C_bot (row 2 of output): A[2:3, :] * B[:, :]
    # Schoolbook: 1 * 4 * 4 = 16 terms
    for j in range(4):
        for k in range(4):
            U_vals = [(2, j)]
            V_vals = [(j, k)]
            W_vals = [(k, 2)]
            signs_U = [1]
            signs_V = [1]
            signs_W = [1]
            U_list.append((U_vals, signs_U))
            V_list.append((V_vals, signs_V))
            W_list.append((W_vals, signs_W))
            col += 1
    
    total_rank = col
    
    # Build U, V, W matrices
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for c in range(total_rank):
        for (idx, sign) in zip(U_list[c][0], U_list[c][1]):
            i, j = idx
            U[i*m + j, c] = sign
        for (idx, sign) in zip(V_list[c][0], V_list[c][1]):
            j, k = idx
            V[j*p + k, c] = sign
        for (idx, sign) in zip(W_list[c][0], W_list[c][1]):
            k, i = idx
            W[k*n + i, c] = sign
    
    # Verify
    if not verify_full(n, m, p, U, V, W):
        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def verify_full(n, m, p, U, V, W):
    """Verify the full tensor decomposition."""
    tol = 1e-8
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            s = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :])
                            expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            if abs(s - expected) > tol:
                                return False
    return True

def try_rank(n, m, p, rank, max_iters=2000, tol=1e-10):
    """Try to find a rank decomposition for n x m times m x p matrix mult."""
    np.random.seed(42)
    
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    best_residual = np.inf
    
    for it in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        residual = compute_residual(n, m, p, U, V, W)
        if residual < best_residual:
            best_residual = residual
        if residual < tol:
            return verify_and_format(n, m, p, U, V, W)
    
    return None

def compute_residual(n, m, p, U, V, W):
    """Compute the tensor residual."""
    res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                res += (s - 1)**2
    return res

def verify_and_format(n, m, p, U, V, W):
    """Verify the decomposition and format output."""
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            s = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :])
                            expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            if abs(s - expected) > 1e-8:
                                return None
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
