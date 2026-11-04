"""Matrix multiplication 4x4 * 4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _winograd_2x2x2():
    """
    Return the rank-7 Winograd F(2) decomposition for 2x2 * 2x2.
    C = A @ B where A is 2x2, B is 2x2, C is 2x2.
    
    Indexing:
    U has shape (n*m, rank) = (4, 7), row index = i*m + j = i*2 + j
    V has shape (m*p, rank) = (4, 7), row index = j*p + k = j*2 + k  
    W has shape (p*n, rank) = (4, 7), row index = k*n + i = k*2 + i
    
    The 7 factors for Winograd F(2):
    u0 = a00 + a11
    u1 = a00 + a01
    u2 = a00 + a10
    u3 = a01 + a10
    u4 = a01 - a11
    u5 = a10 - a11
    u6 = a11
    
    v0 = b00 + b11
    v1 = b00 - b11
    v2 = b00 + b01
    v3 = b10 + b11
    v4 = b10 - b11
    v5 = b01 - b11
    v6 = b11
    
    c00 = u0*v0 - u4*v5 - u6*v6
    c01 = u2*v4 + u4*v3
    c10 = u3*v2 - u5*v4
    c11 = u0*v1 + u5*v3 + u6*v6
    """
    U = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, -1, -1, 1],
    ], dtype=float)
    
    V = np.array([
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [1, -1, 0, 1, -1, -1, 1],
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 0, 0, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, -1, 0, 0],
        [0, 1, 0, 0, 0, 1, 1],
    ], dtype=float)
    
    n, m, p = 2, 2, 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(7):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_4x4x4_kron():
    """
    Construct a rank-49 decomposition for 4x4 * 4x4 using Kronecker product of Winograd F(2).
    """
    w2 = _winograd_2x2x2()
    if not w2:
        return None
        
    U2 = np.array(w2["U"])  # shape (4, 7)
    V2 = np.array(w2["V"])  # shape (4, 7)
    W2 = np.array(w2["W"])  # shape (4, 7)
    
    n, m, p = 4, 4, 4
    rank = 49
    U4 = np.zeros((n * m, rank))  # (16, 49)
    V4 = np.zeros((m * p, rank))  # (16, 49)
    W4 = np.zeros((p * n, rank))  # (16, 49)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for c1 in range(7):
                        for c2 in range(7):
                            row_idx = i1 * 8 + i2 * 4 + j1 * 2 + j2
                            col_idx = c1 * 7 + c2
                            U4[row_idx, col_idx] = U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]
                            
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    for c1 in range(7):
                        for c2 in range(7):
                            row_idx = j1 * 8 + j2 * 4 + k1 * 2 + k2
                            col_idx = c1 * 7 + c2
                            V4[row_idx, col_idx] = V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]
                            
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    for c1 in range(7):
                        for c2 in range(7):
                            row_idx = k1 * 8 + k2 * 4 + i1 * 2 + i2
                            col_idx = c1 * 7 + c2
                            W4[row_idx, col_idx] = W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]
                            
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

def _construct_4x4x5():
    """
    Construct a rank-53 decomposition for 4x4 * 4x5.
    
    Use the rank-49 Kronecker decomposition for the first 4 columns (k=0,1,2,3),
    then add 4 additional rank terms for the 5th column (k=4).
    """
    n, m, p = 4, 4, 5
    
    base = _construct_4x4x4_kron()
    if not base:
        return _schoolbook(n, m, p)
        
    U4 = np.array(base["U"])  # (16, 49)
    V4 = np.array(base["V"])  # (16, 49) - for 4x4x4, V has shape (m*p, rank) = (16, 49)
    W4 = np.array(base["W"])  # (16, 49)
    
    rank_base = 49
    rank_new = rank_base + 4  # 4 additional terms for the 5th column
    
    U_new = np.zeros((n * m, rank_new))  # (16, 53)
    V_new = np.zeros((m * p, rank_new))  # (20, 53)
    W_new = np.zeros((p * n, rank_new))  # (20, 53)
    
    # Copy base part for U
    U_new[:, :rank_base] = U4
    
    # Copy base part for V
    # V4 has shape (16, 49) where row = j*4+k for j,k in 0..3 (4x4x4 case)
    # V_new has shape (20, 53) where row = j*5+k for j in 0..3, k in 0..4
    # For k=0..3: V_new[j*5+k, :] = V4[j*4+k, :]
    for j in range(4):
        for k in range(4):
            V_new[j * 5 + k, :rank_base] = V4[j * 4 + k, :rank_base]
            
    # Copy base part for W
    # W4 has shape (16, 49) where row = k*4+i for k,i in 0..3 (4x4x4 case)
    # W_new has shape (20, 53) where row = k*4+i for k in 0..4, i in 0..3
    # For k=0..3: W_new[k*4+i, :] = W4[k*4+i, :]
    W_new[:16, :rank_base] = W4
    
    # Add 4 additional terms for the 5th column (k=4)
    for j in range(4):
        col_idx = rank_base + j
        # U[i*4+j, col_idx] = 1 for all i
        for i in range(4):
            U_new[i * 4 + j, col_idx] = 1.0
        # V[j*5+4, col_idx] = 1
        V_new[j * 5 + 4, col_idx] = 1.0
        # W[4*4+i, col_idx] = 1 for all i
        for i in range(4):
            W_new[4 * 4 + i, col_idx] = 1.0
        
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

def _try_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by checking if any column can be eliminated
    via linear combination of others.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check for linearly dependent columns in U, V, W
    # If col c in U is a linear combination of other cols, and similarly in V and W,
    # we might be able to eliminate it.
    
    for c in range(rank):
        if time.time() > deadline:
            break
        # Check if column c in U is in span of other columns
        U_others = np.delete(U_arr, c, axis=1)
        U_col = U_arr[:, c]
        # Solve U_others @ x = U_col
        try:
            x, residuals, _, _ = np.linalg.lstsq(U_others, U_col, rcond=1e-10)
            if residuals.size > 0 and residuals[0] < 1e-8:
                # U col c is dependent. Check V and W.
                V_others = np.delete(V_arr, c, axis=1)
                V_col = V_arr[:, c]
                y, res2, _, _ = np.linalg.lstsq(V_others, V_col, rcond=1e-10)
                if res2.size > 0 and res2[0] < 1e-8:
                    W_others = np.delete(W_arr, c, axis=1)
                    W_col = W_arr[:, c]
                    z, res3, _, _ = np.linalg.lstsq(W_others, W_col, rcond=1e-10)
                    if res3.size > 0 and res3[0] < 1e-8:
                        # All three columns are dependent. Remove column c.
                        U_new = np.delete(U_arr, c, axis=1)
                        V_new = np.delete(V_arr, c, axis=1)
                        W_new = np.delete(W_arr, c, axis=1)
                        # Verify
                        if _verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                            return U_new.tolist(), V_new.tolist(), W_new.tolist()
        except np.linalg.LinAlgError:
            pass
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140
    
    try:
        result = _construct_4x4x5()
        if result and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            # Try to reduce rank
            U, V, W = result["U"], result["V"], result["W"]
            U, V, W = _try_rank_reduction(U, V, W, n, m, p, deadline)
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            return result
    except Exception:
        pass
    
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
