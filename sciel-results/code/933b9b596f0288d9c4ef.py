import numpy as np
import time
import itertools

DEADLINE = 115.0
START_TIME = time.time()

def _check_deadline():
    return time.time() - START_TIME > DEADLINE

def _strassen_2x2():
    """Returns U, V, W as 2D arrays of shape (4, 7) for 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Flatten indices: row*2 + col
    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_4x4_factors():
    """
    Builds 4x4x4 factors using recursive Strassen.
    Returns U, V, W of shapes (16, 49).
    """
    if _check_deadline():
        raise TimeoutError()

    U2, V2, W2 = _strassen_2x2()
    
    rank_4 = 49
    U4 = np.zeros((16, rank_4))
    V4 = np.zeros((16, rank_4))
    W4 = np.zeros((16, rank_4))
    
    # U4[i*4+j, R]
    # i, j in 0..3. i = 2*i1 + i0, j = 2*j1 + j0.
    # R = 7*s + r.
    # U4[i*4+j, R] = U2[i1*2+j1, s] * U2[i0*2+j0, r]
    
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2 * i1 + i0
                    j = 2 * j1 + j0
                    idx_u = i * 4 + j
                    idx_u2_block = i1 * 2 + j1
                    idx_u2_inner = i0 * 2 + j0
                    for s in range(7):
                        u_block = U2[idx_u2_block, s]
                        if u_block == 0: continue
                        for r in range(7):
                            u_inner = U2[idx_u2_inner, r]
                            if u_inner == 0: continue
                            R = 7 * s + r
                            U4[idx_u, R] = u_block * u_inner
    
    # V4[j*4+k, R]
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2 * j1 + j0
                    k = 2 * k1 + k0
                    idx_v = j * 4 + k
                    idx_v2_block = j1 * 2 + k1
                    idx_v2_inner = j0 * 2 + k0
                    for s in range(7):
                        v_block = V2[idx_v2_block, s]
                        if v_block == 0: continue
                        for r in range(7):
                            v_inner = V2[idx_v2_inner, r]
                            if v_inner == 0: continue
                            R = 7 * s + r
                            V4[idx_v, R] = v_block * v_inner
    
    # W4[k*4+i, R]
    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2 * k1 + k0
                    i = 2 * i1 + i0
                    idx_w = k * 4 + i
                    idx_w2_block = k1 * 2 + i1
                    idx_w2_inner = k0 * 2 + i0
                    for s in range(7):
                        w_block = W2[idx_w2_block, s]
                        if w_block == 0: continue
                        for r in range(7):
                            w_inner = W2[idx_w2_inner, r]
                            if w_inner == 0: continue
                            R = 7 * s + r
                            W4[idx_w, R] = w_block * w_inner

    return U4, V4, W4

def _rank_of_matrix(M):
    return np.linalg.matrix_rank(M, tol=1e-10)

def _build_4x4x8_factors():
    """
    Builds 4x4x8 factors.
    Returns U, V, W of shapes (16, 98), (32, 98), (32, 98).
    """
    if _check_deadline():
        raise TimeoutError()

    U4, V4, W4 = _build_4x4_factors()
    
    n, m, p = 4, 4, 8
    
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First half (k=0..3)
    # U[i*m+j, R] = U4[i*4+j, R]
    for i in range(n):
        for j in range(m):
            for R in range(49):
                U[i * m + j, R] = U4[i * 4 + j, R]
                
    # V[j*p+k, R] = V4[j*4+k, R] for k in 0..3
    for j in range(m):
        for k in range(4):
            for R in range(49):
                V[j * p + k, R] = V4[j * 4 + k, R]
                
    # W[k*n+i, R] = W4[k*4+i, R] for k in 0..3
    for k in range(4):
        for i in range(n):
            for R in range(49):
                W[k * n + i, R] = W4[k * 4 + i, R]
                
    # Second half (k=4..7)
    for i in range(n):
        for j in range(m):
            for R in range(49):
                U[i * m + j, 49 + R] = U4[i * 4 + j, R]
                
    for j in range(m):
        for kp in range(4):
            k = 4 + kp
            for R in range(49):
                V[j * p + k, 49 + R] = V4[j * 4 + kp, R]
                
    for kp in range(4):
        k = 4 + kp
        for i in range(n):
            for R in range(49):
                W[k * n + i, 49 + R] = W4[kp * 4 + i, R]

    return U, V, W

def _try_reduce_rank(U, V, W):
    """
    Attempts to reduce the rank by checking for linear dependencies.
    Returns reduced U, V, W if successful, else original.
    """
    if _check_deadline():
        return U, V, W

    n, m, p = 4, 4, 8
    rank = U.shape[1]
    
    # Check rank of U
    rU = _rank_of_matrix(U)
    # Check rank of V
    rV = _rank_of_matrix(V)
    # Check rank of W
    rW = _rank_of_matrix(W)
    
    # If any matrix has rank less than the current rank, we might be able to reduce.
    # However, the rank of the tensor is bounded by the rank of any of the factor matrices.
    # But we need to ensure the decomposition still holds.
    # If rank(U) < rank, it means some columns are linear combinations of others.
    # This implies that the corresponding columns in V and W are also linearly dependent in a way that preserves the tensor.
    
    # Let's try to find a basis for the columns of U.
    if rU < rank:
        # Find a pivot set
        U_pivots = np.linalg.matrix_rank(U[:, :rank], tol=1e-10)
        # This is a bit complex to implement robustly without a full QR with column pivoting.
        # For now, let's just return the original if we detect rank deficiency, 
        # but we can't easily map it back to V and W without more work.
        pass
        
    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    
    try:
        U, V, W = _build_4x4x8_factors()
        
        # Attempt rank reduction (currently a no-op placeholder for future work)
        U, V, W = _try_reduce_rank(U, V, W)
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
        
    except Exception as e:
        # Fallback to schoolbook if Strassen fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
