import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Try to find a rank-83 decomposition
    result_83 = _try_find_rank_83(n, m, p, deadline)
    if result_83 is not None:
        return result_83

    # 2. Fallback to known rank-84
    U, V, W, rank = _build_strassen_3x5x6(n, m, p)
    
    # Verify
    if _verify_fast(U, V, W, n, m, p):
        # Convert to lists with integer/half-integer rounding
        U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U.tolist()]
        V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V.tolist()]
        W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W.tolist()]
        return {"U": U_list, "V": V_list, "W": W_list}

    # 3. Fallback to schoolbook (rank 90)
    return _build_schoolbook(n, m, p)

def _try_find_rank_83(n, m, p, deadline):
    """
    Attempt to find a rank-83 decomposition by checking for linear dependencies
    in the rank-84 solution terms.
    """
    U84, V84, W84, rank84 = _build_strassen_3x5x6(n, m, p)
    
    if time.time() > deadline - 10:
        return None

    u_flat = U84.T  # (84, 15)
    v_flat = V84.T  # (84, 30)
    w_flat = W84.T  # (84, 18)
    
    try:
        # Construct the matrix M where columns are the vectorized rank-1 terms
        # M[:, c] = vec(u_c v_c w_c)
        # Shape: (15*30*18, 84) = (8100, 84)
        
        # Efficiently create the 4D tensor and reshape
        # T_terms[i, j, k, c] = u_flat[c, i] * v_flat[c, j] * w_flat[c, k]
        # Using einsum: 'ic,jc,kc->ijkc'
        T_terms = np.einsum('ic,jc,kc->ijkc', u_flat, v_flat, w_flat)
        M = T_terms.reshape(-1, 84)
        
        if time.time() > deadline - 5:
            return None
            
        # Compute SVD to find null space
        # full_matrices=False for efficiency
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        
        # Check if the smallest singular value is close to 0
        # If so, the terms are linearly dependent.
        if s[-1] < 1e-9:
            x = Vt[-1, :]
            
            # Find index with largest |x_i|
            idx = np.argmax(np.abs(x))
            if abs(x[idx]) < 1e-9:
                return None
                
            # Coefficients for the new 83 terms
            # We eliminate term 'idx'.
            # New coeffs for c != idx: 1 - x_c / x_idx
            coeffs = np.ones(84)
            coeffs[idx] = 0
            for c in range(84):
                if c != idx:
                    coeffs[c] = 1 - x[c] / x[idx]
            
            # Apply coefficients to U
            U_new = U84 * coeffs
            V_new = V84
            W_new = W84
            
            # Remove the 'idx' column
            mask = np.ones(84, dtype=bool)
            mask[idx] = False
            U_new = U_new[:, mask]
            V_new = V_new[:, mask]
            W_new = W_new[:, mask]
            
            # Verify
            if _verify_fast(U_new, V_new, W_new, n, m, p):
                U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_new.tolist()]
                V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_new.tolist()]
                W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_new.tolist()]
                return {"U": U_list, "V": V_list, "W": W_list}
                
    except Exception:
        pass
        
    return None

def _get_strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U (A)
    U[0, 0] = 1; U[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    U[1, 6] = 1; U[3, 6] = -1
    
    # V (B)
    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1
    V[1, 2] = 1; V[3, 2] = -1
    V[2, 3] = 1; V[0, 3] = -1
    V[3, 4] = 1
    V[0, 5] = 1; V[1, 5] = 1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W (C)
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[2, 2] = 1; W[2, 4] = 1
    W[1, 1] = 1; W[1, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def _build_strassen_3x5x6(n, m, p):
    sU, sV, sW = _get_strassen_2x2x2()
    rank = 84
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for k_block in range(3):
        k0 = k_block * 2
        k1 = k0 + 1
        
        # Block 1: A[0:2, 0:2] * B[0:2, k0:k1] -> C[0:2, k0:k1]
        # Strassen 2x2
        for c in range(7):
            U[0*5 + 0, col] = sU[0, c]
            U[0*5 + 1, col] = sU[1, c]
            U[1*5 + 0, col] = sU[2, c]
            U[1*5 + 1, col] = sU[3, c]
            V[0*6 + k0, col] = sV[0, c]
            V[0*6 + k1, col] = sV[1, c]
            V[1*6 + k0, col] = sV[2, c]
            V[1*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
            
        # Block 2: A[0:2, 2:4] * B[2:4, k0:k1] -> C[0:2, k0:k1]
        # Strassen 2x2
        for c in range(7):
            U[0*5 + 2, col] = sU[0, c]
            U[0*5 + 3, col] = sU[1, c]
            U[1*5 + 2, col] = sU[2, c]
            U[1*5 + 3, col] = sU[3, c]
            V[2*6 + k0, col] = sV[0, c]
            V[2*6 + k1, col] = sV[1, c]
            V[3*6 + k0, col] = sV[2, c]
            V[3*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
            
        # Block 3: A[0:2, 4] * B[4, k0:k1] -> C[0:2, k0:k1]
        # Scalar products (rank 2)
        for i in range(2):
            for k in range(2):
                U[i*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + i, col] = 1
                col += 1
                
        # Block 4: A[2:4, 0:2] * B[0:2, k0:k1] -> C[2:4, k0:k1]
        # Strassen 2x2
        for c in range(7):
            U[2*5 + 0, col] = sU[0, c]
            U[2*5 + 1, col] = sU[1, c]
            U[3*5 + 0, col] = sU[2, c]
            U[3*5 + 1, col] = sU[3, c]
            V[0*6 + k0, col] = sV[0, c]
            V[0*6 + k1, col] = sV[1, c]
            V[1*6 + k0, col] = sV[2, c]
            V[1*6 + k1, col] = sV[3, c]
            W[k0*3 + 2, col] = sW[0, c]
            W[k0*3 + 3, col] = sW[1, c]
            W[k1*3 + 2, col] = sW[2, c]
            W[k1*3 + 3, col] = sW[3, c]
            col += 1
            
        # Block 5: A[2:4, 2:4] * B[2:4, k0:k1] -> C[2:4, k0:k1]
        # Strassen 2x2
        for c in range(7):
            U[2*5 + 2, col] = sU[0, c]
            U[2*5 + 3, col] = sU[1, c]
            U[3*5 + 2, col] = sU[2, c]
            U[3*5 + 3, col] = sU[3, c]
            V[2*6 + k0, col] = sV[0, c]
            V[2*6 + k1, col] = sV[1, c]
            V[3*6 + k0, col] = sV[2, c]
            V[3*6 + k1, col] = sV[3, c]
            W[k0*3 + 2, col] = sW[0, c]
            W[k0*3 + 3, col] = sW[1, c]
            W[k1*3 + 2, col] = sW[2, c]
            W[k1*3 + 3, col] = sW[3, c]
            col += 1
            
        # Block 6: A[2:4, 4] * B[4, k0:k1] -> C[2:4, k0:k1]
        # Scalar products (rank 2)
        for i in range(2):
            for k in range(2):
                U[(2+i)*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + (2+i), col] = 1
                col += 1
            
    return U, V, W, rank

def _build_schoolbook(n, m, p):
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

def _verify_fast(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)
