import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    dimU = n * m
    dimV = m * p
    dimW = p * n
    
    # Build tensor T
    T = np.zeros((dimU, dimV, dimW), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    deadline = time.time() + 110
    best_result = None
    best_rank = None
    
    # Try ranks from 24 down to 18
    # 24 is a strong candidate (2 * 12)
    for rank in range(24, 17, -1):
        if time.time() > deadline:
            break
            
        result = _try_rank(rank, T, dimU, dimV, dimW, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break
            
    if best_result is not None:
        return best_result
        
    # Fallback to schoolbook 36
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

def _try_rank(rank, T, dimU, dimV, dimW, deadline):
    np.random.seed(42) # Fixed seed for reproducibility in trials, or random? 
    # Using random seed based on rank for variety
    np.random.seed(rank * 1000)
    
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline:
            return None
            
        # Initialize
        U = np.random.randn(dimU, rank)
        V = np.random.randn(dimV, rank)
        W = np.random.randn(dimW, rank)
        
        # Scale
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        # ALS
        converged = False
        for it in range(50):
            if time.time() > deadline:
                return None
                
            # Update U
            # T[a, b, c] = sum_l U[a, l] V[b, l] W[c, l]
            # For fixed a: T[a, :, :] = sum_l U[a, l] (V[:, l] * W[:, l])
            # Z_U[l, b, c] = V[b, l] * W[c, l]
            Z_U = np.einsum('bl,cl->bcl', V, W)
            Z_U_mat = Z_U.transpose(2, 0, 1).reshape(rank, dimV * dimW)
            T_flat = T.reshape(dimU, dimV * dimW)
            U, _, _, _ = np.linalg.lstsq(Z_U_mat.T, T_flat.T, rcond=None)
            U = U.T
            
            # Update V
            Z_V = np.einsum('al,cl->acl', U, W)
            Z_V_mat = Z_V.transpose(2, 0, 1).reshape(rank, dimU * dimW)
            T_flat_V = T.transpose(1, 0, 2).reshape(dimV, dimU * dimW)
            V, _, _, _ = np.linalg.lstsq(Z_V_mat.T, T_flat_V.T, rcond=None)
            V = V.T
            
            # Update W
            Z_W = np.einsum('al,bl->abl', U, V)
            Z_W_mat = Z_W.transpose(2, 0, 1).reshape(rank, dimU * dimV)
            T_flat_W = T.transpose(2, 0, 1).reshape(dimW, dimU * dimV)
            W, _, _, _ = np.linalg.lstsq(Z_W_mat.T, T_flat_W.T, rcond=None)
            W = W.T
            
            # Check residual
            T_decomp = np.einsum('al,bl,cl->abc', U, V, W)
            residual = np.linalg.norm(T - T_decomp)
            if residual < 1e-10:
                break
                
        if residual < 1e-10:
            result = _snap_to_rationals(U, V, W, T)
            if result is not None:
                return result
                
    return None

def _snap_to_rationals(U, V, W, T):
    def check_and_snap(mat, tol=1e-6):
        snapped = np.round(mat * 2) / 2
        if np.allclose(mat, snapped, atol=tol):
            return snapped
        return None

    U_s = check_and_snap(U)
    V_s = check_and_snap(V)
    W_s = check_and_snap(W)
    
    if U_s is not None and V_s is not None and W_s is not None:
        T_check = np.einsum('al,bl,cl->abc', U_s, V_s, W_s)
        if np.allclose(T_check, T, atol=1e-6):
            def to_list(arr):
                return [[float(x) for x in row] for row in arr]
            return {"U": to_list(U_s), "V": to_list(V_s), "W": to_list(W_s)}
            
    return None
