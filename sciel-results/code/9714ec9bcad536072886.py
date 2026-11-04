import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Build the target tensor T
    T = np.zeros((size_u, size_v, size_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def check_decomposition(U, V, W, tol=1e-9):
        R = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(R, T, atol=tol):
            return False
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(R_r, T, atol=1e-9):
            return True
        return False

    def polish_and_round(U, V, W):
        """High-precision polish: iterate ALS a few more times with tight tolerance,
        then round to nearest half-integer."""
        for _ in range(5):
            # Update U
            M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, -1)
            T_1 = T.reshape(size_u, size_v * size_w)
            U, _, _, _ = np.linalg.lstsq(M_vw.T, T_1.T, rcond=1e-15)
            U = U.T
            
            # Update V
            M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, -1)
            T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
            V, _, _, _ = np.linalg.lstsq(M_ua.T, T_2.T, rcond=1e-15)
            V = V.T
            
            # Update W
            M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, -1)
            T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
            W, _, _, _ = np.linalg.lstsq(M_uv.T, T_3.T, rcond=1e-15)
            W = W.T
        
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(R_r, T, atol=1e-9):
            return U_r, V_r, W_r
        return None

    def find_decomposition(rank, max_attempts=15, max_iter=80):
        if time.time() > deadline:
            return None
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
                
            # Structured initialization: use small random values to encourage sparsity
            scale = 0.5
            U = scale * np.random.randn(size_u, rank)
            V = scale * np.random.randn(size_v, rank)
            W = scale * np.random.randn(size_w, rank)
            
            # ALS iterations with convergence check
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                    
                # Update U
                M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, rank)
                T_1 = T.reshape(size_u, size_v * size_w)
                U, _, _, _ = np.linalg.lstsq(M_vw.T, T_1.T, rcond=None)
                U = U.T
                
                # Update V
                M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, rank)
                T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
                V, _, _, _ = np.linalg.lstsq(M_ua.T, T_2.T, rcond=None)
                V = V.T
                
                # Update W
                M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, rank)
                T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
                W, _, _, _ = np.linalg.lstsq(M_uv.T, T_3.T, rcond=None)
                W = W.T
                
                # Compute error
                R = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(R - T)
                
                if abs(prev_err - err) < 1e-10:
                    break
                prev_err = err
            
            # Try to polish and round
            polished = polish_and_round(U, V, W)
            if polished is not None:
                U_r, V_r, W_r = polished
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                
            # Also check without polishing
            if check_decomposition(U, V, W):
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        return None

    best_result = None
    # Try ranks from 23 upward
    for rank in [23, 24, 25, 26, 27, 28]:
        if time.time() > deadline - 15:
            break
        res = find_decomposition(rank, max_attempts=12, max_iter=60)
        if res is not None:
            best_result = res
            break

    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
