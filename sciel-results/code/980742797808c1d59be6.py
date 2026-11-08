import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Build tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    T_flat = T.reshape(n*m, m*p * p*n)
    
    def try_rank(r, max_iters=100, seed=42):
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        
        best_err = np.inf
        best_UVW = None
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: T_flat[a] = U[a] @ (V*W)^T
            # (V*W)[r, b, c] = V[b,r] * W[c,r], flattened to (r, m*p*p*n)
            VW = (V[:, :, None] * W[:, None, :]).reshape(r, m*p*p*n)
            try:
                U = np.linalg.lstsq(VW.T, T_flat.T, rcond=None)[0].T
            except:
                pass
            
            # Update V: T[b] (transposed) = V[b] @ (U*W)^T
            # (U*W)[r, a, c] = U[a,r] * W[c,r], flattened to (r, n*m*p*n)
            UW = (U[:, :, None] * W[:, None, :]).reshape(r, n*m*p*n)
            T_b = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
            try:
                V = np.linalg.lstsq(UW.T, T_b.T, rcond=None)[0].T
            except:
                pass
            
            # Update W: T[c] (transposed) = W[c] @ (U*V)^T
            # (U*V)[r, a, b] = U[a,r] * V[b,r], flattened to (r, n*m*m*p)
            UV = (U[:, :, None] * V[:, None, :]).reshape(r, n*m*m*p)
            T_c = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
            try:
                W = np.linalg.lstsq(UV.T, T_c.T, rcond=None)[0].T
            except:
                pass
            
            # Compute error
            VW = (V[:, :, None] * W[:, None, :]).reshape(r, m*p*p*n)
            T_recon = U @ VW.T
            err = np.linalg.norm(T_flat - T_recon)
            
            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-8:
                break
        
        if best_err < 1e-8:
            return best_UVW
        return None
    
    # Try various ranks
    for r in range(55, 35, -1):
        if time.time() > deadline - 15:
            break
        for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            if time.time() > deadline - 15:
                break
            result = try_rank(r, max_iters=50, seed=seed)
            if result is not None:
                U, V, W = result
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                VW = (V_r[:, :, None] * W_r[:, None, :]).reshape(r, m*p*p*n)
                T_recon = U_r @ VW.T
                err = np.linalg.norm(T_flat - T_recon)
                if err < 1e-9:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback: schoolbook
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
