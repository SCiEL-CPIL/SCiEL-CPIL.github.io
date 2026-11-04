"""Matrix multiplication 2,3,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]
TIME_LIMIT = 145

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 6, 27, 18
    
    # Build tensor T[i,j,k] where:
    # i = row_A * m + col_A (0..nm-1)
    # j = row_B * p + col_B (0..mp-1)
    # k = row_C * n + col_C (0..pn-1)
    # T[i,j,k] = 1 if col_A==row_B, row_A==row_C, col_B==col_C
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for row_A in range(n):
        for col_A in range(m):
            row_B = col_A
            if row_B >= m:
                continue
            for col_B in range(p):
                row_C = row_A
                col_C = col_B
                i = row_A * m + col_A
                j = row_B * p + col_B
                k = row_C * n + col_C
                T[i, j, k] = 1.0
    
    T_mat = T.reshape(nm, mp * pn)
    
    def verify(U, V, W, rank, scale=1.0):
        """Verify exact decomposition. Returns True if exact."""
        M = (V[:, :, None] * W[None, :, :]).reshape(mp * pn, rank)
        T_recon = U @ M.T / (scale ** 3)
        return np.max(np.abs(T_mat - T_recon)) < 1e-10
    
    def try_als(rank, num_restarts=5, max_iters=300, tol=1e-12):
        """ALS for tensor decomposition."""
        rng = np.random.default_rng(rank * 1000 + 42)
        for restart in range(num_restarts):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            U = rng.standard_normal((nm, rank))
            V = rng.standard_normal((mp, rank))
            W = rng.standard_normal((pn, rank))
            
            prev_res = np.inf
            for it in range(max_iters):
                if time.time() - start_time > TIME_LIMIT - 10:
                    break
                
                # Update U
                M_vw = (V[:, :, None] * W[None, :, :]).reshape(mp * pn, rank)
                U_new = np.linalg.lstsq(M_vw, T_mat, rcond=None)[0]
                
                # Update V
                # For each j: minimize ||T[:,j,:] - (U ⊙ W)[:,j,:] @ V[j,:]||
                # (U ⊙ W) has shape (nm, pn, rank)
                UW = U[:, None, :] * W[None, :, :]  # (nm, pn, rank)
                UW_mat = UW.reshape(nm * pn, rank)
                for j in range(mp):
                    T_j = T[:, j, :].reshape(nm * pn)
                    V[j] = np.linalg.lstsq(UW_mat, T_j, rcond=None)[0]
                
                # Update W
                UV = U[:, :, None] * V[None, :, :]  # (nm, mp, rank)
                UV_mat = UV.reshape(nm * mp, rank)
                for k in range(pn):
                    T_k = T[:, :, k].reshape(nm * mp)
                    W[k] = np.linalg.lstsq(UV_mat, T_k, rcond=None)[0]
                
                U = U_new
                # Recompute residual
                M = (V[:, :, None] * W[None, :, :]).reshape(mp * pn, rank)
                res = np.linalg.norm(T_mat - U @ M.T)
                if abs(prev_res - res) < tol * max(1, prev_res):
                    break
                prev_res = res
            
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            M = (V[:, :, None] * W[None, :, :]).reshape(mp * pn, rank)
            res = np.linalg.norm(T_mat - U @ M.T)
            if res < 1e-8:
                return U, V, W
        return None
    
    # Try ranks from lower bound upward
    best = None
    for rank in range(27, 55):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        result = try_als(rank, num_restarts=3, max_iters=200)
        if result is not None:
            U, V, W = result
            best = (U, V, W, rank)
            break
    
    if best is None:
        # Schoolbook fallback
        rank = nm * mp * pn
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W, rank = best
    
    # Convert to half-integers
    # Try scale factor 2
    U2 = np.round(U * 2).astype(int)
    V2 = np.round(V * 2).astype(int)
    W2 = np.round(W * 2).astype(int)
    if verify(U2.astype(float), V2.astype(float), W2.astype(float), rank, scale=2.0):
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    # Try scale factor 4
    U4 = np.round(U * 4).astype(int)
    V4 = np.round(V * 4).astype(int)
    W4 = np.round(W * 4).astype(int)
    if verify(U4.astype(float), V4.astype(float), W4.astype(float), rank, scale=4.0):
        return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
    
    # Try scale factor 8
    U8 = np.round(U * 8).astype(int)
    V8 = np.round(V * 8).astype(int)
    W8 = np.round(W * 8).astype(int)
    if verify(U8.astype(float), V8.astype(float), W8.astype(float), rank, scale=8.0):
        return {"U": U8.tolist(), "V": V8.tolist(), "W": W8.tolist()}
    
    # Fall back to schoolbook
    rank = nm * mp * pn
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
