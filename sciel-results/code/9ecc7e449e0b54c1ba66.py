import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    """Check U,V,W of given shapes reconstruct the matmul tensor exactly."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    if recon.shape != T.shape:
        return False
    return np.allclose(recon, T, atol=1e-9)

def _build_T(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _try_rank15_structured(deadline):
    """
    Try to find a rank-15 decomposition using a structured approach.
    
    For 2x3 * 3x3, we can use the following idea:
    - Use 9 rank terms for the "diagonal" part (schoolbook)
    - Use 6 additional rank terms to handle the off-diagonal parts
      by sharing computations between the two rows.
    """
    n, m, p = 2, 3, 3
    U_dim, V_dim, W_dim = n*m, m*p, p*n
    r = 15
    T = _build_T(n, m, p)
    
    # Initialize with schoolbook for first 9 terms
    U = np.zeros((U_dim, r))
    V = np.zeros((V_dim, r))
    W = np.zeros((W_dim, r))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= r:
                    break
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Optimize the remaining columns
    # Use alternating least squares on all 15 columns
    
    for it in range(5000):
        if time.time() > deadline:
            break
        
        # Update W
        for c in range(W_dim):
            M = np.zeros((U_dim * V_dim, r))
            for ridx in range(r):
                M[:, ridx] = np.outer(U[:, ridx], V[:, ridx]).ravel()
            try:
                W[c, :] = np.linalg.lstsq(M, T[:,:,c].ravel(), rcond=None)[0]
            except:
                pass
        
        # Update U
        for a in range(U_dim):
            M = np.zeros((V_dim * W_dim, r))
            for ridx in range(r):
                M[:, ridx] = np.outer(V[:, ridx], W[:, ridx]).ravel()
            try:
                U[a, :] = np.linalg.lstsq(M, T[a,:,:].ravel(), rcond=None)[0]
            except:
                pass
        
        # Update V
        for b in range(V_dim):
            M = np.zeros((U_dim * W_dim, r))
            for ridx in range(r):
                M[:, ridx] = np.outer(U[:, ridx], W[:, ridx]).ravel()
            try:
                V[b, :] = np.linalg.lstsq(M, T[:,b,:].ravel(), rcond=None)[0]
            except:
                pass
        
        if it % 500 == 0:
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(recon - T))
            if err < 1e-10:
                break
    
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(recon - T))
    
    if err < 1e-8:
        # Try rounding
        for scale in [2, 4, 8, 16]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            
            recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            err_r = np.max(np.abs(recon_r - T))
            
            if err_r < 1e-9:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    def schoolbook():
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
    
    # Try rank 15
    if time.time() < deadline - 10:
        result = _try_rank15_structured(deadline - 5)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    # Fallback to schoolbook rank 18
    return schoolbook()
