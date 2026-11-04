"""Matrix multiplication 4,4,7 — rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _schoolbook():
    """Build the rank-112 schoolbook decomposition for 4x4x7 matmul."""
    n, m, p = 4, 4, 7
    rank = n * m * p  # 112
    U = [[0.0] * rank for _ in range(n * m)]   # 16 x 112
    V = [[0.0] * rank for _ in range(m * p)]   # 28 x 112
    W = [[0.0] * rank for _ in range(p * n)]   # 28 x 112
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _build_tensor():
    """Build the target multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    n, m, p = 4, 4, 7
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _als_decompose(T, target_rank, n_restarts=3, max_iter=50, tol=1e-10):
    """
    Alternating least squares for tensor decomposition.
    Returns (U, V, W, max_err) where max_err is the max absolute error.
    """
    nm, mp, pn = T.shape
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    T_flat_a = T.reshape(nm, mp * pn)
    T_flat_b = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T_flat_c = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    
    for restart in range(n_restarts):
        np.random.seed(42 + restart)
        U = np.random.randn(nm, target_rank) * 0.5
        V = np.random.randn(mp, target_rank) * 0.5
        W = np.random.randn(pn, target_rank) * 0.5
        
        for it in range(max_iter):
            # Update U: T_flat_a = U @ M_U where M_U[r, bc] = V[b,r]*W[c,r]
            M_U = np.zeros((target_rank, mp * pn))
            for r in range(target_rank):
                M_U[r] = np.outer(V[:, r], W[:, r]).reshape(-1)
            try:
                U_new = T_flat_a @ np.linalg.pinv(M_U)
            except np.linalg.LinAlgError:
                break
            
            # Update V: T_flat_b = V @ M_V where M_V[r, ac] = U[a,r]*W[c,r]
            M_V = np.zeros((target_rank, nm * pn))
            for r in range(target_rank):
                M_V[r] = np.outer(U[:, r], W[:, r]).reshape(-1)
            try:
                V_new = T_flat_b @ np.linalg.pinv(M_V)
            except np.linalg.LinAlgError:
                break
            
            # Update W: T_flat_c = W @ M_W where M_W[r, ab] = U[a,r]*V[b,r]
            M_W = np.zeros((target_rank, nm * mp))
            for r in range(target_rank):
                M_W[r] = np.outer(U[:, r], V[:, r]).reshape(-1)
            try:
                W_new = T_flat_c @ np.linalg.pinv(M_W)
            except np.linalg.LinAlgError:
                break
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            err = 0.0
            for r in range(target_rank):
                contrib = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                err = max(err, np.max(np.abs(T - contrib)))
            if err < tol:
                break
        
        # Compute final error
        err = 0.0
        approx = np.zeros_like(T)
        for r in range(target_rank):
            approx += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        err = np.max(np.abs(T - approx))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def construct():
    """Return factor matrices for 4x4x7 matrix multiplication tensor."""
    deadline = time.time() + 140
    
    # Start with schoolbook as safe baseline
    U_school, V_school, W_school = _schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = 112
    
    T = _build_tensor()
    
    # Try to find lower rank decompositions using ALS
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = 28
    # For 4x4x4, best known is 49 (vs 64 schoolbook)
    # For 4x4x7, let's try a few candidate ranks
    
    candidate_ranks = [28, 32, 36, 40, 44, 48, 56, 64, 80, 96, 104, 112]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            break
        
        U, V, W, err = _als_decompose(T, target_rank, n_restarts=3, max_iter=40, tol=1e-10)
        
        if err < 1e-9:
            # Found a valid decomposition
            best_rank = target_rank
            # Convert to list of lists with rounded values for cleanliness
            best_U = [[round(float(x), 10) for x in row] for row in U.tolist()]
            best_V = [[round(float(x), 10) for x in row] for row in V.tolist()]
            best_W = [[round(float(x), 10) for x in row] for row in W.tolist()]
            break  # Found the lowest possible rank, stop
    
    # If no better found, return schoolbook
    if best_rank == 112:
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": best_U, "V": best_V, "W": best_W}
