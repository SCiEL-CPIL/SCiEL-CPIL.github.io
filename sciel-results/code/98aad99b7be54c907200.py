"""Matrix multiplication 2x5 by 5x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def _build_tensor(n, m, p):
    """Build the 3-way tensor T of shape (n*m, m*p, p*n) where
    T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and k==k' and i==i' else 0."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r] = delta_{j,j'}*delta_{k,k'}*delta_{i,i'}"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    T_recon = np.zeros((n * m, m * p, p * n))
    for r in range(R):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    T_true = _build_tensor(n, m, p)
    max_err = np.max(np.abs(T_recon - T_true))
    return max_err < tol

def _make_schoolbook(n, m, p):
    """Create the schoolbook rank n*m*p decomposition."""
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

def _als_refine(U, V, W, T, T_mat1, T_mat2, T_mat3, n, m, p, max_iter=100, tol=1e-10, deadline=None):
    """Refine U, V, W using ALS. Returns (U, V, W, error)."""
    R = U.shape[1]
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        M_U = np.zeros((m * p * p * n, R))
        for r in range(R):
            M_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
        try:
            # Solve M_U U^T = T_mat1^T
            U, _, _, _ = np.linalg.lstsq(M_U, T_mat1.T, rcond=None)
            U = U.T
        except:
            pass
            
        # Update V
        M_V = np.zeros((n * m * p * n, R))
        for r in range(R):
            M_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
        try:
            V, _, _, _ = np.linalg.lstsq(M_V, T_mat2.T, rcond=None)
            V = V.T
        except:
            pass
            
        # Update W
        M_W = np.zeros((n * m * m * p, R))
        for r in range(R):
            M_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
        try:
            W, _, _, _ = np.linalg.lstsq(M_W, T_mat3.T, rcond=None)
            W = W.T
        except:
            pass
        
        # Check error every 5 steps
        if it % 5 == 4:
            T_recon = np.zeros((n * m, m * p, p * n))
            for r in range(R):
                T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.max(np.abs(T_recon - T))
            if err < tol:
                return U, V, W, err
                
    T_recon = np.zeros((n * m, m * p, p * n))
    for r in range(R):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    err = np.max(np.abs(T_recon - T))
    return U, V, W, err

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    T = _build_tensor(n, m, p)
    T_mat1 = T.reshape(n * m, m * p * p * n)
    T_mat2 = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T_mat3 = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
    
    # Start with schoolbook
    best_result = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # SVD lower bound
    s1 = np.linalg.svd(T_mat1, compute_uv=False)
    s2 = np.linalg.svd(T_mat2, compute_uv=False)
    s3 = np.linalg.svd(T_mat3, compute_uv=False)
    lower_bound = max(np.sum(s1 > 1e-10), np.sum(s2 > 1e-10), np.sum(s3 > 1e-10))
    
    # Try to reduce rank from best_rank - 1 down to lower_bound
    for R in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
            
        found = False
        n_restarts = 20
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            rng = np.random.RandomState(restart * 1000 + R)
            
            U = rng.randn(n * m, R)
            V = rng.randn(m * p, R)
            W = rng.randn(p * n, R)
            
            U, V, W, err = _als_refine(U, V, W, T, T_mat1, T_mat2, T_mat3, n, m, p, max_iter=50, tol=1e-9, deadline=deadline)
            
            if err < 1e-8:
                if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-7):
                    best_rank = R
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found = True
                    break
                    
        if found:
            # Continue to lower R
            pass
        else:
            # If not found, we can break if R is close to lower bound and time is low?
            # Or just continue to next R (lower).
            pass
            
    return best_result
