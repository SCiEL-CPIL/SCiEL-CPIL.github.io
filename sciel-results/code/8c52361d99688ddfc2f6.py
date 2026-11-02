"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[a,b,c]."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(T_recon, T, atol=tol)

def _try_rank(r, n, m, p, max_iter=200, tol=1e-10, seed=42):
    """Try to find a rank-r decomposition using alternating least squares."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    U = rng.randn(n*m, r)
    V = rng.randn(m*p, r)
    W = rng.randn(p*n, r)
    
    for it in range(max_iter):
        # Update U
        for a in range(n*m):
            # T[a,:,:] = sum_r U[a,r] * V[:,r] outer W[:,r]
            # For fixed a: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # This is a linear system in U[a,:]
            # T[a,:,:] = U[a,:] @ (V.T @ W) ... no
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Let M[b,c,r] = V[b,r]*W[c,r], then T[a,:,:] = sum_r U[a,r]*M[:,:,r]
            # So T[a] = U[a] @ reshape(M, r, m*p*p) ... 
            # Actually: T[a,b,c] = sum_r U[a,r] * (V[:,r] * W[:,r])_{b,c}
            # where (V[:,r] * W[:,r]) is the outer product
            # So T[a] = sum_r U[a,r] * outer(V[:,r], W[:,r])
            # This is a linear system: T[a].reshape(-1) = U[a] @ A where A[r, (b,c)] = V[b,r]*W[c,r]
            A = np.einsum('br,cr->bcr', V, W).reshape(r, m*p*p)
            rhs = T[a].reshape(-1)
            U[a] = np.linalg.lstsq(A, rhs, rcond=None)[0]
        
        # Update V
        for b in range(m*p):
            A = np.einsum('ar,cr->acr', U, W).reshape(r, n*m*p*n)
            rhs = T[:,b,:].reshape(-1)
            V[b] = np.linalg.lstsq(A, rhs, rcond=None)[0]
        
        # Update W
        for c in range(p*n):
            A = np.einsum('ar,br->abr', U, V).reshape(r, n*m*m*p)
            rhs = T[:,:,c].reshape(-1)
            W[c] = np.linalg.lstsq(A, rhs, rcond=None)[0]
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        if err < tol:
            return U, V, W, True
    
    return U, V, W, False

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_school
    
    # Try ranks from 24 (lower bound) up to 47
    # Use a grid of candidate ranks
    candidate_ranks = list(range(24, 48))
    
    for r in candidate_ranks:
        if time.time() > deadline - 5:
            break
        for seed in range(3):
            if time.time() > deadline - 5:
                break
            try:
                U, V, W, success = _try_rank(r, n, m, p, max_iter=50, tol=1e-8, seed=seed)
                if success:
                    # Verify with higher precision
                    if _verify(U, V, W, n, m, p, tol=1e-9):
                        # Convert to list format
                        U_list = [[float(x) for x in row] for row in U]
                        V_list = [[float(x) for x in row] for row in V]
                        W_list = [[float(x) for x in row] for row in W]
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = r
                        break
            except Exception:
                continue
    
    return best_result
# EVOLVE-BLOCK-END
