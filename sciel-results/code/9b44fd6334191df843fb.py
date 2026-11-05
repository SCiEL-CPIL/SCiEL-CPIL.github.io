"""Matrix multiplication 3x3 x 3x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]
n, m, p = 3, 3, 5

def _build_tensor_flat():
    """The multiplication tensor for 3x3 * 3x5 is all-ones of shape (3,3,5)."""
    return np.ones(n * m * p, dtype=np.float64)

def _residual_flat(U, V, W, rank, T_flat):
    """Compute ||T - T(U,V,W)||^2 where T[i,j,k]=sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
    T_recon = np.zeros(n * m * p, dtype=np.float64)
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i * m * p + j * p + k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    return np.sum((T_recon - T_flat) ** 2)

def _als_step_U(V, W, T_flat, rank):
    """Solve for U given V, W. Linear system: sum_r U[i*m+j,r] * V[j*p+k,r]*W[k*n+i,r] = T[i,j,k]"""
    # Build matrix: rows = (i,j,k) flattened, cols = (i*m+j)*rank + r
    num_eq = n * m * p
    num_unk = n * m * rank
    M = np.zeros((num_eq, num_unk))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq = i * m * p + j * p + k
                for r in range(rank):
                    col = (i * m + j) * rank + r
                    M[eq, col] = V[j * p + k, r] * W[k * n + i, r]
    sol, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
    return sol.reshape(n * m, rank)

def _als_step_V(U, W, T_flat, rank):
    num_eq = n * m * p
    num_unk = m * p * rank
    M = np.zeros((num_eq, num_unk))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq = i * m * p + j * p + k
                for r in range(rank):
                    col = (j * p + k) * rank + r
                    M[eq, col] = U[i * m + j, r] * W[k * n + i, r]
    sol, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
    return sol.reshape(m * p, rank)

def _als_step_W(U, V, T_flat, rank):
    num_eq = n * m * p
    num_unk = p * n * rank
    M = np.zeros((num_eq, num_unk))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq = i * m * p + j * p + k
                for r in range(rank):
                    col = (k * n + i) * rank + r
                    M[eq, col] = U[i * m + j, r] * V[j * p + k, r]
    sol, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
    return sol.reshape(p * n, rank)

def _try_rank(rank, T_flat, max_iters=300, tol=1e-10, deadline=None, ntries=8):
    best_U, best_V, best_W, best_res = None, None, None, np.inf
    
    for trial in range(ntries):
        if deadline is not None and time.time() > deadline:
            break
        np.random.seed(100 + trial * 7)
        U = np.random.randn(n * m, rank) * 0.3
        V = np.random.randn(m * p, rank) * 0.3
        W = np.random.randn(p * n, rank) * 0.3
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            U = _als_step_U(V, W, T_flat, rank)
            V = _als_step_V(U, W, T_flat, rank)
            W = _als_step_W(U, V, T_flat, rank)
            
            res = _residual_flat(U, V, W, rank, T_flat)
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if res < tol:
                return U, V, W
    
    if best_res < tol:
        return best_U, best_V, best_W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 125
    
    T_flat = _build_tensor_flat()
    
    # Schoolbook fallback
    sb_rank = n * m * p
    U_sb = np.zeros((n * m, sb_rank))
    V_sb = np.zeros((m * p, sb_rank))
    W_sb = np.zeros((p * n, sb_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best = (U_sb, V_sb, W_sb, sb_rank)
    
    # Search from lower bound
    for rank in range(15, sb_rank + 1):
        if time.time() > deadline - 3:
            break
        result = _try_rank(rank, T_flat, max_iters=200, tol=1e-10, deadline=deadline, ntries=6)
        if result is not None:
            U, V, W = result
            best = (U, V, W, rank)
            break
    
    U, V, W, _ = best
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
