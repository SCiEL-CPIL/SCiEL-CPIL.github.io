import numpy as np
import time
from scipy.optimize import minimize

def _build_target(n, m, p):
    """Build the target tensor T[i,j,k] = 1 for all i,j,k."""
    T = np.ones((n, m, p))
    return T

def _tensor_to_vec(T, n, m, p):
    """Flatten tensor to vector using convention: index = i*m*p + j*p + k."""
    return T.reshape(-1)

def _vec_to_tensor(v, n, m, p):
    return v.reshape((n, m, p))

def _decomp_to_tensor(U, V, W, n, m, p):
    """Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
    return T

def _verify(U, V, W, n, m, p, tol=1e-8):
    T = _decomp_to_tensor(U, V, W, n, m, p)
    return np.allclose(T, 1.0, atol=tol)

def _als_reduce_rank(U, V, W, n, m, p, target_rank, max_iter=100, tol=1e-10, time_limit=30):
    """
    Try to reduce rank using ALS. Start from current decomposition,
    project to target_rank, then optimize.
    """
    start = time.time()
    cur_rank = U.shape[1]
    
    if target_rank >= cur_rank:
        return U, V, W
    
    # Initialize: take first target_rank columns (or use SVD-based init)
    U0 = U[:, :target_rank].copy()
    V0 = V[:, :target_rank].copy()
    W0 = W[:, :target_rank].copy()
    
    # Flatten for optimization
    nU = n * m
    nV = m * p
    nW = p * n
    
    def objective(x):
        Ux = x[:nU*target_rank].reshape(nU, target_rank)
        Vx = x[nU*target_rank:nU*target_rank+nV*target_rank].reshape(nV, target_rank)
        Wx = x[nU*target_rank+nV*target_rank:].reshape(nW, target_rank)
        T = _decomp_to_tensor(Ux, Vx, Wx, n, m, p)
        return np.sum((T - 1.0)**2)
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    best_x = x0
    best_val = objective(x0)
    
    # Try multiple random initializations
    for trial in range(5):
        if time.time() - start > time_limit:
            break
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        # Add small noise
        noise = 0.01 * np.random.randn(len(x0))
        x0 += noise
        
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'maxfun': 5000})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
        if best_val < tol:
            break
    
    if best_val < tol:
        U_opt = best_x[:nU*target_rank].reshape(nU, target_rank)
        V_opt = best_x[nU*target_rank:nU*target_rank+nV*target_rank].reshape(nV, target_rank)
        W_opt = best_x[nU*target_rank+nV*target_rank:].reshape(nW, target_rank)
        return U_opt, V_opt, W_opt
    
    return None, None, None

def _block_decomposition_356(n, m, p):
    """Block decomposition with rank 44."""
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2
    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))
    
    S_U[0, 0] = 1; S_U[3, 0] = 1
    S_V[0, 0] = 1; S_V[3, 0] = 1
    S_W[0, 0] = 1; S_W[3, 0] = 1
    
    S_U[2, 1] = 1; S_U[3, 1] = 1
    S_V[0, 1] = 1
    S_W[2, 1] = 1
    
    S_U[0, 2] = 1
    S_V[1, 2] = 1; S_V[3, 2] = -1
    S_W[1, 2] = 1
    
    S_U[3, 3] = 1
    S_V[2, 3] = 1; S_V[0, 3] = -1
    S_W[0, 3] = 1; S_W[2, 3] = 1
    
    S_U[0, 4] = 1; S_U[1, 4] = 1
    S_V[3, 4] = 1
    S_W[0, 4] = 1; S_W[3, 4] = -1
    
    S_U[2, 5] = 1; S_U[0, 5] = -1
    S_V[0, 5] = 1; S_V[1, 5] = 1
    S_W[3, 5] = 1
    
    S_U[1, 6] = 1; S_U[3, 6] = -1
    S_V[2, 6] = 1; S_V[3, 6] = 1
    S_W[0, 6] = 1; S_W[3, 6] = -1
    
    col = 0
    # Block 1: 2x2x6
    for bk in range(3):
        ko = bk * 2
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m+j, col] += S_U[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    V[j*p+(ko+k), col] += S_V[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    W[(ko+k)*n+i, col] += S_W[k*2+i, r]
            col += 1
    
    # Block 2: 1x5x6, rank 1
    for j in range(5):
        U[2*m+j, col] = 1
    for j in range(5):
        for k in range(6):
            V[j*p+k, col] = 1
    for k in range(6):
        W[k*n+2, col] = 1
    col += 1
    
    # Block 3: 2x2x6
    for bk in range(3):
        ko = bk * 2
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m+(2+j), col] += S_U[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    V[(2+j)*p+(ko+k), col] += S_V[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    W[(ko+k)*n+i, col] += S_W[k*2+i, r]
            col += 1
    
    # Block 4: 2x1x6, rank 1
    for i in range(2):
        U[i*m+4, col] = 1
    for k in range(6):
        V[4*p+k, col] = 1
    for k in range(6):
        for i in range(2):
            W[k*n+i, col] = 1
    col += 1
    
    return U, V, W

def construct():
    n, m, p = 3, 5, 6
    start = time.time()
    deadline = start + 140
    
    U, V, W = _block_decomposition_356(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback: trivial
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try to reduce rank
    for target_rank in range(best_rank - 1, max(best_rank - 10, 20), -1):
        if time.time() > deadline - 10:
            break
        result = _als_reduce_rank(U, V, W, n, m, p, target_rank, 
                                  max_iter=50, tol=1e-10, 
                                  time_limit=min(15, deadline - time.time() - 10))
        if result[0] is not None:
            if _verify(result[0], result[1], result[2], n, m, p):
                best_U, best_V, best_W = result
                best_rank = target_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
