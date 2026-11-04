import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    # Schoolbook fallback
    U_fallback = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fallback = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fallback = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][col] = 1
                V_fallback[j * p + k][col] = 1
                W_fallback[k * n + i][col] = 1
                col += 1
    
    # Target Tensor
    # Shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if consistent indices else 0
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    T_flat = T.flatten()
    N, M, P = T.shape
    NM, MP, PN = N * M, M * P, P * N
    
    def check_exact(U, V, W):
        # U: (NM, r), V: (MP, r), W: (PN, r)
        # Reconstruct T
        U_r = U.reshape(N, M, -1)
        V_r = V.reshape(M, P, -1)
        W_r = W.reshape(P, N, -1)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec.flatten(), T_flat, atol=1e-9)
    
    # Implement Optimization using L-BFGS-B
    from scipy.optimize import minimize
    
    def objective_and_grad(x, r):
        # x is flat vector of length 54*r
        # Split into U, V, W
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:]
        
        U = U_flat.reshape(N, M, r)
        V = V_flat.reshape(M, P, r)
        W = W_flat.reshape(P, N, r)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        err = T_rec - T
        loss = np.sum(err**2)
        
        # Gradient
        grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
        grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
        grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad
    
    def try_rank_optimized(r, seed=0):
        np.random.seed(seed)
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        x0 = np.random.randn(size_U + size_V + size_W) * 0.1
        
        # L-BFGS-B
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20})
        
        if res.success or res.fun < 1e-8:
            # Extract solution
            U_flat = res.x[:size_U]
            V_flat = res.x[size_U:size_U+size_V]
            W_flat = res.x[size_U+size_V:]
            
            U = U_flat.reshape(N, M, r)
            V = V_flat.reshape(M, P, r)
            W = W_flat.reshape(P, N, r)
            
            # Round to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Check exactness
            if check_exact(U_rounded, V_rounded, W_rounded):
                return U_rounded.flatten(axis=1).tolist(), V_rounded.flatten(axis=1).tolist(), W_rounded.flatten(axis=1).tolist()
            
            # Try rounding to integer
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if check_exact(U_int, V_int, W_int):
                return U_int.flatten(axis=1).tolist(), V_int.flatten(axis=1).tolist(), W_int.flatten(axis=1).tolist()
                
        return None
    
    # Search for lower rank
    # Start from 71 down to 50
    for r in range(rank_schoolbook - 1, 49, -1):
        if time.time() - start_time > 130:
            break
        try:
            res = try_rank_optimized(r)
            if res:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
        except Exception:
            continue
            
    # Fallback
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}

# EVOLVE-BLOCK-END
