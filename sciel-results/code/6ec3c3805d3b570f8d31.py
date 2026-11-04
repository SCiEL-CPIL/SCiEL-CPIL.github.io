import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Target Tensor
    N, M, P = n, m, p
    T = np.zeros((N, M, P))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i, j, k] = 1.0
    T_flat = T.flatten()
    NM, MP, PN = N * M, M * P, P * N

    def check_exact(U, V, W):
        # U: (NM, r), V: (MP, r), W: (PN, r)
        # Reshape to (N, M, r), (M, P, r), (P, N, r)
        r = U.shape[1]
        U_r = U.reshape(N, M, r)
        V_r = V.reshape(M, P, r)
        W_r = W.reshape(P, N, r)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec.flatten(), T_flat, atol=1e-9)

    def objective_and_grad(x, r):
        size_U = N * M * r
        size_V = M * P * r
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:]
        
        U = U_flat.reshape(N, M, r)
        V = V_flat.reshape(M, P, r)
        W = W_flat.reshape(P, N, r)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        err = T_rec - T
        loss = np.sum(err**2)
        
        grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
        grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
        grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def try_rank_optimized(r, n_restarts=1):
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        total_size = size_U + size_V + size_W
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            np.random.seed(42 + r * 100 + restart)
            # Scale initialization
            x0 = np.random.randn(total_size) * 0.5
            
            try:
                res = minimize(objective_and_grad, x0, args=(r,), method='L-BFGS-B', jac=True, 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20})
                
                if res.fun < 1e-6:
                    U_flat = res.x[:size_U]
                    V_flat = res.x[size_U:size_U+size_V]
                    W_flat = res.x[size_U+size_V:]
                    
                    U = U_flat.reshape(N, M, r)
                    V = V_flat.reshape(M, P, r)
                    W = W_flat.reshape(P, N, r)
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if check_exact(U_rounded, V_rounded, W_rounded):
                        return (U_rounded.flatten(axis=1).tolist(), 
                                V_rounded.flatten(axis=1).tolist(), 
                                W_rounded.flatten(axis=1).tolist())
                    
                    # Try rounding to integers
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if check_exact(U_int, V_int, W_int):
                        return (U_int.flatten(axis=1).tolist(), 
                                V_int.flatten(axis=1).tolist(), 
                                W_int.flatten(axis=1).tolist())
            except Exception:
                continue
        return None

    # Fallback Schoolbook
    U_fallback = [[0] * rank_schoolbook for _ in range(NM)]
    V_fallback = [[0] * rank_schoolbook for _ in range(MP)]
    W_fallback = [[0] * rank_schoolbook for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U_fallback[i * M + j][col] = 1
                V_fallback[j * P + k][col] = 1
                W_fallback[k * N + i][col] = 1
                col += 1

    # Search for lower rank
    # Prioritize lower ranks with more restarts
    # Rank 53 is a known candidate for 3x4x6?
    # Let's try 53, 54, 55, 56, 57, 58, 59, 60 with 10 restarts each
    # Then 61-71 with 1 restart each
    
    ranks_to_try = list(range(53, 61)) + list(range(61, 72))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        n_restarts = 10 if r < 61 else 1
        
        res = try_rank_optimized(r, n_restarts)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
# EVOLVE-BLOCK-END
