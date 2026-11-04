import numpy as np
import time
import scipy.optimize

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin from 150s
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Build the target tensor T
    T = np.zeros((size_u, size_v, size_w), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def check_decomposition(U, V, W, tol=1e-9):
        R = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(R, T, atol=tol):
            return False
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(R_r, T, atol=1e-9):
            return True
        return False

    def objective(params, rank):
        # params: flattened U, V, W
        U = params[:size_u * rank].reshape(size_u, rank)
        V = params[size_u * rank : size_u * rank + size_v * rank].reshape(size_v, rank)
        W = params[size_u * rank + size_v * rank :].reshape(size_w, rank)
        
        R = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.sum((R - T) ** 2)
        
        # Penalty for non-integer/half-integer values
        # We want U, V, W to be close to k/2
        penalty_U = np.sum((U * 2 - np.round(U * 2)) ** 2)
        penalty_V = np.sum((V * 2 - np.round(V * 2)) ** 2)
        penalty_W = np.sum((W * 2 - np.round(W * 2)) ** 2)
        
        # Weights: Tensor error is primary, penalty is secondary
        return error + 1e-2 * (penalty_U + penalty_V + penalty_W)

    def find_decomposition(rank, num_attempts=20, max_iter=100):
        if time.time() > deadline:
            return None
            
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
                
            np.random.seed(42 + attempt)
            # Random initialization
            U = np.random.randn(size_u, rank)
            V = np.random.randn(size_v, rank)
            W = np.random.randn(size_w, rank)
            
            # ALS phase to get close
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                    
                M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, rank)
                T_1 = T.reshape(size_u, size_v * size_w)
                U, _, _, _ = np.linalg.lstsq(M_vw.T, T_1.T, rcond=None)
                U = U.T
                
                M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, rank)
                T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
                V, _, _, _ = np.linalg.lstsq(M_ua.T, T_2.T, rcond=None)
                V = V.T
                
                M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, rank)
                T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
                W, _, _, _ = np.linalg.lstsq(M_uv.T, T_3.T, rcond=None)
                W = W.T
            
            # Polish with scipy.optimize
            if time.time() > deadline - 5:
                continue
                
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = scipy.optimize.minimize(objective, params, args=(rank,), method='L-BFGS-B', options={'maxiter': 50})
                if result.success:
                    U_opt = result.x[:size_u * rank].reshape(size_u, rank)
                    V_opt = result.x[size_u * rank : size_u * rank + size_v * rank].reshape(size_v, rank)
                    W_opt = result.x[size_u * rank + size_v * rank :].reshape(size_w, rank)
                    
                    if check_decomposition(U_opt, V_opt, W_opt):
                        U_opt = np.round(U_opt * 2) / 2
                        V_opt = np.round(V_opt * 2) / 2
                        W_opt = np.round(W_opt * 2) / 2
                        return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
            except Exception:
                pass
                
        return None

    best_result = None
    # Try rank 23 first
    for rank in [23, 24, 25]:
        if time.time() > deadline - 10:
            break
        res = find_decomposition(rank, num_attempts=10 if rank == 23 else 5, max_iter=50)
        if res is not None:
            best_result = res
            break

    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
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
