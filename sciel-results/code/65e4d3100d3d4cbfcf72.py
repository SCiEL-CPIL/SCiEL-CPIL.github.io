import numpy as np
import time
import scipy

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin

    n, m, p = 3, 3, 6

    dim_U = n * m  # 9
    dim_V = m * p  # 18
    dim_W = p * n  # 18

    # Construct target tensor T_target
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def snap_and_verify(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        if np.allclose(R, T_target, atol=1e-9):
            return True, U_s.tolist(), V_s.tolist(), W_s.tolist()
        return False, None, None, None

    def get_schoolbook(rank):
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def try_rank_decomposition(target_rank, num_attempts=3):
        """
        Try to find a rank decomposition of the target tensor with the given rank.
        Uses random initialization and L-BFGS-B optimization.
        """
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(100 + target_rank * 10 + attempt)
            
            # Random initialization
            U = np.random.randn(dim_U, target_rank)
            V = np.random.randn(dim_V, target_rank)
            W = np.random.randn(dim_W, target_rank)
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def obj(params):
                U_p = params[:dim_U*target_rank].reshape(dim_U, target_rank)
                V_p = params[dim_U*target_rank : dim_U*target_rank + dim_V*target_rank].reshape(dim_V, target_rank)
                W_p = params[dim_U*target_rank + dim_V*target_rank :].reshape(dim_W, target_rank)
                R = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
                return np.sum((R - T_target)**2)
            
            try:
                res = scipy.optimize.minimize(
                    obj, params, method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                U_opt = res.x[:dim_U*target_rank].reshape(dim_U, target_rank)
                V_opt = res.x[dim_U*target_rank : dim_U*target_rank + dim_V*target_rank].reshape(dim_V, target_rank)
                W_opt = res.x[dim_U*target_rank + dim_V*target_rank :].reshape(dim_W, target_rank)
                
                success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    return {"U": U_s, "V": V_s, "W": W_s}
                    
            except Exception:
                pass
        return None

    # Try ranks from 7 up to 20
    for target_rank in [7, 8, 9, 10, 12, 14, 16, 18, 20]:
        if time.time() > deadline - 20:
            break
        
        result = try_rank_decomposition(target_rank)
        if result is not None:
            return result

    # Fallback to schoolbook
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
