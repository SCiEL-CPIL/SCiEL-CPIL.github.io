import numpy as np
import time
import scipy.optimize as opt

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    deadline = time.time() + 140  # 10s safety margin
    
    # Construct the target tensor for standard matrix multiplication
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # Reshape to (n, m, p) for the einsum 'ijc, jkc, kic -> ijk'
    T_target_flat = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target_flat[u, v, w] = 1.0
                
    # The einsum 'ijc, jkc, kic -> ijk' produces a tensor of shape (n, m, p)
    # We need to map the flattened indices to the 3D indices i, j, k
    # The 3D tensor T_3d[i, j, k] corresponds to T_flat[i*m+j, j*p+k, k*n+i]
    T_target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[i, j, k] = T_target_flat[u, v, w]
                
    # Check if T_target is all ones
    # If so, rank 1 is the answer
    if np.allclose(T_target, np.ones((n, m, p))):
        U = np.ones((n * m, 1))
        V = np.ones((m * p, 1))
        W = np.ones((p * n, 1))
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W, rank
    
    # Check if a decomposition is valid
    def check_valid(U, V, W, tol=1e-10):
        r = U.shape[1]
        if r == 0:
            return False
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T_rec, T_target, atol=tol)
        
    def snap_to_half_int(A):
        return np.round(A * 2) / 2.0

    # Try to find a lower rank decomposition using optimization
    def try_optimize(rank, time_budget=10, seed=None):
        """Try to find a valid decomposition of given rank using optimization."""
        if time.time() > deadline - 5:
            return None
            
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(int(time.time() * 1000) % 100000)

        # Initialize with small random values
        scale = 0.5
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
        
        lambda_penalty = 50.0 # Strong penalty for non-half-integers
        
        def objective(params):
            U_flat, V_flat, W_flat = params
            U_3d = U_flat.reshape(n, m, rank)
            V_3d = V_flat.reshape(m, p, rank)
            W_3d = W_flat.reshape(p, n, rank)
            T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            err = T_rec - T_target
            data_err = np.sum(err ** 2)
            
            # Penalty for deviating from half-integers
            pen_U = np.sum((2 * U_3d - np.round(2 * U_3d)) ** 2)
            pen_V = np.sum((2 * V_3d - np.round(2 * V_3d)) ** 2)
            pen_W = np.sum((2 * W_3d - np.round(2 * W_3d)) ** 2)
            
            return data_err + lambda_penalty * (pen_U + pen_V + pen_W)
        
        def gradient(params):
            U_flat, V_flat, W_flat = params
            U_3d = U_flat.reshape(n, m, rank)
            V_3d = V_flat.reshape(m, p, rank)
            W_3d = W_flat.reshape(p, n, rank)
            T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            err = T_rec - T_target
            
            dU_data = 2 * np.einsum('ijk,jkc->ijc', err, W_3d)
            dV_data = 2 * np.einsum('ijk,kic->jkc', err, U_3d)
            dW_data = 2 * np.einsum('ijk,jkc->kic', err, V_3d)
            
            # Subgradient approximation for penalty
            dU_pen = lambda_penalty * 4 * (2 * U_3d - np.round(2 * U_3d))
            dV_pen = lambda_penalty * 4 * (2 * V_3d - np.round(2 * V_3d))
            dW_pen = lambda_penalty * 4 * (2 * W_3d - np.round(2 * W_3d))
            
            dU = dU_data + dU_pen
            dV = dV_data + dV_pen
            dW = dW_data + dW_pen
            
            return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        
        # Flatten parameters
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            start_time = time.time()
            max_iter = 50
            result = opt.minimize(
                objective,
                x0,
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': max_iter, 'maxfun': 500, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result.fun < 1e-3:
                U_opt = result.x[:n*m*rank].reshape(n, m, rank)
                V_opt = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m, p, rank)
                W_opt = result.x[n*m*rank + m*p*rank :].reshape(p, n, rank)
                
                U_snap = snap_to_half_int(U_opt)
                V_snap = snap_to_half_int(V_opt)
                W_snap = snap_to_half_int(W_opt)
                
                if check_valid(U_snap, V_snap, W_snap, tol=1e-6):
                    return U_snap.reshape(n*m, rank), V_snap.reshape(m*p, rank), W_snap.reshape(p*n, rank), rank
        except Exception:
            pass
        
        return None
    
    best_result = None
    best_rank = n * m * p
    
    ranks_to_try = list(range(60, 72))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        seeds = [42, 123, 999]
        
        found = False
        for seed in seeds:
            if time.time() > deadline - 5:
                break
                
            result = try_optimize(rank, time_budget=5, seed=seed)
            if result is not None:
                U, V, W, r = result
                best_result = (U, V, W, r)
                best_rank = r
                found = True
                break
        
        if found:
            break
    
    if best_result is None:
        U, V, W, rank = schoolbook()
    else:
        U, V, W, rank = best_result
    
    if not check_valid(U, V, W, tol=1e-8):
        U, V, W, rank = schoolbook()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
