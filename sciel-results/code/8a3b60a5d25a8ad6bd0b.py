import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    # T is 9x9x9
    # T[idx_u(i,j), idx_v(j,k), idx_w(k,i)] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    def tensor_to_matrix(tensor, mode):
        # mode 0: U (first dim), mode 1: V (second dim), mode 2: W (third dim)
        # Flatten the other two dims
        if mode == 0:
            return tensor.reshape(tensor.shape[0], -1)
        elif mode == 1:
            return tensor.transpose(1, 0, 2).reshape(tensor.shape[1], -1)
        else:
            return tensor.transpose(2, 0, 1).reshape(tensor.shape[2], -1)

    def residual_func(params, R):
        # params is flattened U, V, W
        # U: 9xR, V: 9xR, W: 9xR
        size = 9 * R
        U = params[0:size].reshape(9, R)
        V = params[size:2*size].reshape(9, R)
        W = params[2*size:3*size].reshape(9, R)
        
        # Compute T_est = sum_r U[:, r] outer V[:, r] outer W[:, r]
        # Efficient way: 
        # T_est[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # This is a tensor contraction.
        # We can do: T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
        T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Residual is T - T_est
        return (T - T_est).ravel()

    def try_rank(R):
        # Initialize with rank-26 solution, drop last 3 terms if R < 26
        # Or random init
        # Let's try using the first R terms of the rank-26 solution
        U26, V26, W26 = _get_rank26_solution()
        
        if R <= 26:
            U_init = U26[:, :R]
            V_init = V26[:, :R]
            W_init = W26[:, :R]
        else:
            U_init = np.random.randn(9, R)
            V_init = np.random.randn(9, R)
            W_init = np.random.randn(9, R)
            
        params_init = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
        
        # Use a local search
        # Since the problem is non-convex, we might need multiple restarts
        # But time is limited.
        
        # Try a few restarts
        best_res = np.inf
        best_params = None
        
        # We can also try to "merge" terms by initializing with duplicates
        # For now, just use the truncated rank-26 solution
        
        # L-BFGS-B is good for large scale, but this is small (23*27=621 vars)
        # Nelder-Mead might be better for non-smooth? No, it's smooth.
        
        try:
            res = opt.minimize(
                lambda x: np.sum(residual_func(x, R)**2), 
                params_init, 
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12}
            )
            if res.success or res.fun < best_res:
                best_res = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
        if best_params is not None and best_res < 1e-6:
            # Check if we can snap to integers/half-integers
            U = best_params[0:9*R].reshape(9, R)
            V = best_params[9*R:18*R].reshape(9, R)
            W = best_params[18*R:27*R].reshape(9, R)
            
            # Snap to nearest integer or half-integer
            # Multiply by 2, round, divide by 2
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            # Verify
            T_snapped = np.einsum('ir,jr,kr->ijk', U_snapped, V_snapped, W_snapped)
            if np.allclose(T_snapped, T, atol=1e-9):
                return U_snapped, V_snapped, W_snapped, R
                
        return None

    def _get_rank26_solution():
        R = 26
        U = np.zeros((9, R), dtype=np.float64)
        V = np.zeros((9, R), dtype=np.float64)
        W = np.zeros((9, R), dtype=np.float64)
        
        def idx_u(i, j): return i * 3 + j
        def idx_v(j, k): return j * 3 + k
        def idx_w(k, i): return k * 3 + i
        
        # Strassen 7 terms
        r = 0
        U[idx_u(0,0), r] = 1; U[idx_u(1,1), r] = 1
        V[idx_v(0,0), r] = 1; V[idx_v(1,1), r] = 1
        W[idx_w(0,0), r] = 1; W[idx_w(1,1), r] = 1
        
        r = 1
        U[idx_u(1,0), r] = 1; U[idx_u(1,1), r] = 1
        V[idx_v(0,0), r] = 1
        W[idx_w(0,1), r] = 1; W[idx_w(1,1), r] = -1
        
        r = 2
        U[idx_u(0,0), r] = 1
        V[idx_v(0,1), r] = 1; V[idx_v(1,1), r] = -1
        W[idx_w(1,0), r] = 1; W[idx_w(1,1), r] = 1
        
        r = 3
        U[idx_u(1,1), r] = 1
        V[idx_v(1,0), r] = 1; V[idx_v(0,0), r] = -1
        W[idx_w(0,0), r] = 1; W[idx_w(0,1), r] = 1
        
        r = 4
        U[idx_u(0,0), r] = 1; U[idx_u(0,1), r] = 1
        V[idx_v(1,1), r] = 1
        W[idx_w(0,0), r] = -1; W[idx_w(1,0), r] = 1
        
        r = 5
        U[idx_u(1,0), r] = 1; U[idx_u(0,0), r] = -1
        V[idx_v(0,0), r] = 1; V[idx_v(0,1), r] = 1
        W[idx_w(1,1), r] = 1
        
        r = 6
        U[idx_u(0,1), r] = 1; U[idx_u(1,1), r] = -1
        V[idx_v(1,0), r] = 1; V[idx_v(1,1), r] = 1
        W[idx_w(0,0), r] = 1
        
        # Border 19 terms
        col = 7
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[idx_u(i,j), col] = 1
                    V[idx_v(j,k), col] = 1
                    W[idx_w(k,i), col] = 1
                    col += 1
                    
        return U, V, W

    # Try to find rank 23
    if time.time() < deadline - 10:
        sol = try_rank(23)
        if sol:
            U, V, W, R = sol
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to rank 26
    U, V, W = _get_rank26_solution()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
