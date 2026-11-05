import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
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

    def snap_and_verify(U, V, W, max_denom=8):
        # Try snapping to rationals with small denominators
        for denom in [1, 2, 4, max_denom]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
            if np.allclose(R, T_target, atol=1e-9):
                return True, U_s.tolist(), V_s.tolist(), W_s.tolist()
        return False, None, None, None

    def obj_and_grad(params, rank):
        U_p = params[:dim_U*rank].reshape(dim_U, rank)
        V_p = params[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
        W_p = params[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
        
        R = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
        diff = R - T_target
        obj_val = np.sum(diff**2)
        
        grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V_p, W_p)
        grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U_p, W_p)
        grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U_p, V_p)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return obj_val, grad

    # Strategy 1: Try to find a good 3x3x3 decomposition first
    # Then use it to construct a 3x3x6 decomposition
    
    def find_333_decomp(target_rank=23, max_time=30):
        """Find a rank-r decomposition for 3x3x3 matrix multiplication."""
        dim3 = 9
        T333 = np.zeros((dim3, dim3, dim3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T333[i*3+j, j*3+k, k*3+i] = 1.0
        
        def obj333(params, r):
            U3 = params[:dim3*r].reshape(dim3, r)
            V3 = params[dim3*r:2*dim3*r].reshape(dim3, r)
            W3 = params[2*dim3*r:].reshape(dim3, r)
            R3 = np.einsum('ir,jr,kr->ijk', U3, V3, W3)
            diff = R3 - T333
            obj_val = np.sum(diff**2)
            grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V3, W3)
            grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U3, W3)
            grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U3, V3)
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return obj_val, grad
        
        best_err = float('inf')
        best_params = None
        
        for attempt in range(30):
            if time.time() > start_time + max_time:
                break
            
            np.random.seed(777 + attempt)
            U3 = np.random.randn(dim3, target_rank) * 0.5
            V3 = np.random.randn(dim3, target_rank) * 0.5
            W3 = np.random.randn(dim3, target_rank) * 0.5
            params = np.concatenate([U3.flatten(), V3.flatten(), W3.flatten()])
            
            try:
                res = opt.minimize(
                    lambda x: obj333(x, target_rank),
                    params,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x.copy()
                    if best_err < 1e-10:
                        break
            except:
                continue
        
        if best_err < 1e-8 and best_params is not None:
            U3 = best_params[:dim3*target_rank].reshape(dim3, target_rank)
            V3 = best_params[dim3*target_rank:2*dim3*target_rank].reshape(dim3, target_rank)
            W3 = best_params[2*dim3*target_rank:].reshape(dim3, target_rank)
            
            # Try to snap
            for denom in [1, 2, 4, 8]:
                U3s = np.round(U3 * denom) / denom
                V3s = np.round(V3 * denom) / denom
                W3s = np.round(W3 * denom) / denom
                R3s = np.einsum('ir,jr,kr->ijk', U3s, V3s, W3s)
                if np.allclose(R3s, T333, atol=1e-9):
                    return U3s, V3s, W3s, target_rank
            return U3, V3, W3, target_rank
        return None, None, None, None
    
    # Try to find 3x3x3 decomposition
    U3, V3, W3, r333 = find_333_decomp(target_rank=23, max_time=40)
    
    if U3 is not None and r333 is not None:
        # Construct 3x3x6 from two 3x3x3 blocks
        # U_full (9 x 2*r333) = [U3, U3]
        # V_full (18 x 2*r333): rows 0-8 (k=0,1,2) = [V3, 0], rows 9-17 (k=3,4,5) = [0, V3]
        # W_full (18 x 2*r333): rows 0-8 (k=0,1,2) = [W3, 0], rows 9-17 (k=3,4,5) = [0, W3]
        
        rank_full = 2 * r333
        U_full = np.zeros((dim_U, rank_full))
        V_full = np.zeros((dim_V, rank_full))
        W_full = np.zeros((dim_W, rank_full))
        
        U_full[:, :r333] = U3
        U_full[:, r333:] = U3
        
        V_full[:9, :r333] = V3
        V_full[9:, r333:] = V3
        W_full[:9, :r333] = W3
        W_full[9:, r333:] = W3
        
        # Verify
        R_full = np.einsum('ir,jr,kr->ijk', U_full, V_full, W_full)
        if np.allclose(R_full, T_target, atol=1e-9):
            # Try to snap
            success, U_list, V_list, W_list = snap_and_verify(U_full, V_full, W_full)
            if success:
                return {"U": U_list, "V": V_list, "W": W_list}
            else:
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Strategy 2: Direct optimization for 3x3x6 with various ranks
    best_solution = None
    
    for rank in range(46, 55):
        if time.time() > deadline - 15:
            break
        
        num_attempts = 25 if rank < 50 else 15
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 8:
                break
            
            np.random.seed(2000 + rank * 10 + attempt)
            scale = 0.8
            U = np.random.randn(dim_U, rank) * scale
            V = np.random.randn(dim_V, rank) * scale
            W = np.random.randn(dim_W, rank) * scale
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = opt.minimize(
                    lambda x: obj_and_grad(x, rank),
                    params,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                if res.fun < 1e-8:
                    U_opt = res.x[:dim_U*rank].reshape(dim_U, rank)
                    V_opt = res.x[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                    W_opt = res.x[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                    
                    success, U_list, V_list, W_list = snap_and_verify(U_opt, V_opt, W_opt)
                    if success:
                        best_solution = {"U": U_list, "V": V_list, "W": W_list}
                        return best_solution
                        
            except:
                pass
    
    # Fallback: schoolbook
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
