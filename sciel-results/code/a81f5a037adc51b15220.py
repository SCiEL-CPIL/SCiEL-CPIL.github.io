import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    n, m, p = 3, 3, 6
    dim_U = n * m  # 9
    dim_V = m * p  # 18
    dim_W = p * n  # 18
    
    # Construct target tensor T_target for 3x3x6
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

    def snap_and_verify(U, V, W, T):
        # Snap to nearest 0.5
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        # Check exactness
        R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        if np.allclose(R, T, atol=1e-9):
            return True, U_s, V_s, W_s
        return False, None, None, None

    # Helper to find 3x3x3 decomposition
    def find_333_decomposition(rank_333, max_attempts=10, time_limit=100.0):
        n3, m3, p3 = 3, 3, 3
        dU3, dV3, dW3 = 9, 9, 9
        T333 = np.zeros((dU3, dV3, dW3))
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    a = i * m3 + j
                    b = j * p3 + k
                    c = k * n3 + i
                    T333[a, b, c] = 1.0
        
        def obj_and_grad_333(params):
            U_p = params[:dU3*rank_333].reshape(dU3, rank_333)
            V_p = params[dU3*rank_333 : dU3*rank_333 + dV3*rank_333].reshape(dV3, rank_333)
            W_p = params[dU3*rank_333 + dV3*rank_333 :].reshape(dW3, rank_333)
            
            R = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
            diff = R - T333
            obj_val = np.sum(diff**2)
            
            grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V_p, W_p)
            grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U_p, W_p)
            grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U_p, V_p)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return obj_val, grad

        for attempt in range(max_attempts):
            if time.time() > start_time + time_limit:
                break
            if time.time() > deadline - 5:
                break
                
            np.random.seed(2000 + rank_333 * 10 + attempt)
            U = np.random.randn(dU3, rank_333)
            V = np.random.randn(dV3, rank_333)
            W = np.random.randn(dW3, rank_333)
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = opt.minimize(
                    obj_and_grad_333,
                    params,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9}
                )
                
                if res.fun < 1e-8:
                    U_opt = res.x[:dU3*rank_333].reshape(dU3, rank_333)
                    V_opt = res.x[dU3*rank_333 : dU3*rank_333 + dV3*rank_333].reshape(dV3, rank_333)
                    W_opt = res.x[dU3*rank_333 + dV3*rank_333 :].reshape(dW3, rank_333)
                    
                    success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt, T333)
                    if success:
                        return U_s, V_s, W_s
                        
            except Exception as e:
                pass
        return None

    # Helper to expand 3x3x3 to 3x3x6
    def expand_333_to_336(U333, V333, W333):
        rank_333 = U333.shape[1]
        rank_336 = 2 * rank_333
        
        U336 = np.hstack([U333, U333])
        
        V336 = np.zeros((18, rank_336))
        V336[:9, :rank_333] = V333
        V336[9:, rank_333:] = V333
        
        W336 = np.zeros((18, rank_336))
        W336[:9, :rank_333] = W333
        W336[9:, rank_333:] = W333
        
        success, U_s, V_s, W_s = snap_and_verify(U336, V336, W336, T_target)
        if success:
            return U_s, V_s, W_s
        return None

    # Try to find 3x3x3 rank 23
    sol_333 = find_333_decomposition(23, max_attempts=5, time_limit=60.0)
    if sol_333:
        U333, V333, W333 = sol_333
        sol_336 = expand_333_to_336(U333, V333, W333)
        if sol_336:
            U_s, V_s, W_s = sol_336
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Try to find 3x3x3 rank 24
    sol_333 = find_333_decomposition(24, max_attempts=5, time_limit=60.0)
    if sol_333:
        U333, V333, W333 = sol_333
        sol_336 = expand_333_to_336(U333, V333, W333)
        if sol_336:
            U_s, V_s, W_s = sol_336
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Fallback to schoolbook
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
