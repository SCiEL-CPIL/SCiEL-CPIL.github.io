import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    n3, m3, p3 = 2, 3, 3
    
    def is_valid_decomp(U, V, W, n, m, p):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def round_half(x):
        return np.round(x * 2) / 2.0

    best_decomp = None
    best_rank = n3 * m3 * p3 # 18

    # Try to find lower rank for M_{2,3,3}
    for R3 in range(17, 15, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try a few random restarts
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            U3 = np.random.randn(n3*m3, R3)
            V3 = np.random.randn(m3*p3, R3)
            W3 = np.random.randn(p3*n3, R3)
            
            params = np.concatenate([U3.flatten(), V3.flatten(), W3.flatten()])
            
            def loss(params):
                U_f = params[0:n3*m3*R3].reshape(n3*m3, R3)
                V_f = params[n3*m3*R3:n3*m3*R3 + m3*p3*R3].reshape(m3*p3, R3)
                W_f = params[n3*m3*R3 + m3*p3*R3:].reshape(p3*n3, R3)
                loss_val = 0.0
                for i in range(n3):
                    for j in range(m3):
                        for k in range(p3):
                            val = np.sum(U_f[i*m3+j] * V_f[j*p3+k] * W_f[k*n3+i])
                            loss_val += (val - 1.0)**2
                return loss_val

            try:
                from scipy.optimize import minimize
                res = minimize(loss, params, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
                if res.fun < 1e-6:
                    U_opt = res.x[0:n3*m3*R3].reshape(n3*m3, R3)
                    V_opt = res.x[n3*m3*R3:n3*m3*R3 + m3*p3*R3].reshape(m3*p3, R3)
                    W_opt = res.x[n3*m3*R3 + m3*p3*R3:].reshape(p3*n3, R3)
                    
                    U_r = round_half(U_opt)
                    V_r = round_half(V_opt)
                    W_r = round_half(W_opt)
                    
                    if is_valid_decomp(U_r, V_r, W_r, n3, m3, p3):
                        best_decomp = (U_r, V_r, W_r)
                        best_rank = R3
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    if best_decomp is None:
        # Schoolbook fallback
        rank3 = n3 * m3 * p3
        U3 = np.zeros((n3*m3, rank3))
        V3 = np.zeros((m3*p3, rank3))
        W3 = np.zeros((p3*n3, rank3))
        col = 0
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    U3[i*m3+j, col] = 1
                    V3[j*p3+k, col] = 1
                    W3[k*n3+i, col] = 1
                    col += 1
        best_decomp = (U3, V3, W3)
        best_rank = rank3

    # Expand to M_{2,3,9}
    R = 3 * best_rank
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    U3, V3, W3 = best_decomp
    
    for l in range(3):
        U[:, 3*l : 3*(l+1)] = U3
        for j in range(m3):
            for k in range(p3):
                V[j*p + 3*l + k, 3*l : 3*(l+1)] = V3[j*p3 + k, :]
        for i in range(n3):
            for k in range(p3):
                W[(3*l + k)*n + i, 3*l : 3*(l+1)] = W3[k*n3 + i, :]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
