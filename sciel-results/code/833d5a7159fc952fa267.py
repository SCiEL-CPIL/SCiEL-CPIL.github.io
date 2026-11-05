import numpy as np
import time
from scipy.optimize import differential_evolution

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 9
    M = m * p  # 12
    P = p * n  # 12
    
    # Build target tensor T[a, b, c]
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    start_time = time.time()
    deadline = 140.0

    def check_half_integer(U, V, W, tol=1e-8):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=tol):
            return True, U_r, V_r, W_r
        return False, None, None

    def objective(params, R):
        if time.time() > deadline:
            return 1e10
        U = params[:N*R].reshape(N, R)
        V = params[N*R:N*R+M*R].reshape(M, R)
        W = params[N*R+M*R:].reshape(P, R)
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_calc - T)**2)

    best_result = None

    # Try ranks from 26 to 30
    for R in range(26, 31):
        if time.time() > deadline:
            break
        
        # Define bounds for DE
        # Heuristic: values are usually between -2 and 2
        bounds = [(-2, 2)] * (N*R + M*R + P*R)
        
        # Run DE
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                args=(R,),
                maxiter=50, 
                popsize=15,
                tol=1e-6,
                seed=42
            )
            
            if res.fun < 1e-5:
                U, V, W = check_half_integer(
                    res.x[:N*R].reshape(N, R),
                    res.x[N*R:N*R+M*R].reshape(M, R),
                    res.x[N*R+M*R:].reshape(P, R)
                )
                if U is not None:
                    best_result = (R, U, V, W)
                    break
        except Exception:
            continue

    if best_result is not None:
        R, U_r, V_r, W_r = best_result
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
