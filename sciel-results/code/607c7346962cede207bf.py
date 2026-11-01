# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i,j,k,l] = 1 if j==k else 0
    T_target = np.zeros((n, m, m, p))
    for j in range(m):
        for i in range(n):
            for l in range(p):
                T_target[i, j, j, l] = 1.0

    def compute_tensor(U, V, W):
        R = U.shape[1]
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                U_row = U[i*m+j]
                for k in range(m):
                    V_row = V[j*p+k]
                    prod_UV = U_row * V_row
                    for l in range(p):
                        W_row = W[k*n+l]
                        T[i,j,k,l] = np.dot(prod_UV, W_row)
        return T

    def loss_fn(params, R):
        U_size = n * m * R
        V_size = m * p * R
        W_size = p * n * R
        
        U_flat = params[:U_size]
        V_flat = params[U_size:U_size+V_size]
        W_flat = params[U_size+V_size:]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        T = compute_tensor(U, V, W)
        error = T - T_target
        return np.sum(error**2)

    def try_rank(R):
        best_loss = float('inf')
        best_params = None
        
        for attempt in range(10):
            np.random.seed(42 + attempt)
            init_scale = 0.5
            params0 = np.random.randn(n*m*R + m*p*R + p*n*R) * init_scale
            
            result = minimize(loss_fn, params0, args=(R,), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
                
        if best_loss > 1e-4:
            return None, None, None
        
        U_size = n * m * R
        V_size = m * p * R
        W_size = p * n * R
        
        U_flat = best_params[:U_size]
        V_flat = best_params[U_size:U_size+V_size]
        W_flat = best_params[U_size+V_size:]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        return U, V, W

    # Try ranks 8, 9, 10
    for R in [8, 9, 10]:
        U, V, W = try_rank(R)
        if U is not None:
            def snap_and_verify(U, V, W):
                # Try integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                if np.max(np.abs(U_int - U)) < 1e-5 and np.max(np.abs(V_int - V)) < 1e-5 and np.max(np.abs(W_int - W)) < 1e-5:
                    T_check = compute_tensor(U_int, V_int, W_int)
                    if np.max(np.abs(T_check - T_target)) < 1e-8:
                        return U_int, V_int, W_int
                
                # Try half-integers
                U_half = np.round(U * 2) / 2.0
                V_half = np.round(V * 2) / 2.0
                W_half = np.round(W * 2) / 2.0
                
                if np.max(np.abs(U_half - U)) < 1e-5 and np.max(np.abs(V_half - V)) < 1e-5 and np.max(np.abs(W_half - W)) < 1e-5:
                    T_check = compute_tensor(U_half, V_half, W_half)
                    if np.max(np.abs(T_check - T_target)) < 1e-8:
                        return U_half, V_half, W_half
                        
                return None, None, None

            U_snapped, V_snapped, W_snapped = snap_and_verify(U, V, W)
            if U_snapped is not None:
                U_list = [[float(x) for x in row] for row in U_snapped]
                V_list = [[float(x) for x in row] for row in V_snapped]
                W_list = [[float(x) for x in row] for row in W_snapped]
                return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: return schoolbook rank 12
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
# EVOLVE-BLOCK-END
