import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    def to_list(mat):
        return mat.tolist()

    # Schoolbook solution (Rank 24)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {'U': to_list(U), 'V': to_list(V), 'W': to_list(W)}
    
    best_solution = get_schoolbook()

    # Verification function
    def verify(U, V, W):
        # U, V, W are lists of lists
        # Convert to numpy for speed
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Check if entries are integer or half-integer
        U_int = np.round(U_arr * 2)
        V_int = np.round(V_arr * 2)
        W_int = np.round(W_arr * 2)
        
        if not np.allclose(U_arr * 2, U_int, atol=1e-6): return False
        if not np.allclose(V_arr * 2, V_int, atol=1e-6): return False
        if not np.allclose(W_arr * 2, W_int, atol=1e-6): return False
        
        # Check tensor decomposition
        # Sum_r U_int * V_int * W_int should be 8
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.sum(U_int[u_idx] * V_int[v_idx] * W_int[w_idx])
                    if s != 8:
                        return False
        return True

    # Optimization function for a given rank R
    def try_rank(R):
        if time.time() > deadline:
            return None
            
        n_vars = (n * m + m * p + p * n) * R
        U_size = n * m * R
        V_size = m * p * R
        
        def objective(x):
            U = x[:U_size].reshape(n * m, R)
            V = x[U_size:U_size + V_size].reshape(m * p, R)
            W = x[U_size + V_size:].reshape(p * n, R)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        s = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                        err += (s - 1.0) ** 2
            return err

        best_x = None
        best_err = float('inf')
        
        # Random restarts
        num_restarts = 100
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
                
            x0 = np.random.normal(0, 1, n_vars)
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
                if res.fun < best_err:
                    best_err = res.fun
                    best_x = res.x
            except Exception:
                continue
        
        if best_x is None:
            return None
            
        # Reconstruct U, V, W
        U = best_x[:U_size].reshape(n * m, R)
        V = best_x[U_size:U_size + V_size].reshape(m * p, R)
        W = best_x[U_size + V_size:].reshape(p * n, R)
        
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        
        # Verify
        if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
            return {'U': to_list(U_rounded), 'V': to_list(V_rounded), 'W': to_list(W_rounded)}
            
        return None

    # Try ranks 20 to 23
    for R in [20, 21, 22, 23]:
        if time.time() > deadline - 10:
            break
            
        sol = try_rank(R)
        if sol is not None:
            return sol
            
    return best_solution
# EVOLVE-BLOCK-END
