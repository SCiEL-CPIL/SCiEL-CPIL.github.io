import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to check validity
    def check_decomposition(U, V, W, rank):
        if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
            return False
        # Check integrality
        if not np.allclose(2 * U, np.round(2 * U), atol=1e-5): return False
        if not np.allclose(2 * V, np.round(2 * V), atol=1e-5): return False
        if not np.allclose(2 * W, np.round(2 * W), atol=1e-5): return False
        
        # Check correctness
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for jj in range(m):
                    u_vec = U[ii*m+jj, :]
                    v_vec = V[jj*p+kk, :]
                    w_val = W[kk*n+ii, :]
                    temp = u_vec * v_vec
                    acc += np.dot(temp, w_val)
                Cf[ii, kk] = acc
        
        if not np.allclose(C, Cf, atol=1e-5):
            return False
        return True

    # Search for low rank
    best_result = None
    for rank in [16, 18, 20, 22, 24, 26, 27]:
        if time.time() > 120.0: break
        found = False
        for restart in range(10):
            if time.time() > 120.0: break
            np.random.seed(rank * 100 + restart)
            params = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * 0.5
            
            def objective(params):
                U_flat = params[:n*m*rank].reshape(n*m, rank)
                V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U_flat[u_idx] * V_flat[v_idx], W_flat[w_idx])
                            err += (1.0 - val) ** 2
                return err

            try:
                res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
                if res.fun < 1e-6:
                    U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_opt = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    if check_decomposition(U_opt, V_opt, W_opt, rank):
                        U_int = np.round(2 * U_opt) / 2.0
                        V_int = np.round(2 * V_opt) / 2.0
                        W_int = np.round(2 * W_opt) / 2.0
                        if check_decomposition(U_int, V_int, W_int, rank):
                            best_result = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                            found = True
                            break
            except:
                pass
        if found:
            break

    if best_result:
        return best_result

    # Fallback to Strassen Rank 28
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    def apply_strassen(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_offset * 2 + c)
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = b_row_offset * 2 + r
                g_col = b_col_offset * 2 + c
                g_idx = g_row * p + g_col
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = r
                g_col = c_col_offset * 2 + c
                g_idx = g_col * n + g_row
                W[g_idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
