import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0
    
    # Target tensor T[i,j,k] = 1
    T_target = np.ones((n, m, p))
    
    def verify_decomposition(U, V, W, R):
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T_recon, T_target, atol=1e-8)

    def get_rank_26_solution():
        """Robust Rank-26 Solution"""
        R = 26
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        def idx_u(i, j): return i * m + j
        def idx_v(j, k): return j * p + k
        def idx_w(k, i): return k * n + i

        # Strassen 7 terms
        r = 0
        U[idx_u(0, 0), r] = 1; U[idx_u(1, 1), r] = 1
        V[idx_v(0, 0), r] = 1; V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = 1; W[idx_w(1, 1), r] = 1
        
        r = 1
        U[idx_u(1, 0), r] = 1; U[idx_u(1, 1), r] = 1
        V[idx_v(0, 0), r] = 1
        W[idx_w(0, 1), r] = 1; W[idx_w(1, 1), r] = -1
        
        r = 2
        U[idx_u(0, 0), r] = 1
        V[idx_v(0, 1), r] = 1; V[idx_v(1, 1), r] = -1
        W[idx_w(1, 0), r] = 1; W[idx_w(1, 1), r] = 1
        
        r = 3
        U[idx_u(1, 1), r] = 1
        V[idx_v(1, 0), r] = 1; V[idx_v(0, 0), r] = -1
        W[idx_w(0, 0), r] = 1; W[idx_w(0, 1), r] = 1
        
        r = 4
        U[idx_u(0, 0), r] = 1; U[idx_u(0, 1), r] = 1
        V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = -1; W[idx_w(1, 0), r] = 1
        
        r = 5
        U[idx_u(1, 0), r] = 1; U[idx_u(0, 0), r] = -1
        V[idx_v(0, 0), r] = 1; V[idx_v(0, 1), r] = 1
        W[idx_w(1, 1), r] = 1
        
        r = 6
        U[idx_u(0, 1), r] = 1; U[idx_u(1, 1), r] = -1
        V[idx_v(1, 0), r] = 1; V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = 1

        # Border terms (19 terms)
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        
        return U, V, W

    def attempt_rank_R(R, max_restarts=5, max_iter=200):
        if time.time() > deadline - 5:
            return None
            
        U0, V0, W0 = get_rank_26_solution()
        
        if R >= 26:
            return U0[:, :R].copy(), V0[:, :R].copy(), W0[:, :R].copy()
        
        best_result = None
        best_error = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + restart)
            U_init = np.random.randn(n*m, R)
            V_init = np.random.randn(m*p, R)
            W_init = np.random.randn(p*n, R)
            
            # Normalize
            for r in range(R):
                U_init[:, r] /= np.linalg.norm(U_init[:, r])
                V_init[:, r] /= np.linalg.norm(V_init[:, r])
                W_init[:, r] /= np.linalg.norm(W_init[:, r])
            
            def objective(params):
                U_e = params[:n*m*R].reshape(n*m, R)
                V_e = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_e = params[2*n*m*R:].reshape(p*n, R)
                
                U_r = U_e.reshape(n, m, R)
                V_r = V_e.reshape(m, p, R)
                W_r = W_e.reshape(p, n, R)
                
                T_extra = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                diff = T_extra - T_target
                return np.sum(diff**2)
            
            p0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(
                    objective, 
                    p0, 
                    method='L-BFGS-B', 
                    options={'maxiter': max_iter, 'ftol': 1e-12}
                )
                
                if res.fun < best_error:
                    best_error = res.fun
                    if res.fun < 1e-8:
                        U_e_opt = res.x[:n*m*R].reshape(n*m, R)
                        V_e_opt = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                        W_e_opt = res.x[2*n*m*R:].reshape(p*n, R)
                        
                        # Snap to half-integers
                        U_snap = np.round(U_e_opt * 2) / 2
                        V_snap = np.round(V_e_opt * 2) / 2
                        W_snap = np.round(W_e_opt * 2) / 2
                        
                        if verify_decomposition(U_snap, V_snap, W_snap, R):
                            return U_snap, V_snap, W_snap
                            
                        best_result = (U_e_opt, V_e_opt, W_e_opt)
                        
            except Exception:
                pass
        
        return None

    # Try ranks from 23 up to 25
    for R in [23, 24, 25]:
        if time.time() > deadline - 10:
            break
            
        result = attempt_rank_R(R)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Rank 26
    U, V, W = get_rank_26_solution()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
