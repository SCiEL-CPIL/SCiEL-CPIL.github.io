import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    time_limit = 110  # 10s margin
    
    # Target: Find rank r decomposition for 2x3x3
    # Then use it to build 2x3x6 with rank 2r
    # 2x3x3 dimensions: n=2, m=3, p=3
    
    n_sub, m_sub, p_sub = 2, 3, 3
    
    # Tensor T_sub of shape (n_sub*m_sub, m_sub*p_sub, p_sub*n_sub)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub), dtype=float)
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[i * m_sub + j, j * p_sub + k, k * n_sub + i] = 1.0
    
    def try_find_decomposition(r, max_time=20):
        """
        Attempts to find a rank r decomposition for T_sub.
        Returns (U, V, W) as lists of lists if found, else None.
        """
        if time.time() - start_time > time_limit:
            return None
            
        # Random initialization
        # Use small integers
        U = np.random.randint(-2, 3, size=(n_sub * m_sub, r)) * 0.5
        V = np.random.randint(-2, 3, size=(m_sub * p_sub, r)) * 0.5
        W = np.zeros((p_sub * n_sub, r))
        # Initialize W with standard basis to help convergence?
        # Or random
        W = np.random.randint(-2, 3, size=(p_sub * n_sub, r)) * 0.5
        
        def objective(params):
            U_flat = params[:n_sub * m_sub * r].reshape(n_sub * m_sub, r)
            V_flat = params[n_sub * m_sub * r : n_sub * m_sub * r + m_sub * p_sub * r].reshape(m_sub * p_sub, r)
            W_flat = params[n_sub * m_sub * r + m_sub * p_sub * r :].reshape(p_sub * n_sub, r)
            
            # Compute sum of outer products
            # T_comp[i, j, k] = sum_t U[i, t] * V[j, t] * W[k, t]
            T_comp = np.einsum('it,jt,kt->ijk', U_flat, V_flat, W_flat)
            diff = T_comp - T_sub
            return np.sum(diff * diff)
        
        # Optimization
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            from scipy.optimize import minimize
            # Use BFGS or L-BFGS-B
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 1000})
            
            if time.time() - start_time > time_limit:
                return None
                
            if res.success or res.fun < 1e-6:
                U_opt = res.x[:n_sub * m_sub * r].reshape(n_sub * m_sub, r)
                V_opt = res.x[n_sub * m_sub * r : n_sub * m_sub * r + m_sub * p_sub * r].reshape(m_sub * p_sub, r)
                W_opt = res.x[n_sub * m_sub * r + m_sub * p_sub * r :].reshape(p_sub * n_sub, r)
                
                # Snap to nearest half-integer
                U_int = np.round(U_opt * 2) / 2
                V_int = np.round(V_opt * 2) / 2
                W_int = np.round(W_opt * 2) / 2
                
                # Verify
                T_check = np.einsum('it,jt,kt->ijk', U_int, V_int, W_int)
                if np.allclose(T_check, T_sub, atol=1e-9):
                    # Convert to lists of lists
                    return U_int.tolist(), V_int.tolist(), W_int.tolist()
                    
        except Exception:
            pass
            
        # If optimization failed, try a few random restarts if time permits
        # But keep it simple for now
        
        return None

    # Try ranks from 15 upwards
    for r in [15, 16, 17]:
        if time.time() - start_time > time_limit:
            break
        res = try_find_decomposition(r)
        if res:
            U_sub, V_sub, W_sub = res
            # Construct global 2x3x6 decomposition
            global_rank = 2 * r
            U_global = [[0] * global_rank for _ in range(n * m)]
            V_global = [[0] * global_rank for _ in range(m * p)]
            W_global = [[0] * global_rank for _ in range(p * n)]
            
            # U: 6 x 2r
            # First r cols: U_sub
            # Next r cols: U_sub
            for t in range(r):
                for i in range(n * m):
                    U_global[i][t] = U_sub[i][t]
                    U_global[i][t + r] = U_sub[i][t]
                    
            # V: 18 x 2r
            # Rows 0-8: B0. Rows 9-17: B1.
            # Cols 0-r-1: V_sub on B0, 0 on B1.
            # Cols r-2r-1: 0 on B0, V_sub on B1.
            for t in range(r):
                for j in range(m_sub * p_sub): # 0..8
                    V_global[j][t] = V_sub[j][t]
                    V_global[j + m_sub * p_sub][t + r] = V_sub[j][t]
                    
            # W: 12 x 2r
            # Rows 0-5: C0. Rows 6-11: C1.
            # Cols 0-r-1: W_sub on C0, 0 on C1.
            # Cols r-2r-1: 0 on C0, W_sub on C1.
            for t in range(r):
                for k in range(p_sub * n_sub): # 0..5
                    W_global[k][t] = W_sub[k][t]
                    W_global[k + p_sub * n_sub][t + r] = W_sub[k][t]
                    
            return {"U": U_global, "V": V_global, "W": W_global}
            
    # Fallback
    return {"U": U_school, "V": V_school, "W": W_school}
