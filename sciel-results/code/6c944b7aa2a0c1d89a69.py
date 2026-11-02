import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 3, 6
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p  # 36
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    def find_2x3x3_rank(target_rank, max_trials=200):
        """Find a decomposition for 2x3x3 with given rank."""
        n_s, m_s, p_s = 2, 3, 3
        # Tensor T[i,j,k] = 1
        # We need sum_r U[i,r] V[j,r] W[k,r] = 1
        
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                return None
            
            np.random.seed(trial * 13 + 7)
            
            U = np.random.randn(n_s, target_rank) * 0.1
            V = np.random.randn(m_s, target_rank) * 0.1
            W = np.random.randn(p_s, target_rank) * 0.1
            
            # Add a structured component to help convergence
            # Schoolbook terms for 2x3x3
            # For each (i,j,k), add a term
            # But we have limited rank.
            
            def objective(params):
                U = params[:n_s*target_rank].reshape(n_s, target_rank)
                V = params[n_s*target_rank:n_s*target_rank + m_s*target_rank].reshape(m_s, target_rank)
                W = params[n_s*target_rank + m_s*target_rank:].reshape(p_s, target_rank)
                
                # Compute T_approx[i,j,k]
                # T_approx = sum_r U[:,r] V[:,r] W[:,r]
                # Vectorized:
                # U: n x R, V: m x R, W: p x R
                # T_approx[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # This is the trace of U^T V W^T? No.
                # T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
                
                T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
                T_target = np.ones((n_s, m_s, p_s))
                diff = T_approx - T_target
                return np.sum(diff**2)
            
            initial_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(objective, initial_params, method='L-BFGS-B', maxiter=200, 
                                  options={'maxls': 50, 'gtol': 1e-6, 'ftol': 1e-9})
                if result.fun < 1e-6:
                    U_final = result.x[:n_s*target_rank].reshape(n_s, target_rank)
                    V_final = result.x[n_s*target_rank:n_s*target_rank + m_s*target_rank].reshape(m_s, target_rank)
                    W_final = result.x[n_s*target_rank + m_s*target_rank:].reshape(p_s, target_rank)
                    
                    # Verify
                    T_approx = np.einsum('ir,jr,kr->ijk', U_final, V_final, W_final)
                    if np.allclose(T_approx, np.ones((n_s, m_s, p_s)), atol=1e-5):
                        return U_final, V_final, W_final
            except Exception:
                pass
        return None

    # Try to find rank 12 decomposition for 2x3x3
    decomp = find_2x3x3_rank(12, max_trials=100)
    
    if decomp:
        U_sub, V_sub, W_sub = decomp
        R = 12
        total_rank = 2 * R # 24
        
        U_full = [[0] * total_rank for _ in range(n * m)]
        V_full = [[0] * total_rank for _ in range(m * p)]
        W_full = [[0] * total_rank for _ in range(p * n)]
        
        # Block 1: k in 0..2
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    U_full[i * m + j][r] = U_sub[i][j]
            for j in range(m):
                for k in range(3):
                    V_full[j * p + k][r] = V_sub[j][k]
            for k in range(3):
                for i in range(n):
                    W_full[k * n + i][r] = W_sub[k][i]
                    
        # Block 2: k in 3..5
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    U_full[i * m + j][R + r] = U_sub[i][j]
            for j in range(m):
                for k in range(3):
                    V_full[j * p + (k + 3)][R + r] = V_sub[j][k]
            for k in range(3):
                for i in range(n):
                    W_full[(k + 3) * n + i][R + r] = W_sub[k][i]
        
        # Verify full decomposition
        U_arr = np.array(U_full, dtype=float)
        V_arr = np.array(V_full, dtype=float)
        W_arr = np.array(W_full, dtype=float)
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    dec_val = 0.0
                    for r in range(total_rank):
                        dec_val += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                    if abs(dec_val - 1.0) > 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = total_rank
            best_U = U_full
            best_V = V_full
            best_W = W_full

    return {"U": best_U, "V": best_V, "W": best_W}
