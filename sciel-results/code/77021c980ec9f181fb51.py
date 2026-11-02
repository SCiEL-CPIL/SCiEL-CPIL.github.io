import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Function to verify a decomposition
    def verify(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check if U[i,j] V[j,k] W[k,i] sums to 1
        # Flatten indices
        U_flat = U.reshape(n, m, -1)
        V_flat = V.reshape(m, p, -1)
        W_flat = W.reshape(p, n, -1)
        
        # Compute tensor T_ijk
        # T_ijk = sum_c U_ic V_jc W_ki
        # This is a batched matrix multiplication or einsum
        # U: (n, m, r), V: (m, p, r), W: (p, n, r)
        # We want sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Let's use einsum for efficiency
        # 'ijr,jkr,kin->ik' ? No, we want 'ijk'
        # U[i,j,r] * V[j,k,r] -> (i, j, k, r) sum over r? No, V is (m,p,r) so j is index of V.
        # U[i,j,r] * V[j,k,r] -> sum_r -> (i, j, k) ? No, V[j,k,r] has j as first index.
        # So U[i,j,r] * V[j,k,r] is not a standard dot product on r for fixed i,j,k because V depends on j.
        # Let's do it manually or with loops for small dims.
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum over c
                    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    T[i, j, k] = np.dot(U[i*m+j], V[j*p+k]) # This is sum U*V
                    # Wait, the formula is sum_c U_c V_c W_c.
                    # So it's element-wise product of three vectors, then sum.
                    # T[i,j,k] = sum_c (U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c])
        
        # Let's re-calculate correctly
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = U[i*m+j]
                    v_vec = V[j*p+k]
                    w_vec = W[k*n+i]
                    T[i, j, k] = np.sum(u_vec * v_vec * w_vec)
        
        # Check if T is all 1s
        return np.allclose(T, 1.0, atol=1e-6)

    # Function to create schoolbook decomposition
    def schoolbook():
        rank = rank_schoolbook
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Try to find a lower rank decomposition (e.g., rank 17)
    target_rank = rank_schoolbook - 1
    start_time = time.time()
    deadline = 110.0 # Leave 10s margin
    
    best_U, best_V, best_W = None, None, None
    
    # Attempt numerical search for target_rank
    if target_rank >= 1:
        # Number of variables
        num_vars = (n*m + m*p + p*n) * target_rank
        
        def residuals(params):
            U = params[:n*m*target_rank].reshape(n*m, target_rank)
            V = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
            
            # We want T to be 1
            err = T.flatten() - 1.0
            return err

        # Try a few random restarts
        max_restarts = 5
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize near schoolbook? 
            # Schoolbook has rank 18. We need 17.
            # We can take the first 17 columns of the schoolbook decomposition.
            U_sb, V_sb, W_sb = schoolbook()
            U_init = U_sb[:, :target_rank]
            V_init = V_sb[:, :target_rank]
            W_init = W_sb[:, :target_rank]
            
            # Add some noise
            noise = 0.1
            U_init = U_init + noise * np.random.randn(n*m, target_rank)
            V_init = V_init + noise * np.random.randn(m*p, target_rank)
            W_init = W_init + noise * np.random.randn(p*n, target_rank)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                sol = least_squares(residuals, params_init, method='lm', max_nfev=2000)
                if sol.success or np.linalg.norm(sol.fun) < 1e-4:
                    U_sol = sol.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = sol.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = sol.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                    
                    # Check if we can round to half-integers
                    # Multiply by 2, round, divide by 2
                    U_round = np.round(2 * U_sol) / 2
                    V_round = np.round(2 * V_sol) / 2
                    W_round = np.round(2 * W_sol) / 2
                    
                    if verify(U_round, V_round, W_round):
                        best_U, best_V, best_W = U_round, V_round, W_round
                        break
            except Exception:
                pass
    
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        U, V, W = schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
