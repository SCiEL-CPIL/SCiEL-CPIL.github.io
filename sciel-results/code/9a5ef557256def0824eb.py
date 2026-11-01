import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Target Rank
    target_rank = 14
    
    best_result = None
    best_score = n * m * p  # Schoolbook rank
    
    # Generate Schoolbook as fallback
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    U_shape = (n*m, target_rank)
    V_shape = (m*p, target_rank)
    W_shape = (p*n, target_rank)
    
    def compute_tensor(U, V, W):
        """Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            # U[:,c] is shape (n*m,), reshape to (n,m)
            u_c = U[:, c].reshape(n, m)
            # V[:,c] is shape (m*p,), reshape to (m,p)
            v_c = V[:, c].reshape(m, p)
            # W[:,c] is shape (p*n,), reshape to (p,n)
            w_c = W[:, c].reshape(p, n)
            
            # For each i, j, k: T[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
            # We can compute this as: for each j, u_c[:,j] (n-vector) * v_c[j,:] (p-vector) gives n x p matrix
            # Then element-wise multiply with w_c.T (n x p matrix where w_c.T[i,k] = w_c[k,i])
            for j in range(m):
                # u_c[:, j] is shape (n,)
                # v_c[j, :] is shape (p,)
                # outer product gives (n, p)
                outer = np.outer(u_c[:, j], v_c[j, :])
                # w_c is shape (p, n), so w_c.T is shape (n, p) where w_c.T[i,k] = w_c[k,i]
                T[:, j, :] += outer * w_c.T
        return T
    
    def residuals(params):
        u_len = U_shape[0] * U_shape[1]
        v_len = V_shape[0] * V_shape[1]
        w_len = W_shape[0] * W_shape[1]
        
        U = params[:u_len].reshape(U_shape)
        V = params[u_len:u_len+v_len].reshape(V_shape)
        W = params[u_len+v_len:].reshape(W_shape)
        
        T = compute_tensor(U, V, W)
        return (T - 1).flatten()
    
    def verify_solution(U, V, W, atol=1e-9):
        """Verify that the solution exactly decomposes the tensor"""
        T = compute_tensor(U, V, W)
        return np.allclose(T, 1, atol=atol)
    
    def try_round_and_verify(U_sol, V_sol, W_sol):
        """Try rounding to nearest half-integer and verify"""
        # Try rounding to nearest integer
        U_int = np.round(U_sol)
        V_int = np.round(V_sol)
        W_int = np.round(W_sol)
        
        if verify_solution(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        # Try rounding to nearest half-integer
        U_half = np.round(U_sol * 2) / 2
        V_half = np.round(V_sol * 2) / 2
        W_half = np.round(W_sol * 2) / 2
        
        if verify_solution(U_half, V_half, W_half):
            return U_half, V_half, W_half
        
        return None
    
    # Try to find rank 14 decomposition
    from scipy.optimize import least_squares
    
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Random initialization
        params = np.random.uniform(-2, 2, size=(U_shape[0]*U_shape[1] + V_shape[0]*V_shape[1] + W_shape[0]*W_shape[1]))
        
        try:
            result = least_squares(residuals, params, method='trf', max_nfev=10000, xtol=1e-10, ftol=1e-10)
            
            if result.cost < 1e-8:
                u_len = U_shape[0] * U_shape[1]
                v_len = V_shape[0] * V_shape[1]
                
                U_sol = result.x[:u_len].reshape(U_shape)
                V_sol = result.x[u_len:u_len+v_len].reshape(V_shape)
                W_sol = result.x[u_len+v_len:].reshape(W_shape)
                
                rounded = try_round_and_verify(U_sol, V_sol, W_sol)
                if rounded is not None:
                    U_int, V_int, W_int = rounded
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                    
                # If rounding didn't work, try a polish step
                # Use the current solution as initial guess for a tighter solve
                result2 = least_squares(residuals, result.x, method='trf', max_nfev=20000, xtol=1e-12, ftol=1e-12)
                
                if result2.cost < 1e-10:
                    U_sol2 = result2.x[:u_len].reshape(U_shape)
                    V_sol2 = result2.x[u_len:u_len+v_len].reshape(V_shape)
                    W_sol2 = result2.x[u_len+v_len:].reshape(W_shape)
                    
                    rounded2 = try_round_and_verify(U_sol2, V_sol2, W_sol2)
                    if rounded2 is not None:
                        U_int, V_int, W_int = rounded2
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
        except Exception as e:
            pass
            
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
