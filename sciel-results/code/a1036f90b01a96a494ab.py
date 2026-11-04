import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    max_time = 130  # Leave margin
    
    # Target 2x3x2
    n, m, p = 2, 3, 2
    T = np.ones((n, m, p))
    
    def try_rank(R, max_restarts=1000):
        # Variables: U (6xR), V (6xR), W (4xR)
        # Total params: R * (6+6+4) = 16R
        num_params = R * (n*m + m*p + p*n)
        
        def unpack(params):
            U = params[:R*n*m].reshape(R, n*m).T
            V = params[R*n*m : R*(n*m + m*p)].reshape(R, m*p).T
            W = params[R*(n*m + m*p) :].reshape(R, p*n).T
            return U, V, W

        def residuals(params):
            U, V, W = unpack(params)
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            # T_approx[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            return (T - A_approx).flatten()

        for attempt in range(max_restarts):
            if time.time() - start_time > max_time:
                return None
            
            # Random init: Try wider range to escape local minima
            # Also try some structured inits
            if attempt % 10 == 0:
                # Structured-ish: small integers
                init_params = np.random.randint(-2, 3, size=num_params) * 0.5
            else:
                init_params = np.random.uniform(-2.0, 2.0, size=num_params)
            
            try:
                sol = least_squares(residuals, init_params, method='trf', max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                
                # Check if the solution is close enough to be snapped
                if sol.cost < 1e-6:
                    # Try rounding to 0.5
                    rounded_params = np.round(sol.x * 2) / 2.0
                    res_check = residuals(rounded_params)
                    if np.linalg.norm(res_check) < 1e-9:
                        U, V, W = unpack(rounded_params)
                        return {"U": U, "V": V, "W": W}
                    
                    # Try rounding to integer
                    rounded_int = np.round(sol.x)
                    res_int = residuals(rounded_int)
                    if np.linalg.norm(res_int) < 1e-9:
                        U, V, W = unpack(rounded_int)
                        return {"U": U, "V": V, "W": W}
                        
                    # If close but not exact, try local search around the snapped solution?
                    # Or just accept if very close and hope for the best? No, must be exact.
                    # Let's try a few perturbations of the snapped solution
                    for _ in range(5):
                        perturbed = rounded_params + np.random.normal(0, 0.1, size=num_params)
                        sol2 = least_squares(residuals, perturbed, method='trf', max_nfev=50)
                        if sol2.cost < 1e-8:
                             rounded_params2 = np.round(sol2.x * 2) / 2.0
                             res_check2 = residuals(rounded_params2)
                             if np.linalg.norm(res_check2) < 1e-9:
                                U, V, W = unpack(rounded_params2)
                                return {"U": U, "V": V, "W": W}

            except Exception:
                pass
        return None

    dec_232 = None
    # Try ranks 10, 11
    for R in [10, 11]:
        dec_232 = try_rank(R)
        if dec_232:
            break
    
    if dec_232:
        # Construct 2x3x4
        n_final, m_final, p_final = 2, 3, 4
        R_final = 2 * R
        U_232 = dec_232["U"]
        V_232 = dec_232["V"]
        W_232 = dec_232["W"]
        
        U_234 = np.zeros((n_final*m_final, R_final))
        V_234 = np.zeros((m_final*p_final, R_final))
        W_234 = np.zeros((p_final*n_final, R_final))
        
        # Block 1 (k=0,1)
        U_234[:, 0:R] = U_232
        V_234[0:6, 0:R] = V_232
        W_234[0:4, 0:R] = W_232
        
        # Block 2 (k=2,3)
        U_234[:, R:2*R] = U_232
        # V mapping
        for j in range(m_final):
            for k_prime in range(2):
                idx_234 = j * p_final + (k_prime + 2)
                idx_232 = j * 2 + k_prime
                V_234[idx_234, R:2*R] = V_232[idx_232, :]
        # W mapping
        W_234[4:8, R:2*R] = W_232
        
        # Verify
        U3 = U_234.reshape(n_final, m_final, R_final)
        V3 = V_234.reshape(m_final, p_final, R_final)
        W3 = W_234.reshape(p_final, n_final, R_final)
        T_final = np.ones((n_final, m_final, p_final))
        A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        if np.allclose(A_approx, T_final, atol=1e-6):
            return {
                "U": U_234.tolist(),
                "V": V_234.tolist(),
                "W": W_234.tolist()
            }
    
    # Fallback 24
    n, m, p = 2, 3, 4
    rank = 24
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
