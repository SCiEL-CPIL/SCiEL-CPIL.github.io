import scipy
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Safety margin: 140s out of 150s limit
    deadline = time.time() + 140

    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    def verify(U, V, W, tol=1e-8):
        """
        Verify if U, V, W form a valid decomposition.
        U: (nm, r), V: (mp, r), W: (pn, r)
        Condition: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
        Also, for j != j', sum_t U[i*m+j, t] * V[j'*p+k, t] * W[k*n+i, t] = 0
        """
        max_err = 0.0
        r = U.shape[1]
        # Precompute products for speed if possible, but r is small enough for direct dot
        for i in range(n):
            for j in range(m):
                u_vec = U[i*m+j, :]
                for k in range(p):
                    v_vec = V[j*p+k, :]
                    w_vec = W[k*n+i, :]
                    # Diagonal term
                    val_diag = np.dot(u_vec, v_vec * w_vec)
                    max_err = max(max_err, abs(val_diag - 1))
                    
                    # Off-diagonal terms
                    for jp in range(m):
                        if jp == j:
                            continue
                        v_vec_jp = V[jp*p+k, :]
                        val_off = np.dot(u_vec, v_vec_jp * w_vec)
                        max_err = max(max_err, abs(val_off))
        return max_err

    def make_residuals(r):
        def residuals(x):
            U = x[:r*n*m].reshape(n*m, r)
            V = x[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
            W = x[r*n*m + r*m*p:].reshape(p*n, r)
            
            res = []
            for i in range(n):
                for j in range(m):
                    u_vec = U[i*m+j, :]
                    for k in range(p):
                        w_vec = W[k*n+i, :]
                        # Diagonal
                        v_vec = V[j*p+k, :]
                        res.append(np.dot(u_vec, v_vec * w_vec) - 1)
                        # Off-diagonal
                        for jp in range(m):
                            if jp == j:
                                continue
                            v_vec_jp = V[jp*p+k, :]
                            res.append(np.dot(u_vec, v_vec_jp * w_vec))
            return np.array(res)
        return residuals

    # Try ranks from 83 down to a lower bound
    # Lower bound is max(nm, mp, pn) = 28
    # We start from rank_school - 1 down to 28
    # If we find a solution, we break.
    
    # Given the difficulty, we only try a few ranks close to schoolbook
    # and use very few attempts to stay within time limits.
    
    search_ranks = range(rank_school - 1, 27, -1)
    
    for r in search_ranks:
        if time.time() > deadline - 10:
            break
            
        n_vars = r * (n*m + m*p + p*n)
        
        # Heuristic: skip if variables are too many or too few
        if n_vars < n * m * p:
             # This condition in the original code was: if n_vars < n*m*p: continue
             # This means if r*(nm+mp+pn) < nmp, skip.
             # 84 * 61 = 5124. 84*84 is not the case.
             # For r=83: 83*61 = 5063 < 5124? No, 83*61 = 5063. 3*4*7=84.
             # Wait, n*m*p = 84.
             # n_vars = r * (12 + 28 + 21) = r * 61.
             # For r=1, n_vars=61 < 84.
             # The condition `if n_vars < n*m*p` in the original code seems to be a heuristic to avoid under-determined systems?
             # Actually, for r=84, n_vars = 84*61 = 5124 >> 84.
             # For r=1, n_vars = 61 < 84.
             # This check skips very low ranks.
             pass

        # Try a few random initializations
        n_attempts = 5 if r > 60 else 2
        
        found = False
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            if attempt == 0:
                # Start from schoolbook truncated? Or random?
                # Random is better for finding global minima, but schoolbook truncated might be closer to a solution if one exists near schoolbook.
                # Let's try random first.
                x0 = np.random.randn(n_vars)
            else:
                x0 = np.random.randn(n_vars) * 0.5

            try:
                residuals = make_residuals(r)
                # Use least_squares with LM method
                result = scipy.optimize.least_squares(
                    residuals, x0, method='lm', max_nfev=2000, xtol=1e-10, ftol=1e-10
                )
                
                if result.cost < 1e-8:
                    sol = result.x
                    U_sol = sol[:r*n*m].reshape(n*m, r)
                    V_sol = sol[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
                    W_sol = sol[r*n*m + r*m*p:].reshape(p*n, r)
                    
                    # Check continuous error
                    err_cont = verify(U_sol, V_sol, W_sol)
                    
                    if err_cont < 1e-6:
                        # Try to snap to half-integers
                        U_half = np.round(U_sol * 2) / 2
                        V_half = np.round(V_sol * 2) / 2
                        W_half = np.round(W_sol * 2) / 2
                        
                        err_half = verify(U_half, V_half, W_half)
                        
                        if err_half < 1e-6:
                            best_rank = r
                            best_U, best_V, best_W = U_half, V_half, W_half
                            found = True
                            break
                        
                        # Try integers
                        U_int = np.round(U_sol)
                        V_int = np.round(V_sol)
                        W_int = np.round(W_sol)
                        
                        err_int = verify(U_int, V_int, W_int)
                        if err_int < 1e-6:
                            best_rank = r
                            best_U, best_V, best_W = U_int, V_int, W_int
                            found = True
                            break
                            
            except Exception:
                pass
        
        if found:
            break

    # Final check and return
    # Ensure the returned arrays are of the correct type (float or int, but list of lists of numbers)
    # The interface expects 'U', 'V', 'W' with integer/half-integer entries.
    
    # If we found a better rank, best_U/V/W are already snapped.
    # If not, we use schoolbook.
    
    U_out = best_U
    V_out = best_V
    W_out = best_W
    
    # Convert to lists
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
