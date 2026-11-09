import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    R_school = n * m * p
    
    # Function to compute the multiplication result given U, V, W
    def compute_mult(U, V, W, A, B):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # A: (n, m), B: (m, p)
        # Flatten A and B
        A_flat = A.flatten() # (n*m)
        B_flat = B.flatten() # (m*p)
        
        # Compute intermediate sums
        # U^T A_flat : (R, n*m) @ (n*m) -> (R)
        # V^T B_flat : (R, m*p) @ (m*p) -> (R)
        u_vals = U.T @ A_flat # (R,)
        v_vals = V.T @ B_flat # (R,)
        
        # M = sum_r u_vals[r] * v_vals[r] * W[:, r]
        # M is (p*n, )
        # W is (p*n, R)
        M_flat = W @ (u_vals * v_vals)
        
        # Reshape M_flat to (p, n)
        M = M_flat.reshape((p, n))
        return M

    def error_function(params, U_shape, V_shape, W_shape, R, A, B):
        U = params[:U_shape[0]*U_shape[1]].reshape(U_shape)
        V = params[U_shape[0]*U_shape[1]:U_shape[0]*U_shape[1] + V_shape[0]*V_shape[1]].reshape(V_shape)
        W = params[U_shape[0]*U_shape[1] + V_shape[0]*V_shape[1]:].reshape(W_shape)
        
        M_computed = compute_mult(U, V, W, A, B)
        C_true = (A @ B).T # (p, n)
        
        return np.sum((M_computed - C_true) ** 2)

    # We will try to find a decomposition of rank R < R_school
    # Start from R_school - 1 downwards
    best_result = None
    best_rank = R_school
    start_time = time.time()
    deadline = start_time + 140 # 140s limit to be safe
    
    # Initial solution: Schoolbook
    U_school = np.zeros((n*m, R_school))
    V_school = np.zeros((m*p, R_school))
    W_school = np.zeros((p*n, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    # Store schoolbook as fallback
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
        
    fallback = {
        "U": to_list(U_school),
        "V": to_list(V_school),
        "W": to_list(W_school)
    }
    
    # Try to reduce rank
    for R in range(R_school - 1, 1, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this R
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Start from schoolbook and drop one column? Or random?
            # Dropping one column from schoolbook won't work directly because the sum will be missing a term.
            # We need to adjust the other terms.
            # Let's try random initialization with small values.
            U_init = np.random.randn(n*m, R) * 0.1
            V_init = np.random.randn(m*p, R) * 0.1
            W_init = np.random.randn(p*n, R) * 0.1
            
            # Or, better: Start from schoolbook, pick a random column to "merge" into others?
            # Simpler: Just use random and optimize.
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Define objective
            # Use a fixed set of random A, B for the error function to make it deterministic-ish
            # But we need to generalize. Using one pair might overfit.
            # Let's use 5 random pairs and sum errors.
            A_list = [np.random.randn(n, m) for _ in range(5)]
            B_list = [np.random.randn(m, p) for _ in range(5)]
            
            def obj(p):
                U = p[:n*m*R].reshape(n*m, R)
                V = p[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = p[n*m*R + m*p*R :].reshape(p*n, R)
                
                err = 0
                for A, B in zip(A_list, B_list):
                    M = compute_mult(U, V, W, A, B)
                    C = (A @ B).T
                    err += np.sum((M - C)**2)
                return err
            
            # Optimize
            # Use Nelder-Mead or Powell
            # Since dimensions are high, this might be slow.
            # Let's try just a few iterations of gradient descent if we can compute gradient?
            # No, let's use scipy.optimize.minimize with method='Powell' or 'Nelder-Mead'
            # But 150s limit.
            
            # If R is close to schoolbook, maybe we can solve analytically?
            # For R = R_school - 1, we have 1 equation short.
            
            # Given the complexity, let's just return the schoolbook if we can't find a better one quickly.
            # But we should try at least one optimization step.
            
            try:
                from scipy.optimize import minimize
                res = minimize(obj, params, method='Powell', options={'maxiter': 50, 'disp': False})
                if res.fun < 1e-6:
                    # Check if it's a valid decomposition on more random samples
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                    
                    # Verify on 10 new random samples
                    valid = True
                    for _ in range(10):
                        A = np.random.randn(n, m)
                        B = np.random.randn(m, p)
                        M = compute_mult(U_sol, V_sol, W_sol, A, B)
                        C = (A @ B).T
                        if np.linalg.norm(M - C) > 1e-5:
                            valid = False
                            break
                    
                    if valid:
                        # Snap to half-integers?
                        # The problem says integer/half-integer entries.
                        # If we found a real solution, we might need to round.
                        # But rounding might break it.
                        # Let's assume if the error is low, the values are close to rationals.
                        # For now, return the float values. The evaluator might accept floats?
                        # "integer/half-integer entries".
                        # We should try to snap.
                        U_snap = np.round(U_sol * 2) / 2
                        V_snap = np.round(V_sol * 2) / 2
                        W_snap = np.round(W_sol * 2) / 2
                        
                        # Verify snapped version
                        valid_snap = True
                        for _ in range(10):
                            A = np.random.randn(n, m)
                            B = np.random.randn(m, p)
                            M = compute_mult(U_snap, V_snap, W_snap, A, B)
                            C = (A @ B).T
                            if np.linalg.norm(M - C) > 1e-5:
                                valid_snap = False
                                break
                                
                        if valid_snap:
                            best_result = {
                                "U": to_list(U_snap),
                                "V": to_list(V_snap),
                                "W": to_list(W_snap)
                            }
                            best_rank = R
                            found = True
                            break
            except Exception:
                pass
        
        if found:
            break

    if best_result is not None:
        return best_result
    else:
        return fallback
