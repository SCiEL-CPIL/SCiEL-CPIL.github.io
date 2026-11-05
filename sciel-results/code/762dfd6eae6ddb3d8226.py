import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

# Strassen decomposition for 3x3x3
# Reference: https://en.wikipedia.org/wiki/Strassen%27s_matrix_multiplication_algorithm
# The 19 rank-1 tensors for 3x3x3.
# Indices i, j, k in 0..2.
# U[i*3+j], V[j*3+k], W[k*3+i]
# We need to map this to the general form.

def get_strassen_3x3x3():
    # Returns U, V, W for 3x3x3 multiplication
    # U shape (9, 19), V shape (9, 19), W shape (9, 19)
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # The 19 products for Strassen 3x3x3
    # A = [[a,b,c],[d,e,f],[g,h,i]]
    # B = [[j,k,l],[m,n,o],[p,q,r]]
    # C = A*B
    
    # Standard Strassen 2x2x2 uses 7.
    # 3x3x3 uses 19.
    # We can generate them or hardcode.
    # Hardcoding is safer.
    
    # From literature:
    # The 19 terms are:
    # 1. a*j + b*m + c*p -> C_00? No, C_00 = a*j + b*m + c*p.
    # This is schoolbook for one element.
    # Strassen combines them.
    
    # Let's use a known set of 19 vectors.
    # U, V, W are 9x19 matrices.
    
    # Since I don't have the exact 19 vectors memorized, I will generate a valid 19-rank decomposition
    # by solving the system for 3x3x3 with r=19 using random init and least_squares,
    # or just use schoolbook for 3x3x3 (rank 27) and then try to reduce?
    # No, 19 is the known minimum.
    
    # To be safe and fast, I will use a random search for the full 3x3x7 problem.
    # If I find a rank < 63, it's an improvement.
    # Schoolbook is 63.
    # Random search might find ~40-50.
    
    # Let's try to find a decomposition for 3x3x7 directly.
    
    pass

def construct():
    n, m, p = DIMENSIONS
    # Target: minimize rank
    # Start with a reasonable guess.
    # Lower bound for 3x3x3 is 19.
    # For 3x3x7, let's try ranks from 25 to 40.
    
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, keep 20s margin
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find lower rank
    # We will try ranks 20 to 50
    for r in range(20, 50):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Random init
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Define the residual function
            def residual(params):
                # params is flattened U, V, W
                U_flat = params[:n*m*r]
                V_flat = params[n*m*r : n*m*r + m*p*r]
                W_flat = params[n*m*r + m*p*r:]
                
                U = U_flat.reshape(n*m, r)
                V = V_flat.reshape(m*p, r)
                W = W_flat.reshape(p*n, r)
                
                res = np.zeros(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            res[idx] = val - 1.0
                            idx += 1
                return res
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                from scipy.optimize import least_squares
                result = least_squares(residual, params0, method='lm', max_nfev=1000)
                if result.success or result.fun is not None:
                    res_norm = np.linalg.norm(result.fun)
                    if res_norm < 1e-5:
                        # Found a solution
                        U_sol = result.x[:n*m*r].reshape(n*m, r)
                        V_sol = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                        W_sol = result.x[n*m*r + m*p*r:].reshape(p*n, r)
                        
                        # Try to snap to half-integers
                        # Multiply by 2 to check integers
                        U_snap = np.round(U_sol * 2) / 2
                        V_snap = np.round(V_sol * 2) / 2
                        W_snap = np.round(W_sol * 2) / 2
                        
                        # Verify snapped solution
                        res_snap = residual(np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()]))
                        if np.linalg.norm(res_snap) < 1e-9:
                            best_rank = r
                            best_U = U_snap
                            best_V = V_snap
                            best_W = W_snap
                            break # Found minimal for this r, try lower? No, r is increasing.
                        else:
                            # Use the floating point solution if it's very good
                            if r < best_rank:
                                best_rank = r
                                best_U = U_sol
                                best_V = V_sol
                                best_W = W_sol
            except Exception as e:
                pass
                
        if best_rank == r:
            # We found a solution for this r. Can we go lower?
            # We are iterating r upwards. If we find one at r, we can stop if we assume minimal?
            # No, we might find a better one at r-1 if we missed it.
            # But we started from 20. If we found one at 25, we should check 20-24?
            # We already checked them in previous iterations (if they didn't find a solution).
            # So we can break?
            # Actually, if we found a solution at r, it's the first one found.
            # So it's likely minimal or close.
            break

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
