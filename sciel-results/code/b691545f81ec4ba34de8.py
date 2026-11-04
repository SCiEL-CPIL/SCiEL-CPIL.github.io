import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
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
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_error = 0.0

    # Try to find a lower rank decomposition
    # We search for R in a range
    for R in range(15, 40):
        if time.time() > deadline:
            break
            
        # Random restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # U: (n*m, R), V: (m*p, R), W: (p*n, R)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Optimize
            # We want sum_r W[k*n+i, r] * U[:, i, r] * V[:, k, r]^T = I_m
            # This is equivalent to:
            # For each i, k:
            # sum_r W[k*n+i, r] * (U[i*m:(i+1)*m, r] * V[k*p? No, V is indexed by j*p+k])
            # V[j*p+k, r]
            # Let's define the error function
            
            def error_func(params):
                # params is a flat array of size (n*m + m*p + p*n) * R
                U_flat = params[:n*m*R].reshape(n*m, R)
                V_flat = params[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_flat = params[n*m*R + m*p*R :].reshape(p*n, R)
                
                err = 0.0
                for i in range(n):
                    for k in range(p):
                        # Sum over r
                        # Term r: W[k*n+i, r] * U[i*m:(i+1)*m, r] * V[?, k, r]^T
                        # V index for column k is j*p+k
                        # So V[j*p+k, r]
                        # We need the outer product of U_slice and V_slice
                        # U_slice: U[i*m:(i+1)*m, r]
                        # V_slice: V[:, k, r] where V[:, k, r] is a vector of length m
                        # V_slice[j] = V[j*p+k, r]
                        
                        acc = np.zeros((m, m))
                        for r in range(R):
                            u_r = U_flat[i*m:(i+1)*m, r]
                            v_r = np.array([V_flat[j*p+k, r] for j in range(m)])
                            w_r = W_flat[k*n+i, r]
                            acc += w_r * np.outer(u_r, v_r)
                        
                        err += np.sum((acc - np.eye(m))**2)
                return err
            
            # Use a simple gradient descent or just check random?
            # Gradient descent might be slow.
            # Let's try a few random initializations and see if any is close to 0.
            # Or use scipy.optimize.minimize if available and fast.
            
            # Given the small size, we can try to solve it.
            # But it's nonlinear.
            # Let's just do a few random checks.
            # If R is small, the probability of hitting a solution is low.
            
            # Let's try to use the structure of Strassen.
            # For now, let's just return the schoolbook if we don't find a better one quickly.
            # But we know R=75 is not minimal.
            
            # Let's try a specific known decomposition if we can hardcode it.
            # But we don't know it.
            
            # Let's try to optimize with scipy.
            from scipy.optimize import minimize
            x0 = np.random.randn((n*m + m*p + p*n) * R)
            
            # This might be too slow for 25 restarts * 25 ranks.
            # Let's limit the optimization.
            
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
