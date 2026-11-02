import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Leave safety margin
    
    # Precompute indices for the tensor equations
    # Equation: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    # for all i in [0, n), j in [0, m), k in [0, p)
    
    def objective(params):
        # params is a flat vector of length r * (nm + mp + pn)
        r = len(params) // (n*m + m*p + p*n)
        if r == 0:
            return 0.0
        
        U = params[0:r*n*m].reshape(r, n*m)
        V = params[r*n*m : r*n*m + r*m*p].reshape(r, m*p)
        W = params[r*n*m + r*m*p :].reshape(r, p*n)
        
        # Calculate residual
        # We want sum_l U[i*m+j, l] V[j*p+k, l] W[k*n+i, l] = 1
        # This is a batch of dot products.
        # Let's compute the tensor T_calc
        # T_calc[i, j, k] = sum_l U.T[l, i*m+j] * V.T[l, j*p+k] * W.T[l, k*n+i]
        
        # Efficient computation:
        # For each l, we have outer products? No, it's a contraction.
        # Residual vector R of size n*m*p
        R = np.zeros(n*m*p)
        for l in range(r):
            u_l = U[l]
            v_l = V[l]
            w_l = W[l]
            # For each i, j, k:
            # R[i*m*p + j*p + k] += u_l[i*m+j] * v_l[j*p+k] * w_l[k*n+i]
            # This loop is slow in Python. Vectorize.
            
            # u_l is shape (n*m)
            # v_l is shape (m*p)
            # w_l is shape (p*n)
            
            # We can reshape u_l to (n, m), v_l to (m, p), w_l to (p, n)
            u_mat = u_l.reshape(n, m)
            v_mat = v_l.reshape(m, p)
            w_mat = w_l.reshape(p, n)
            
            # We need to compute T_l[i, j, k] = u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
            # And sum over l.
            # This is not a standard matrix product.
            
            # Iterate over i, j, k is O(nmp).
            # n=3, m=5, p=7 => 105 operations per l.
            # r is up to ~50. 50 * 105 = 5250 ops. Very fast.
            
            for i in range(n):
                for j in range(m):
                    u_val = u_mat[i, j]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = v_mat[j, k]
                        w_val = w_mat[k, i]
                        R[i*m*p + j*p + k] += u_val * v_val * w_val
                        
        # Target is 1 for all entries
        error = R - 1.0
        return np.sum(error**2)

    def snap_to_half_int(x):
        return np.round(x * 2) / 2.0

    def verify_exact(U, V, W):
        # Check if entries are integer/half-integer
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2)):
                return False
        
        # Check decomposition
        r = U.shape[0]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for l in range(r):
                        s += U[l, i*m+j] * V[l, j*p+k] * W[l, k*n+i]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    best_result = None
    best_rank = n * m * p
    current_best_U = [[0] * best_rank for _ in range(n * m)]
    current_best_V = [[0] * best_rank for _ in range(m * p)]
    current_best_W = [[0] * best_rank for _ in range(p * n)]
    # Initialize with schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                current_best_U[i*m+j][col] = 1
                current_best_V[j*p+k][col] = 1
                current_best_W[k*n+i][col] = 1
                col += 1

    # Search for lower ranks
    # Try ranks from 20 to 104
    for r in range(20, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale matters. Let's try small random values.
            np.random.seed(restart * 100 + r)
            U_init = np.random.randn(r, n*m) * 0.5
            V_init = np.random.randn(r, m*p) * 0.5
            W_init = np.random.randn(r, p*n) * 0.5
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(objective, params_init, method='Powell', 
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                # Extract U, V, W
                U_num = res.x[0:r*n*m].reshape(r, n*m)
                V_num = res.x[r*n*m : r*n*m + r*m*p].reshape(r, m*p)
                W_num = res.x[r*n*m + r*m*p :].reshape(r, p*n)
                
                # Snap to half-integers
                U_snap = snap_to_half_int(U_num)
                V_snap = snap_to_half_int(V_num)
                W_snap = snap_to_half_int(W_num)
                
                if verify_exact(U_snap, V_snap, W_snap):
                    # Convert to lists of floats (or ints if integer)
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    
                    # Check if they are integers to be safe, but float is fine if exact
                    # The prompt says "integer/half-integer entries"
                    # Python floats representing 0.5, 1.0, etc. are fine.
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
            else:
                # Even if not exact, keep the best numerical solution found so far?
                # No, we need exact integer/half-integer.
                pass

    # If no lower rank found, return schoolbook
    return {"U": current_best_U, "V": current_best_V, "W": current_best_W}
# EVOLVE-BLOCK-END
