import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target: Find rank minimal decomposition for n x m * m x p
    # Lower bound: max(n*m, m*p, p*n)
    lb = max(n*m, m*p, p*n)
    
    # Strategy:
    # 1. Try to find a decomposition for 2x3 * 3x3 (sub-problem) with rank R_sub.
    #    If found, construct 2x3 * 3x6 with rank 2*R_sub.
    # 2. If not found or if 2*R_sub > current best, try direct search for 2x3 * 3x6.
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Helper to create tensor
    def make_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0
        return T

    # Helper to verify decomposition
    def verify(U, V, W, T, tol=1e-6):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        R = U.shape[1]
        # Compute sum of outer products
        # Efficient way:
        # T_approx[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # This is a tensor contraction.
        # We can do:
        # Temp = np.einsum('ur,vr->uvr', U, V)
        # T_approx = np.einsum('uvr,wr->uvw', Temp, W)
        try:
            Temp = np.einsum('ur,vr->uvr', U, V)
            T_approx = np.einsum('uvr,wr->uvw', Temp, W)
            return np.allclose(T_approx, T, atol=tol)
        except:
            return False

    def try_find(n, m, p, target_rank, time_limit):
        T = make_tensor(n, m, p)
        nm, mp, pn = n*m, m*p, p*n
        best_local = None
        
        # Time budget per attempt
        start_time = time.time()
        max_attempts = 50
        attempt_time = time_limit / max_attempts if max_attempts > 0 else time_limit
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            # Scale matters. Entries should be small integers.
            # Try random integers from {-1, 0, 1} or {-2, -1, 0, 1, 2}
            # Or random floats and then optimize.
            
            U = np.random.randint(-2, 3, (nm, target_rank))
            V = np.random.randint(-2, 3, (mp, target_rank))
            W = np.random.randint(-2, 3, (pn, target_rank))
            
            # Check if random integer guess works (unlikely but possible for small rank)
            if verify(U, V, W, T, tol=1e-9):
                return U, V, W
            
            # Use gradient descent to refine
            # Flatten parameters
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_x = x[:nm*target_rank].reshape(nm, target_rank)
                V_x = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_x = x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                try:
                    Temp = np.einsum('ur,vr->uvr', U_x, V_x)
                    T_approx = np.einsum('uvr,wr->uvw', Temp, W_x)
                    err = np.sum((T_approx - T)**2)
                    return err
                except:
                    return 1e10

            # Optimize
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.fun < 1e-4:
                # Try to snap to integers/half-integers
                U_opt = res.x[:nm*target_rank].reshape(nm, target_rank)
                V_opt = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_opt = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                
                # Round to nearest 0.5
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if verify(U_round, V_round, W_round, T, tol=1e-6):
                    # Convert to lists
                    return U_round.tolist(), V_round.tolist(), W_round.tolist()
        
        return None

    # Step 1: Try sub-problem 2x3 * 3x3
    # If we find rank R_sub, we can get rank 2*R_sub for 2x3 * 3x6
    # Lower bound for 2x3 * 3x3 is 9.
    # So try R_sub = 9, 10, 11...
    
    time_limit_total = 100 # Leave 20s for safety
    t_start = time.time()
    
    # Try R_sub = 9
    res_sub = try_find(2, 3, 3, 9, 10)
    if res_sub:
        U_sub, V_sub, W_sub = res_sub
        # Construct 2x3 * 3x6
        R = 9
        U_new = np.hstack([np.array(U_sub), np.array(U_sub)])
        V_new = np.zeros((18, 2*R))
        W_new = np.zeros((12, 2*R))
        
        for j in range(3):
            for k in range(3):
                # First block (k 0..2)
                V_new[j*6 + k, 0:R] = V_sub[j*3 + k, 0:R]
                # Second block (k 3..5)
                V_new[j*6 + (k+3), R:2*R] = V_sub[j*3 + k, 0:R]
                
        for k in range(3):
            for i in range(2):
                # First block (k 0..2)
                W_new[k*2 + i, 0:R] = W_sub[k*2 + i, 0:R]
                # Second block (k 3..5)
                W_new[(k+3)*2 + i, R:2*R] = W_sub[k*2 + i, 0:R]
                
        # Verify
        T_full = make_tensor(2, 3, 6)
        if verify(U_new, V_new, W_new, T_full, tol=1e-6):
            return {
                "U": U_new.tolist(),
                "V": V_new.tolist(),
                "W": W_new.tolist()
            }
            
    # Step 2: If sub-problem failed or time is up, try direct search for 2x3 * 3x6
    # Start from rank 18 (lower bound)
    remaining_time = time_limit_total - (time.time() - t_start)
    if remaining_time > 5:
        for R in range(18, 37):
            if time.time() - t_start > time_limit_total:
                break
            res_direct = try_find(2, 3, 6, R, min(5, remaining_time))
            if res_direct:
                U_d, V_d, W_d = res_direct
                return {
                    "U": U_d,
                    "V": V_d,
                    "W": W_d
                }
            remaining_time = time_limit_total - (time.time() - t_start)
            if remaining_time < 5:
                break

    # Fallback: Schoolbook
    rank = n * m * p
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
