"""Strassen-like recursive partitioning with coefficient local search."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - t0)

    # ---- schoolbook (rank 63) ----
    r0 = n * m * p  # 63
    U0 = np.zeros((n * m, r0))
    V0 = np.zeros((m * p, r0))
    W0 = np.zeros((p * n, r0))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, c] = 1.0
                V0[j * p + k, c] = 1.0
                W0[k * n + i, c] = 1.0
                c += 1

    def verify_matmul_fast(U, V, W, tol=1e-8):
        """Fast vectorized verification."""
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        # C[i, k, j1, j2] = sum_r W[k, i, r] * U[i, j1, r] * V[j2, k, r]
        try:
            C = np.einsum('kir,ijr,jkr->ikij', W_r, U_r, V_r, optimize=True)
        except Exception:
            return False
        I_m = np.eye(m)
        for i in range(n):
            for k in range(p):
                if not np.allclose(C[i, k, :, :], I_m, atol=tol):
                    return False
        return True

    if not verify_matmul_fast(U0, V0, W0):
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    best_rank = r0
    best_solution = (U0.copy(), V0.copy(), W0.copy())

    # Strategy: Split p=7 into 3+3+1.
    # Blocks:
    # 1. k in {0,1,2} -> M_3,3,3 (Rank 27 schoolbook)
    # 2. k in {3,4,5} -> M_3,3,3 (Rank 27 schoolbook)
    # 3. k in {6}     -> M_3,3,1 (Rank 9 schoolbook, target 7)
    
    # We will try to find a rank-7 decomposition for the M_3,3,1 part.
    # If successful, total rank = 27 + 27 + 7 = 61.

    def find_m331_rank7(max_attempts=200, time_limit=10.0):
        """
        Find a rank-7 decomposition for 3x3 matrix * 3x1 vector.
        Returns (U_local, V_local, W_local) or None.
        U_local: (9, 7)
        V_local: (3, 7)
        W_local: (3, 7)
        """
        t_start = time.time()
        np.random.seed(42)
        
        # Constraints:
        # For each i in 0..2:
        #   sum_r W[i, r] * U[i*3 + j1, r] * V[j2, r] = delta_{j1, j2}
        
        def objective(params):
            # Unpack params
            # U: 9*7, V: 3*7, W: 3*7
            U_flat = params[0:63]
            V_flat = params[63:84]
            W_flat = params[84:105]
            
            U = U_flat.reshape(9, 7)
            V = V_flat.reshape(3, 7)
            W = W_flat.reshape(3, 7)
            
            err = 0.0
            for i in range(3):
                for j1 in range(3):
                    for j2 in range(3):
                        target = 1.0 if j1 == j2 else 0.0
                        # sum_r W[i, r] * U[i*3 + j1, r] * V[j2, r]
                        val = np.dot(W[i], U[i*3 + j1] * V[j2])
                        err += (val - target) ** 2
            return err

        for attempt in range(max_attempts):
            if time.time() - t_start > time_limit:
                break
            
            # Random initialization
            # Scale matters. Try different scales.
            scale = 1.0 + np.random.rand()
            U_init = np.random.randn(9, 7) * scale
            V_init = np.random.randn(3, 7) * scale
            W_init = np.random.randn(3, 7) * scale
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                
                if res.fun < 1e-6:
                    U_sol = res.x[0:63].reshape(9, 7)
                    V_sol = res.x[63:84].reshape(3, 7)
                    W_sol = res.x[84:105].reshape(3, 7)
                    return (U_sol, V_sol, W_sol)
            except Exception:
                continue
        
        return None

    # Try to find the rank-7 solution for the last block
    m331_solution = find_m331_rank7(max_attempts=100, time_limit=10.0)
    
    if m331_solution is not None and time_left() > 5.0:
        U_sol, V_sol, W_sol = m331_solution
        
        # Construct the combined rank-61 solution
        new_rank = 61
        U_new = np.zeros((n * m, new_rank))
        V_new = np.zeros((m * p, new_rank))
        W_new = np.zeros((p * n, new_rank))
        
        # Copy Schoolbook for Block 1 (k=0,1,2) and Block 2 (k=3,4,5)
        # These occupy the first 54 columns (27 + 27)
        for c in range(54):
            # Map c to (i, j, k) in schoolbook ordering
            i = c // (m * p)
            rem = c % (m * p)
            j = rem // p
            k = rem % p
            
            U_new[i * m + j, c] = 1.0
            V_new[j * p + k, c] = 1.0
            W_new[k * n + i, c] = 1.0
            
        # Fill Block 3 (k=6) with the rank-7 solution
        # Columns 54 to 60
        start_col = 54
        for r in range(7):
            # U: rows i*3+j
            U_new[:, start_col + r] = U_sol[:, r]
            
            # V: rows j*7+6 (since k=6)
            for j in range(3):
                V_new[j * p + 6, start_col + r] = V_sol[j, r]
                
            # W: rows 6*3+i (since k=6)
            for i in range(3):
                W_new[6 * n + i, start_col + r] = W_sol[i, r]
                
        # Verify the new solution
        if verify_matmul_fast(U_new, V_new, W_new):
            best_rank = new_rank
            best_solution = (U_new.copy(), V_new.copy(), W_new.copy())

    # Return the best solution found
    U_best, V_best, W_best = best_solution

    def to_list(arr):
        result = []
        for row in arr:
            new_row = []
            for x in row:
                hx = round(x * 2) / 2.0
                if abs(x - hx) < 1e-9:
                    new_row.append(hx)
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result

    return {"U": to_list(U_best), "V": to_list(V_best), "W": to_list(W_best)}

# EVOLVE-BLOCK-END
