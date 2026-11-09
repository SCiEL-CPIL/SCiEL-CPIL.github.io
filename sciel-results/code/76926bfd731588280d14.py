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

    def objective(params, target_rank):
        """Objective function for optimizer: sum of squared residuals."""
        # Unpack params into U, V, W
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n, m, target_rank)
        V = V_flat.reshape(m, p, target_rank)
        W = W_flat.reshape(p, n, target_rank)
        
        # Compute residuals
        # C[i, k, j1, j2] = sum_r W[k, i, r] * U[i, j1, r] * V[j2, k, r]
        C = np.einsum('kir,ijr,jkr->ikij', W, U, V, optimize=True)
        
        I_m = np.eye(m)
        residuals = np.zeros((n, p, m, m))
        for i in range(n):
            for k in range(p):
                residuals[i, k] = C[i, k] - I_m
                
        return np.sum(residuals**2)

    random.seed(42)
    np.random.seed(42)

    # Try ranks from 62 down to 50
    for target_rank in range(r0 - 1, 49, -1):
        if time_left() < 10:
            break
        if target_rank >= best_rank:
            continue

        found = False
        # Try a few random restarts
        for attempt in range(20):
            if time_left() < 5:
                break
            
            # Initialize parameters randomly
            size_U = n * m * target_rank
            size_V = m * p * target_rank
            size_W = p * n * target_rank
            total_size = size_U + size_V + size_W
            
            # Small random initialization
            params0 = np.random.randn(total_size) * 0.1
            
            # Use Nelder-Mead or Powell for derivative-free optimization
            # Powell is often more robust for this kind of problem
            try:
                result = minimize(
                    objective, 
                    params0, 
                    args=(target_rank,), 
                    method='Powell', 
                    options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                
                if result.fun < 1e-6:
                    # Reconstruct U, V, W
                    U_flat = result.x[:size_U]
                    V_flat = result.x[size_U:size_U + size_V]
                    W_flat = result.x[size_U + size_V:]
                    
                    U_r = U_flat.reshape(n * m, target_rank)
                    V_r = V_flat.reshape(m * p, target_rank)
                    W_r = W_flat.reshape(p * n, target_rank)
                    
                    # Final verification
                    if verify_matmul_fast(U_r, V_r, W_r):
                        best_rank = target_rank
                        best_solution = (U_r.copy(), V_r.copy(), W_r.copy())
                        found = True
                        break
                        
            except Exception:
                continue

        if found:
            break

    # ---- Return the best solution found ----
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
