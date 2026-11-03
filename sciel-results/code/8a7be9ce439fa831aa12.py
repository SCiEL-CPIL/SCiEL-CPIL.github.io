import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_time = 140  # Safety margin
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find lower rank solutions
    # The tensor T_ijk = 1 for all i,j,k
    # We want sum_c U_ijk_c V_jkc_c W_kic_c = 1
    # Indices:
    # U: (n*m) x R, index (i, j) -> i*m + j
    # V: (m*p) x R, index (j, k) -> j*p + k
    # W: (p*n) x R, index (k, i) -> k*n + i
    
    # Precompute indices for the equations
    # Equation for each (i, j, k)
    # i in [0, n), j in [0, m), k in [0, p)
    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                eq_indices.append((u_idx, v_idx, w_idx))
                
    num_eqs = len(eq_indices)
    
    def residuals(params, R):
        # params is a flat vector of length R * (nm + mp + pn)
        nm = n * m
        mp = m * p
        pn = p * n
        U_flat = params[:R * nm].reshape(R, nm).T # (nm, R)
        V_flat = params[R * nm : R * nm + R * mp].reshape(R, mp).T # (mp, R)
        W_flat = params[R * nm + R * mp :].reshape(R, pn).T # (pn, R)
        
        res = np.zeros(num_eqs)
        for idx, (u_idx, v_idx, w_idx) in enumerate(eq_indices):
            # sum over c of U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
            val = np.dot(U_flat[u_idx, :], np.multiply(V_flat[v_idx, :], W_flat[w_idx, :]))
            res[idx] = val - 1.0
        return res

    # Try ranks from lower bound up to schoolbook - 1
    # Lower bound is max(n*m, m*p, p*n) = 18
    # We'll try a few promising ranks.
    ranks_to_try = [18, 20, 22, 24, 26, 28, 30, 32, 34, 35]
    
    start_time = time.time()
    
    for R in ranks_to_try:
        if time.time() - start_time > target_time - 20:
            break
            
        # Random restarts for this rank
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() - start_time > target_time - 10:
                break
                
            # Initialize parameters
            # Random initialization
            np.random.seed(42 + restart)
            params0 = np.random.randn(R * (n*m + m*p + p*n))
            
            try:
                # Use least squares
                result = least_squares(residuals, params0, args=(R,), method='lm', max_nfev=2000)
                if result.success or result.cost < 1e-8:
                    # Check if the solution is valid
                    if result.cost < 1e-8:
                        # Reconstruct matrices
                        nm = n * m
                        mp = m * p
                        pn = p * n
                        U_sol = result.x[:R * nm].reshape(R, nm).T
                        V_sol = result.x[R * nm : R * nm + R * mp].reshape(R, mp).T
                        W_sol = result.x[R * nm + R * mp :].reshape(R, pn).T
                        
                        # The prompt requires integer/half-integer entries.
                        # Numerical solutions are floats. We need to check if they are close to half-integers.
                        # If they are, we can round them.
                        U_rounded = np.round(U_sol * 2) / 2
                        V_rounded = np.round(V_sol * 2) / 2
                        W_rounded = np.round(W_sol * 2) / 2
                        
                        # Verify the rounded solution
                        valid = True
                        for idx, (u_idx, v_idx, w_idx) in enumerate(eq_indices):
                            val = np.dot(U_rounded[u_idx, :], np.multiply(V_rounded[v_idx, :], W_rounded[w_idx, :]))
                            if not np.isclose(val, 1.0, atol=1e-6):
                                valid = False
                                break
                            
                        if valid:
                            best_rank = R
                            best_U = U_rounded
                            best_V = V_rounded
                            best_W = W_rounded
                            # If we found a solution, we can stop early if we hit the theoretical lower bound
                            if R == max(n*m, m*p, p*n):
                                break
            except Exception:
                pass
        
        if R == max(n*m, m*p, p*n):
            break

    # Convert to lists of lists of numbers
    # The interface expects {'U': U, 'V': V, 'W': W}
    # The seed program used lists of lists.
    # We can return numpy arrays if the evaluator accepts them, but the seed used lists.
    # Let's convert to lists to be safe, or just return the arrays if they are compatible.
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # Returning numpy arrays is usually fine for such evaluators.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
