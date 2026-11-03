"""Program for matrix multiplication 2x3 by 3x5 using tensor code / Berlekamp-Massey approach."""
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 6
    V_dim = m * p  # 15
    W_dim = p * n  # 10

    # Schoolbook rank
    rank_school = n * m * p
    
    # Try to find a lower rank decomposition using alternating minimization
    # We test ranks from rank_school - 1 down to a lower bound
    # Lower bound estimate: maybe rank_school - 5
    
    def solve_for_U(V, W, R):
        # Equations: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # For fixed V, W, this is linear in U.
        # Unknowns: U is (n*m) x R.
        # We have n*m*p equations.
        A = np.zeros((n * m * p, n * m * R))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(R):
                        # Coefficient for U[i*m+j, c]
                        val = V[j * p + k, c] * W[k * n + i, c]
                        A[eq_idx, (i * m + j) * R + c] = val
                    eq_idx += 1
        
        # Solve A * U_flat = b
        # U_flat is (n*m*R)
        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        U = sol.reshape((n * m, R))
        return U

    def solve_for_V(U, W, R):
        A = np.zeros((n * m * p, m * p * R))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(R):
                        val = U[i * m + j, c] * W[k * n + i, c]
                        A[eq_idx, (j * p + k) * R + c] = val
                    eq_idx += 1
        
        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        V = sol.reshape((m * p, R))
        return V

    def solve_for_W(U, V, R):
        A = np.zeros((n * m * p, p * n * R))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(R):
                        val = U[i * m + j, c] * V[j * p + k, c]
                        A[eq_idx, (k * n + i) * R + c] = val
                    eq_idx += 1
        
        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        W = sol.reshape((p * n, R))
        return W

    def check_error(U, V, W, R):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(R):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err += (s - 1.0) ** 2
        return np.sqrt(err)

    # Try ranks from rank_school - 1 down to max(1, rank_school - 10)
    # If we find one, we try to snap to half-integers
    for R in range(rank_school - 1, max(1, rank_school - 15), -1):
        # Initialize with small random values
        np.random.seed(42) # Reproducibility
        U = np.random.randn(U_dim, R) * 0.1
        V = np.random.randn(V_dim, R) * 0.1
        W = np.random.randn(W_dim, R) * 0.1
        
        converged = False
        for it in range(100):
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
            
            err = check_error(U, V, W, R)
            if err < 1e-8:
                converged = True
                break
        
        if converged:
            # Try to snap to half-integers
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            err_snapped = check_error(U_snapped, V_snapped, W_snapped, R)
            if err_snapped < 1e-8:
                return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
            else:
                # If snapping fails, return the float solution? 
                # The prompt asks for integer/half-integer. 
                # If the float solution is exact but not half-integer, it might be invalid.
                # However, if the true solution is half-integer, the float solution should be close.
                # If it's not close, maybe the rank R is not achievable with half-integers?
                # Or maybe the convergence was to a different local minimum.
                # Let's try a few random seeds if this fails.
                pass

    # Fallback to schoolbook
    rank = rank_school
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
