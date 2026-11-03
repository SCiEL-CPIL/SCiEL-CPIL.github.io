import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
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
    
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Precompute target tensor as a 3D array for fast access
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    def compute_residual(U, V, W):
        """Compute Frobenius residual using matrix operations for speed."""
        residual = 0.0
        for alpha in range(n * m):
            for beta in range(m * p):
                # Vectorized over gamma
                tmp = U[alpha] * V[beta]  # shape (R,)
                vals = W @ tmp  # shape (p*n,)
                diff = vals - T[alpha, beta, :]
                residual += np.dot(diff, diff)
        return residual

    def exact_verify(U, V, W):
        """Verify exact decomposition with rational arithmetic."""
        # Convert to fractions for exactness
        U_frac = [[Fraction(U[i, j]).limit_denominator(1000) for j in range(U.shape[1])] for i in range(U.shape[0])]
        V_frac = [[Fraction(V[i, j]).limit_denominator(1000) for j in range(V.shape[1])] for i in range(V.shape[0])]
        W_frac = [[Fraction(W[i, j]).limit_denominator(1000) for j in range(W.shape[1])] for i in range(W.shape[0])]
        
        for alpha in range(n * m):
            for beta in range(m * p):
                for gamma in range(p * n):
                    t_val = T[alpha, beta, gamma]
                    val = sum(U_frac[alpha][r] * V_frac[beta][r] * W_frac[gamma][r] for r in range(len(U_frac[alpha])))
                    if val != Fraction(int(t_val)):
                        return False
        return True

    def snap_to_halfint(U, V, W):
        """Snap values to nearest half-integer and verify."""
        U_p = np.round(U * 2) / 2.0
        V_p = np.round(V * 2) / 2.0
        W_p = np.round(W * 2) / 2.0
        
        # Quick float check first
        for alpha in range(n * m):
            for beta in range(m * p):
                tmp = U_p[alpha] * V_p[beta]
                vals = W_p @ tmp
                if np.max(np.abs(vals - T[alpha, beta, :])) > 1e-9:
                    return None
        return U_p, V_p, W_p

    def als_solve_fast(R, max_iters=50, tol=1e-10, seed=None):
        """Fast ALS with precomputed structures."""
        rng = np.random.RandomState(seed)
        
        # Structured initialization: start from schoolbook-like patterns
        U = rng.randn(n * m, R) * 0.1
        V = rng.randn(m * p, R) * 0.1
        W = rng.randn(p * n, R) * 0.1
        
        # Add some structure: initialize with partial schoolbook
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = (i * m + j) * p + k
                    if r < R:
                        U[i * m + j, r] += 1.0
                        V[j * p + k, r] += 1.0
                        W[k * n + i, r] += 1.0
        
        prev_res = float('inf')
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U rows
            for alpha in range(n * m):
                # Build design matrix: rows are (beta, gamma) pairs
                # A[beta * p*n + gamma, r] = V[beta, r] * W[gamma, r]
                A = np.empty((m * p * p * n, R))
                b = np.empty(m * p * p * n)
                idx = 0
                for beta in range(m * p):
                    for gamma in range(p * n):
                        A[idx, :] = V[beta] * W[gamma]
                        b[idx] = T[alpha, beta, gamma]
                        idx += 1
                U[alpha], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Update V rows
            for beta in range(m * p):
                A = np.empty((n * m * p * n, R))
                b = np.empty(n * m * p * n)
                idx = 0
                for alpha in range(n * m):
                    for gamma in range(p * n):
                        A[idx, :] = U[alpha] * W[gamma]
                        b[idx] = T[alpha, beta, gamma]
                        idx += 1
                V[beta], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Update W rows
            for gamma in range(p * n):
                A = np.empty((n * m * m * p, R))
                b = np.empty(n * m * m * p)
                idx = 0
                for alpha in range(n * m):
                    for beta in range(m * p):
                        A[idx, :] = U[alpha] * V[beta]
                        b[idx] = T[alpha, beta, gamma]
                        idx += 1
                W[gamma], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Check convergence
            res = compute_residual(U, V, W)
            if abs(prev_res - res) < tol:
                break
            prev_res = res
            
        return U, V, W, prev_res

    def polish_high_precision(U, V, W, max_iters=20):
        """High-precision polish: alternate between float solve and snap."""
        U_p, V_p, W_p = U.copy(), V.copy(), W.copy()
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Snap to half-integers
            U_s = np.round(U_p * 2) / 2.0
            V_s = np.round(V_p * 2) / 2.0
            W_s = np.round(W_p * 2) / 2.0
            
            # Check if this is already exact
            if exact_verify(U_s, V_s, W_s):
                return U_s, V_s, W_s
            
            # If not exact, do one more ALS iteration from snapped values
            R = U_p.shape[1]
            # Update U
            for alpha in range(n * m):
                A = np.empty((m * p * p * n, R))
                b = np.empty(m * p * p * n)
                idx = 0
                for beta in range(m * p):
                    for gamma in range(p * n):
                        A[idx, :] = V_s[beta] * W_s[gamma]
                        b[idx] = T[alpha, beta, gamma]
                        idx += 1
                U_p[alpha], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Update V
            for beta in range(m * p):
                A = np.empty((n * m * p * n, R))
                b = np.empty(n * m * p * n)
                idx = 0
                for alpha in range(n * m):
                    for gamma in range(p * n):
                        A[idx, :] = U_p[alpha] * W_s[gamma]
                        b[idx] = T[alpha, beta, gamma]
                        idx += 1
                V_p[beta], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Update W
            for gamma in range(p * n):
                A = np.empty((n * m * m * p, R))
                b = np.empty(n * m * m * p)
                idx = 0
                for alpha in range(n * m):
                    for beta in range(m * p):
                        A[idx, :] = U_p[alpha] * V_p[beta]
                        b[idx] = T[alpha, beta, gamma]
                        idx += 1
                W_p[gamma], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Final snap and check
        U_f = np.round(U_p * 2) / 2.0
        V_f = np.round(V_p * 2) / 2.0
        W_f = np.round(W_p * 2) / 2.0
        if exact_verify(U_f, V_f, W_f):
            return U_f, V_f, W_f
        return None

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try ranks from lower bound to schoolbook
    lower_bound = max(n * m, m * p, p * n)  # = 25
    for R in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 15
        
        while not found and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            seed = attempts * 1000 + R
            U, V, W, res = als_solve_fast(R, max_iters=30, tol=1e-10, seed=seed)
            
            # Try to polish
            if res < 1e-3:
                polished = polish_high_precision(U, V, W, max_iters=10)
                if polished is not None:
                    U_p, V_p, W_p = polished
                    if exact_verify(U_p, V_p, W_p):
                        found = True
                        best_rank = R
                        best_U = U_p.tolist()
                        best_V = V_p.tolist()
                        best_W = W_p.tolist()
                        break
            elif res < 1e-2:
                # Try direct snap
                snapped = snap_to_halfint(U, V, W)
                if snapped is not None:
                    U_p, V_p, W_p = snapped
                    if exact_verify(U_p, V_p, W_p):
                        found = True
                        best_rank = R
                        best_U = U_p.tolist()
                        best_V = V_p.tolist()
                        best_W = W_p.tolist()
                        break
        
        if found:
            break

    # Final safety check
    if time.time() > deadline - 5:
        # Return schoolbook if we're running out of time
        pass

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
