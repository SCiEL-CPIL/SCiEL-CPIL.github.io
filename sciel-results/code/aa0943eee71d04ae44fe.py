"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s kill
    
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_exact(U_list, V_list, W_list, R):
        """Check if decomposition is exact: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if s != 1:
                        return False
        return True

    def verify_float(U_np, V_np, W_np, R, tol=1e-9):
        """Quick float check"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_rank(R, max_iters=100, n_restarts=5, seed_base=0):
        """Try to find exact decomposition of rank R using ALS-like refinement"""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(seed_base + restart)
            U = rng.randn(n*m, R)
            V = rng.randn(m*p, R)
            W = rng.randn(p*n, R)
            
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, R))
                        for i in range(n):
                            N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        P = np.zeros((m, R))
                        for j in range(m):
                            P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err += (s - 1.0)**2
                if err < 1e-10:
                    # Check if close to integers or half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    if np.allclose(U, U_rounded, atol=1e-5) and \
                       np.allclose(V, V_rounded, atol=1e-5) and \
                       np.allclose(W, W_rounded, atol=1e-5):
                        # Convert to lists of floats/ints for exact check
                        U_list = [[float(x) for x in row] for row in U_rounded]
                        V_list = [[float(x) for x in row] for row in V_rounded]
                        W_list = [[float(x) for x in row] for row in W_rounded]
                        # Exact check with Fraction to avoid float issues if possible, 
                        # but inputs are half-integers so they are exact in float64.
                        if verify_exact(U_list, V_list, W_list, R):
                            return {"U": U_list, "V": V_list, "W": W_list}
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
        return None

    # Try ranks from 35 to 39
    # Theoretical lower bound is higher, but let's start reasonably.
    for R in range(35, n*m*p):
        if time.time() > deadline:
            break
        result = try_rank(R, max_iters=50, n_restarts=3, seed_base=R)
        if result is not None:
            return result

    # Fallback to Schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
