"""Matrix multiplication 2,4,5 — rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
import scipy.optimize as so

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

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
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(R):
                        s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if s != 1:
                        return False
        return True

    def error_func(x, R):
        """
        x is a flat array of size (n*m + m*p + p*n) * R
        First n*m*R elements are U
        Next m*p*R elements are V
        Last p*n*R elements are W
        """
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R:].reshape(p*n, R)
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    residuals.append(val - 1.0)
        return np.array(residuals)

    def try_rank_scipy(R, max_restarts=10, max_iter=200):
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(restart * 1000 + R)
            # Initialize with small random values
            x0 = rng.randn((n*m + m*p + p*n) * R) * 0.1
            
            # Use least_squares
            try:
                res = so.least_squares(error_func, x0, args=(R,), method='lm', max_nfev=5000)
            except:
                # If LM fails, try BFGS or default
                try:
                    res = so.least_squares(error_func, x0, args=(R,), max_nfev=5000)
                except:
                    continue
            
            if res.cost < 1e-15:
                # Extract U, V, W
                U_flat = res.x[:n*m*R]
                V_flat = res.x[n*m*R:n*m*R + m*p*R]
                W_flat = res.x[n*m*R + m*p*R:]
                
                U = U_flat.reshape(n*m, R)
                V = V_flat.reshape(m*p, R)
                W = W_flat.reshape(p*n, R)
                
                # Convert to fractions
                U_frac = [[Fraction(U[i, r]).limit_denominator(10000) for r in range(R)] for i in range(n*m)]
                V_frac = [[Fraction(V[i, r]).limit_denominator(10000) for r in range(R)] for i in range(m*p)]
                W_frac = [[Fraction(W[i, r]).limit_denominator(10000) for r in range(R)] for i in range(p*n)]
                
                if verify_exact(U_frac, V_frac, W_frac, R):
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
            
            # Also check if a simple rounding works if cost is very small
            if res.cost < 1e-12:
                U = res.x[:n*m*R].reshape(n*m, R)
                V = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Try snapping to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                U_frac = [[Fraction(U_snap[i, r]).limit_denominator(2) for r in range(R)] for i in range(n*m)]
                V_frac = [[Fraction(V_snap[i, r]).limit_denominator(2) for r in range(R)] for i in range(m*p)]
                W_frac = [[Fraction(W_snap[i, r]).limit_denominator(2) for r in range(R)] for i in range(p*n)]
                
                if verify_exact(U_frac, V_frac, W_frac, R):
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
            
            if time.time() > deadline:
                return None

    # Try ranks from 21 up to 39
    for R in range(21, 40):
        if time.time() > deadline:
            break
        # Allocate more time/restarts for lower ranks
        restarts = 5 if R < 30 else 2
        result = try_rank_scipy(R, max_restarts=restarts)
        if result is not None:
            return result

    return make_schoolbook()
# EVOLVE-BLOCK-END
