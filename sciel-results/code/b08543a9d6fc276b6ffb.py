import numpy as np
from scipy.optimize import least_squares
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 3]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if the decomposition is exact with integer/half-integer entries.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    R = U.shape[1]
    # Convert to Fractions for exact arithmetic
    U_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in U]
    V_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in V]
    W_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[i * m + j][r] * V_frac[j * p + k][r] * W_frac[k * n + i][r]
                if total != 1:
                    return False
    return True

def _snap_to_half_integer(arr):
    return np.round(arr * 2) / 2.0

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Try ranks from 12 up to 17
    for R in range(12, 18):
        if time.time() > deadline:
            break
            
        # Number of variables: n*m*R + m*p*R + p*n*R
        num_vars = (n * m + m * p + p * n) * R
        
        def residual(x):
            # Unpack
            U = x[:n*m*R].reshape(n*m, R)
            V = x[n*m*R:(n*m + m*p)*R].reshape(m*p, R)
            W = x[(n*m + m*p)*R:].reshape(p*n, R)
            
            # Calculate residuals for all i, j, k
            res = np.zeros(n * m * p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Sum over r
                        # U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                        val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                        res[idx] = val - 1.0
                        idx += 1
            return res

        found = False
        # Try a few random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Use small integers to help convergence?
            # Let's try uniform [-1, 1]
            x0 = np.random.uniform(-1, 1, num_vars)
            
            try:
                sol = least_squares(residual, x0, method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if sol.cost < 1e-10:
                # Extract U, V, W
                U = sol.x[:n*m*R].reshape(n*m, R)
                V = sol.x[n*m*R:(n*m + m*p)*R].reshape(m*p, R)
                W = sol.x[(n*m + m*p)*R:].reshape(p*n, R)
                
                # Snap to half-integers
                U_snapped = _snap_to_half_integer(U)
                V_snapped = _snap_to_half_integer(V)
                W_snapped = _snap_to_half_integer(W)
                
                # Verify
                if _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p):
                    return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
        
    # Fallback to schoolbook
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
