import numpy as np
import time
from fractions import Fraction
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def _verify(U, V, W):
        """
        Verifies if U, V, W exactly decompose the tensor with half-integer entries.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        Returns True if valid, False otherwise.
        """
        R = U.shape[1]
        
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Check if snap was close
        if not (np.allclose(U, U_snap, atol=1e-6) and 
                np.allclose(V, V_snap, atol=1e-6) and 
                np.allclose(W, W_snap, atol=1e-6)):
            return False
            
        # Exact verification using Fractions
        U_frac = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in U_snap]
        V_frac = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in V_snap]
        W_frac = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in W_snap]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = Fraction(0)
                    for r in range(R):
                        s += U_frac[u_idx][r] * V_frac[v_idx][r] * W_frac[w_idx][r]
                    if s != 1:
                        return False
        return True

    def _schoolbook():
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

    def _residual_func(params, n, m, p, R):
        """Compute residual for given parameters."""
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    res[idx] = np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :]) - 1.0
                    idx += 1
        return res

    def _try_rank(R, max_restarts=20, time_per_restart=5.0):
        """
        Attempts to find a decomposition of rank R.
        """
        num_vars = R * (n*m + m*p + p*n)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
                
            np.random.seed(restart * 7919 + R * 104729)
            
            # Multiple initialization strategies
            if restart % 3 == 0:
                # Strategy 1: Small random
                U = np.random.randn(n*m, R) * 0.3
                V = np.random.randn(m*p, R) * 0.3
                W = np.random.randn(p*n, R) * 0.3
            elif restart % 3 == 1:
                # Strategy 2: Medium random with some structure
                U = np.random.randn(n*m, R) * 0.5
                V = np.random.randn(m*p, R) * 0.5
                W = np.random.randn(p*n, R) * 0.5
            else:
                # Strategy 3: Sparse initialization
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                for r in range(R):
                    # Random non-zero entries
                    u_rows = np.random.choice(n*m, size=min(3, n*m), replace=False)
                    v_rows = np.random.choice(m*p, size=min(3, m*p), replace=False)
                    w_rows = np.random.choice(p*n, size=min(3, p*n), replace=False)
                    for idx in u_rows:
                        U[idx, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    for idx in v_rows:
                        V[idx, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    for idx in w_rows:
                        W[idx, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
            
            # Gauge fixing: set first row of U to 1s to remove scaling ambiguity
            U[0, :] = 1.0
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use time-limited optimization
            restart_start = time.time()
            
            try:
                sol = least_squares(
                    _residual_func, params, 
                    args=(n, m, p, R),
                    method='lm', 
                    max_nfev=5000, 
                    ftol=1e-12, xtol=1e-12, gtol=1e-12
                )
            except Exception:
                continue
                
            if time.time() - restart_start > time_per_restart:
                continue
                
            if np.linalg.norm(sol.fun) < 1e-8:
                U_sol = sol.x[0:n*m*R].reshape(n*m, R)
                V_sol = sol.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_sol = sol.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Try to snap to half-integers
                if _verify(U_sol, V_sol, W_sol):
                    return U_sol, V_sol, W_sol
                
                # Try a polish step with tighter tolerance
                try:
                    sol2 = least_squares(
                        _residual_func, sol.x, 
                        args=(n, m, p, R),
                        method='lm', 
                        max_nfev=10000, 
                        ftol=1e-14, xtol=1e-14, gtol=1e-14
                    )
                    if np.linalg.norm(sol2.fun) < 1e-10:
                        U_sol2 = sol2.x[0:n*m*R].reshape(n*m, R)
                        V_sol2 = sol2.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                        W_sol2 = sol2.x[n*m*R + m*p*R:].reshape(p*n, R)
                        if _verify(U_sol2, V_sol2, W_sol2):
                            return U_sol2, V_sol2, W_sol2
                except Exception:
                    pass
            
        return None

    # Try ranks from 50 down to 45 (reasonable lower bound for this problem)
    # Based on known results, rank should be at least around 45-50 for 2x3x9
    for R in range(50, 44, -1):
        if time.time() > deadline:
            break
            
        result = _try_rank(R)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return _schoolbook()
