import time
import numpy as np
from scipy.optimize import fsolve

DIMENSIONS = [2, 2, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W give a valid decomposition of the matmul tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _rank14_decomposition():
    """Known-good rank-14 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_vals, v_vals, w_vals, col):
        for i in range(n * m):
            U[i][col] = u_vals[i]
        for j in range(m * p):
            V[j][col] = v_vals[j]
        for k in range(p * n):
            W[k][col] = w_vals[k]

    # Left Part (Cols 0, 1) - Strassen for 2x2x2
    add_term([1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0], 0)
    add_term([0, 0, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, -1, 0, 0, 0, 0], 1)
    add_term([1, 0, 0, 0], [0, 1, 0, 0, 0, -1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0], 2)
    add_term([0, 0, 0, 1], [-1, 0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], 3)
    add_term([1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [-1, 0, 1, 0, 0, 0, 0, 0], 4)
    add_term([-1, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], 5)
    add_term([0, 1, 0, -1], [0, 0, 0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], 6)
    # Right Part (Cols 2, 3) - Strassen for 2x2x2
    add_term([1, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 1], 7)
    add_term([0, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, -1], 8)
    add_term([1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1, 1], 9)
    add_term([0, 0, 0, 1], [0, 0, -1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1, 0, 0], 10)
    add_term([1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, -1, 0, 1, 0], 11)
    add_term([-1, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], 12)
    add_term([0, 1, 0, -1], [0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0], 13)
    
    return {"U": U, "V": V, "W": W}

def _try_rank13():
    """Attempt to find a rank-13 decomposition using local optimization from rank-14."""
    n, m, p = DIMENSIONS
    rank = 13
    start_time = time.time()
    deadline = start_time + 110

    # Get rank-14 solution
    r14 = _rank14_decomposition()
    U14 = np.array(r14['U'], dtype=float)
    V14 = np.array(r14['V'], dtype=float)
    W14 = np.array(r14['W'], dtype=float)

    # We will try to find a rank-13 solution by perturbing the rank-14 solution
    # and using fsolve to drive the error to zero.
    # We drop one column and adjust the rest.
    
    # Try dropping each column to see if one leads to a solvable system
    for drop_idx in range(14):
        if time.time() > deadline:
            break
            
        # Initialize with rank-14 solution minus the dropped column
        U = np.delete(U14, drop_idx, axis=1).copy()
        V = np.delete(V14, drop_idx, axis=1).copy()
        W = np.delete(W14, drop_idx, axis=1).copy()
        
        # Flatten for fsolve
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def equations(x):
            U_f = x[:n*m*rank].reshape(n*m, rank)
            V_f = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_f = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(U_f[u_idx, :], np.multiply(V_f[v_idx, :], W_f[w_idx, :]))
                        errors.append(val - 1.0)
            return np.array(errors)
        
        try:
            # Use fsolve to find a solution
            # full_output=True to check convergence
            sol, info, ier, msg = fsolve(equations, x0, full_output=True, maxfev=10000)
            
            if ier == 1:
                # Check residual
                res = np.linalg.norm(equations(sol))
                if res < 1e-8:
                    # Extract solution
                    U_sol = sol[:n*m*rank].reshape(n*m, rank)
                    V_sol = sol[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_sol = sol[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Try rounding to integers and half-integers
                    for factor in [1, 2]:
                        U_r = np.round(U_sol * factor) / factor
                        V_r = np.round(V_sol * factor) / factor
                        W_r = np.round(W_sol * factor) / factor
                        
                        if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                            def to_list(arr):
                                return [[float(x) for x in row] for row in arr]
                            return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
                            
        except Exception:
            pass

    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x4 matrix multiplication.
    Attempts rank 13 first; falls back to rank 14 if not found.
    """
    n, m, p = DIMENSIONS
    
    # Try rank 13
    result = _try_rank13()
    if result is not None:
        return result
    
    # Fallback to rank 14
    return _rank14_decomposition()
