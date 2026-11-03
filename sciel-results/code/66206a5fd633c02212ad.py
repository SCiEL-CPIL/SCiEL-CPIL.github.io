import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Construct the target tensor T of shape (n*m, m*p, p*n)
    # T[i, j, k] = 1 if the indices correspond to a valid matrix multiplication term
    # Indices:
    # U index: i * m + j  (A_ij)
    # V index: j * p + k  (B_jk)
    # W index: k * n + i  (C_ki? No, C_ik flattened as k*n+i)
    # The term A_ij * B_jk contributes to C_ik.
    # So T[(i*m+j), (j*p+k), (k*n+i)] = 1.
    # All other entries are 0.
    
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Helper to verify a solution exactly
    def verify(U, V, W, r):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Reconstruct tensor
        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
        if not np.allclose(T_rec, T, atol=1e-9):
            return False
        return True

    # Helper to check if values are close to half-integers
    def is_half_integer(arr, tol=1e-6):
        doubled = 2 * arr
        rounded = np.round(doubled)
        if np.allclose(doubled, rounded, atol=tol):
            return rounded / 2.0
        return None

    # Try different ranks
    # Known rank for 3x3x4 is likely 18 or 19. Schoolbook is 36.
    # We try from 15 to 30.
    for r in range(15, 31):
        # Try a few random initializations
        for attempt in range(5):
            # Initialize with small random floats
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Flatten parameters for optimization
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def loss(params):
                u = params[:n*m*r].reshape(n*m, r)
                v = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                w = params[n*m*r + m*p*r :].reshape(p*n, r)
                T_rec = np.einsum('ir,jr,kr->ijk', u, v, w)
                return np.sum((T_rec - T)**2)
            
            # Use BFGS or L-BFGS-B
            try:
                res = minimize(loss, params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
                if res.fun < 1e-8:
                    # Extract solution
                    u_sol = res.x[:n*m*r].reshape(n*m, r)
                    v_sol = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    w_sol = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                    
                    # Check for half-integer rounding
                    u_rounded = is_half_integer(u_sol)
                    v_rounded = is_half_integer(v_sol)
                    w_rounded = is_half_integer(w_sol)
                    
                    if u_rounded is not None and v_rounded is not None and w_rounded is not None:
                        if verify(u_rounded, v_rounded, w_rounded, r):
                            return {"U": u_rounded.tolist(), "V": v_rounded.tolist(), "W": w_rounded.tolist()}
            except Exception:
                continue

    # Fallback: Schoolbook algorithm
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

# EVOLVE-BLOCK-END
