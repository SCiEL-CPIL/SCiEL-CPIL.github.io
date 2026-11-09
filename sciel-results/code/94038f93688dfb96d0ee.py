"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition satisfies T[i,j,k] = 1 for all valid (i,j,k)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _try_rank(r, n, m, p, deadline, seed=42):
    """Try to find a rank-r decomposition using random restarts and local optimization."""
    rng = np.random.RandomState(seed)
    best_err = np.inf
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization with small values
        U = rng.randn(n*m, r) * 0.5
        V = rng.randn(m*p, r) * 0.5
        W = rng.randn(p*n, r) * 0.5
        
        # Use alternating least squares to refine
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Update U: for each row i*m+j, solve for U[i*m+j,:] given V,W
            # The constraint is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is a linear system in U[i*m+j,:]
            for i in range(n):
                for j in range(m):
                    # Build the system: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                    A_mat = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            A_mat[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Solve A_mat @ U[i*m+j,:] = ones(p)
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                        U[i*m+j, :] = sol
                    except:
                        pass
            
            # Update V: for each row j*p+k, solve for V[j*p+k,:] given U,W
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            A_mat[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                        V[j*p+k, :] = sol
                    except:
                        pass
            
            # Update W: for each row k*n+i, solve for W[k*n+i,:] given U,V
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A_mat[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                        W[k*n+i, :] = sol
                    except:
                        pass
        
        # Check the error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0)**2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-8:
        # Round to nearest half-integer
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    deadline = time.time() + 145
    
    # Try ranks from 12 up to 23
    for target_rank in range(12, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank(target_rank, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
