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

def _solve_exact_or_lstsq(A, b, r):
    """Solve A @ x = b. If rank(A) >= len(b), try to find exact solution.
    Otherwise use lstsq. Returns solution or None if no solution exists."""
    try:
        # Check if system is consistent
        rank_A = np.linalg.matrix_rank(A)
        if rank_A < len(b):
            # Underdetermined or inconsistent - use lstsq
            sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            # Verify it's close to a solution
            if np.linalg.norm(A @ sol - b) < 1e-8:
                return sol
            return None
        else:
            # Try to find exact solution
            sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            if np.linalg.norm(A @ sol - b) < 1e-10:
                return sol
            return None
    except:
        return None

def _compute_error(U, V, W, n, m, p, r):
    """Compute the squared error of the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err += (s - 1.0)**2
    return err

def _als_iteration(U, V, W, n, m, p, r):
    """Perform one ALS iteration. Returns (U_new, V_new, W_new, success)."""
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    # Update U
    for i in range(n):
        for j in range(m):
            A_mat = np.zeros((p, r))
            for k in range(p):
                for c in range(r):
                    A_mat[k, c] = V_new[j*p+k, c] * W_new[k*n+i, c]
            sol = _solve_exact_or_lstsq(A_mat, np.ones(p), r)
            if sol is not None:
                U_new[i*m+j, :] = sol
    
    # Update V
    for j in range(m):
        for k in range(p):
            A_mat = np.zeros((n, r))
            for i in range(n):
                for c in range(r):
                    A_mat[i, c] = U_new[i*m+j, c] * W_new[k*n+i, c]
            sol = _solve_exact_or_lstsq(A_mat, np.ones(n), r)
            if sol is not None:
                V_new[j*p+k, :] = sol
    
    # Update W
    for k in range(p):
        for i in range(n):
            A_mat = np.zeros((m, r))
            for j in range(m):
                for c in range(r):
                    A_mat[j, c] = U_new[i*m+j, c] * V_new[j*p+k, c]
            sol = _solve_exact_or_lstsq(A_mat, np.ones(m), r)
            if sol is not None:
                W_new[k*n+i, :] = sol
    
    return U_new, V_new, W_new

def _try_rank(r, n, m, p, deadline, seed=42):
    """Try to find a rank-r decomposition using random restarts and ALS."""
    rng = np.random.RandomState(seed)
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    num_attempts = 100
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Random initialization - use a mix of scales
        scale = 0.3 + 0.5 * rng.rand()
        U = rng.randn(n*m, r) * scale
        V = rng.randn(m*p, r) * scale
        W = rng.randn(p*n, r) * scale
        
        # Ensure no rows are zero
        for row in range(n*m):
            if np.all(np.abs(U[row]) < 1e-10):
                U[row, rng.randint(r)] = 0.1
        for row in range(m*p):
            if np.all(np.abs(V[row]) < 1e-10):
                V[row, rng.randint(r)] = 0.1
        for row in range(p*n):
            if np.all(np.abs(W[row]) < 1e-10):
                W[row, rng.randint(r)] = 0.1
        
        # ALS iterations
        prev_err = np.inf
        for iteration in range(300):
            if time.time() > deadline:
                break
            
            U, V, W = _als_iteration(U, V, W, n, m, p, r)
            
            err = _compute_error(U, V, W, n, m, p, r)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-12:
                break
            
            # Check for convergence
            if abs(prev_err - err) < 1e-15:
                break
            prev_err = err
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-8 and best_U is not None:
        # Try rounding to nearest half-integer
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        # Try integer rounding
        U_int = np.round(best_U)
        V_int = np.round(best_V)
        W_int = np.round(best_W)
        
        if _verify_decomposition(U_int, V_int, W_int, n, m, p):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
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
    
    deadline = time.time() + 140
    
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
