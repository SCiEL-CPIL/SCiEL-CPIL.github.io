"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return U, V, W as lists for the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W, tol=1e-6):
    """Verify that U, V, W form a valid rank-r decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_find_decomposition(n, m, p, target_rank, max_iter=500, tol=1e-8):
    """
    Try to find a rank decomposition using alternating least squares.
    Returns (U, V, W) as lists if found, else None.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    np.random.seed(42)
    
    for attempt in range(5):
        # Random initialization with small values
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        for iteration in range(max_iter):
            # Fix U and V, solve for W
            # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Group by (k, i): for fixed k, i, sum over j: sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
            
            # Build linear system for W
            # Unknowns: W[k*n+i, r] for all k, i, r
            # Equations: for each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # For each (k, i), the coefficient of W[k*n+i, r] in equation (i,j,k) is U[i*m+j, r] * V[j*p+k, r]
            # Sum over j: for fixed (k, i), sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (for each j)
            
            # Actually, let's do it differently. For each (i, j, k), we have:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is linear in W[k*n+i, :] for fixed U, V, i, j, k
            
            # Build the full linear system
            # Number of unknowns: pn * target_rank
            # Number of equations: n * m * p
            
            A = np.zeros((n * m * p, pn * target_rank))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            col_idx = (k * n + i) * target_rank + r
                            A[eq_idx, col_idx] = U[i*m+j, r] * V[j*p+k, r]
                        eq_idx += 1
            
            # Solve using least squares
            W_new, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            W = W_new.reshape(pn, target_rank)
            
            # Fix V and W, solve for U
            A = np.zeros((n * m * p, nm * target_rank))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            col_idx = (i * m + j) * target_rank + r
                            A[eq_idx, col_idx] = V[j*p+k, r] * W[k*n+i, r]
                        eq_idx += 1
            
            U_new, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            U = U_new.reshape(nm, target_rank)
            
            # Fix U and W, solve for V
            A = np.zeros((n * m * p, mp * target_rank))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            col_idx = (j * p + k) * target_rank + r
                            A[eq_idx, col_idx] = U[i*m+j, r] * W[k*n+i, r]
                        eq_idx += 1
            
            V_new, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            V = V_new.reshape(mp, target_rank)
            
            # Check convergence
            current = np.zeros((n, m, p))
            for r in range(target_rank):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            current[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
            
            target = np.ones((n, m, p))
            loss = np.sum((current - target) ** 2)
            
            if loss < tol:
                # Check if entries are close to integer or half-integer
                all_entries = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                rounded = np.round(all_entries * 2) / 2
                if np.max(np.abs(all_entries - rounded)) < 1e-4:
                    U_list = [[rounded[i*target_rank + r] for r in range(target_rank)] for i in range(nm)]
                    V_list = [[rounded[(nm + i)*target_rank + r] for r in range(target_rank)] for i in range(mp)]
                    W_list = [[rounded[(nm + mp + i)*target_rank + r] for r in range(target_rank)] for i in range(pn)]
                    
                    # Verify the decomposition
                    if _verify_decomposition(n, m, p, U_list, V_list, W_list, tol=1e-3):
                        return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions from rank 59 down to 30
    for r in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 15:
            break
        result = _try_find_decomposition(n, m, p, r, max_iter=100, tol=1e-6)
        if result is not None:
            U, V, W = result
            best_rank = r
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
