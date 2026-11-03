"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 115  # safety margin before 120s limit

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _compute_tensor(U, V, W, n, m, p):
    """Compute the decomposed tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return T

def _error(T):
    """Compute max absolute error from all-ones tensor."""
    return np.max(np.abs(T - 1.0))

def _als_iteration(U, V, W, n, m, p):
    """
    One ALS iteration: solve for each factor given the other two.
    Returns new U, V, W.
    """
    rank = U.shape[1]
    U_new = np.zeros((n * m, rank))
    V_new = np.zeros((m * p, rank))
    W_new = np.zeros((p * n, rank))
    
    # Solve for U: for each (i,j), find U[i*m+j,:] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
    for i in range(n):
        for j in range(m):
            A_mat = np.zeros((p, rank))
            for k in range(p):
                A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
            b_vec = np.ones(p)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                U_new[i * m + j, :] = sol
            except:
                U_new[i * m + j, :] = U[i * m + j, :]
    
    # Solve for V: for each (j,k), find V[j*p+k,:] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
    for j in range(m):
        for k in range(p):
            A_mat = np.zeros((n, rank))
            for i in range(n):
                A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
            b_vec = np.ones(n)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                V_new[j * p + k, :] = sol
            except:
                V_new[j * p + k, :] = V[j * p + k, :]
    
    # Solve for W: for each (k,i), find W[k*n+i,:] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
    for k in range(p):
        for i in range(n):
            A_mat = np.zeros((m, rank))
            for j in range(m):
                A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b_vec = np.ones(m)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                W_new[k * n + i, :] = sol
            except:
                W_new[k * n + i, :] = W[k * n + i, :]
    
    return U_new, V_new, W_new

def _als_with_polish(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """Run ALS iterations with polish."""
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _error(_compute_tensor(U, V, W, n, m, p))
    
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        U, V, W = _als_iteration(U, V, W, n, m, p)
        T = _compute_tensor(U, V, W, n, m, p)
        err = _error(T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    # Polish pass: tighter tolerance, more iterations
    if best_err < 0.1 and time.time() - _START_TIME < _TIMEOUT - 10:
        U, V, W = best_U, best_V, best_W
        for it in range(30):
            if time.time() - _START_TIME > _TIMEOUT:
                break
            U, V, W = _als_iteration(U, V, W, n, m, p)
            T = _compute_tensor(U, V, W, n, m, p)
            err = _error(T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-14:
                break
    
    return best_U, best_V, best_W, best_err

def _try_rank(target_rank, n, m, p, num_restarts=5):
    """Try to find a rank-target_rank decomposition."""
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for restart in range(num_restarts):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        # Random initialization with structure
        np.random.seed(42 + restart * 1000)
        
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Normalize
        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)
        
        # Try structured init: use schoolbook-like patterns
        if restart == 0:
            # Initialize with partial schoolbook structure
            for r in range(target_rank):
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                # Add small random perturbation
                U[i * m + j, r] += 0.01 * np.random.randn()
                V[j * p + k, r] += 0.01 * np.random.randn()
                W[k * n + i, r] += 0.01 * np.random.randn()
        
        U_sol, V_sol, W_sol, err = _als_with_polish(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_sol, V_sol, W_sol
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    max_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = max_rank
    
    # Try to find lower rank decompositions
    # Known lower bound: for 3x4 * 4x4, rank is at least 33
    # Try ranks from 33 up to 47
    for target_rank in range(33, max_rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            break
        
        U_try, V_try, W_try, err = _try_rank(target_rank, n, m, p, num_restarts=3)
        
        if err < 1e-10:
            # Found exact decomposition
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = target_rank
            # Try even lower
            continue
        else:
            # If error is small, we might be close - try more restarts
            if err < 0.01 and time.time() - _START_TIME < _TIMEOUT - 10:
                U_try2, V_try2, W_try2, err2 = _try_rank(target_rank, n, m, p, num_restarts=5)
                if err2 < 1e-10:
                    best_U, best_V, best_W = U_try2, V_try2, W_try2
                    best_rank = target_rank
                    continue
    
    # Convert to lists for return
    U_list = [[float(U_list_val) for U_list_val in row] for row in best_U.tolist()]
    V_list = [[float(V_list_val) for V_list_val in row] for row in best_V.tolist()]
    W_list = [[float(W_list_val) for W_list_val in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
