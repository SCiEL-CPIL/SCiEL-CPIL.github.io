import time
import numpy as np
from scipy import linalg

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 115  # safety margin before 120s limit

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
    
    # Precompute V and W reshaped for faster access
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    U3 = U.reshape(n, m, rank)
    
    # Solve for U
    for i in range(n):
        for j in range(m):
            A_mat = np.zeros((p, rank))
            for k in range(p):
                A_mat[k, :] = V3[j, k, :] * W3[k, i, :]
            b_vec = np.ones(p)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                U_new[i * m + j, :] = sol
            except:
                U_new[i * m + j, :] = U[i * m + j, :]
    
    # Solve for V
    for j in range(m):
        for k in range(p):
            A_mat = np.zeros((n, rank))
            for i in range(n):
                A_mat[i, :] = U3[i, j, :] * W3[k, i, :]
            b_vec = np.ones(n)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                V_new[j * p + k, :] = sol
            except:
                V_new[j * p + k, :] = V[j * p + k, :]
    
    # Solve for W
    for k in range(p):
        for i in range(n):
            A_mat = np.zeros((m, rank))
            for j in range(m):
                A_mat[j, :] = U3[i, j, :] * V3[j, k, :]
            b_vec = np.ones(m)
            try:
                sol = linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                W_new[k * n + i, :] = sol
            except:
                W_new[k * n + i, :] = W[k * n + i, :]
    
    return U_new, V_new, W_new

def _als_with_polish(U, V, W, n, m, p, max_iter=100, tol=1e-12):
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
    
    return best_U, best_V, best_W, best_err

def _generate_init(target_rank, n, m, p, restart):
    """Generate diverse initializations for a given rank."""
    np.random.seed(42 + restart * 1000)
    
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    U /= np.sqrt(target_rank)
    V /= np.sqrt(target_rank)
    W /= np.sqrt(target_rank)
    
    if restart == 0:
        # Structured: partial schoolbook pattern
        for r in range(target_rank):
            i = r % n
            j = (r // n) % m
            k = (r // (n * m)) % p
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
            U[i * m + j, r] += 0.01 * np.random.randn()
            V[j * p + k, r] += 0.01 * np.random.randn()
            W[k * n + i, r] += 0.01 * np.random.randn()
    elif restart == 1:
        # Sparse random
        for r in range(target_rank):
            idx_u = np.random.choice(n * m, size=min(2, n * m), replace=False)
            idx_v = np.random.choice(m * p, size=min(2, m * p), replace=False)
            idx_w = np.random.choice(p * n, size=min(2, p * n), replace=False)
            U[idx_u, r] = np.random.randn(len(idx_u))
            V[idx_v, r] = np.random.randn(len(idx_v))
            W[idx_w, r] = np.random.randn(len(idx_w))
    elif restart == 2:
        # Block-diagonal inspired
        block_size = max(1, target_rank // 4)
        for r in range(target_rank):
            block = r // block_size
            i = block % n
            j = (block // n) % m
            k = (block // (n * m)) % p
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
            U[i * m + j, r] += 0.05 * np.random.randn()
            V[j * p + k, r] += 0.05 * np.random.randn()
            W[k * n + i, r] += 0.05 * np.random.randn()
    elif restart == 3:
        # Binary-like init
        U = (np.random.rand(n * m, target_rank) > 0.5).astype(float) * 2 - 1
        V = (np.random.rand(m * p, target_rank) > 0.5).astype(float) * 2 - 1
        W = (np.random.rand(p * n, target_rank) > 0.5).astype(float) * 2 - 1
        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)
    elif restart == 4:
        # Random sparse with 3 non-zeros
        for r in range(target_rank):
            idx_u = np.random.choice(n * m, size=min(3, n * m), replace=False)
            idx_v = np.random.choice(m * p, size=min(3, m * p), replace=False)
            idx_w = np.random.choice(p * n, size=min(3, p * n), replace=False)
            U[idx_u, r] = np.random.randn(len(idx_u))
            V[idx_v, r] = np.random.randn(len(idx_v))
            W[idx_w, r] = np.random.randn(len(idx_w))

    return U, V, W

def _try_rank(target_rank, n, m, p, num_restarts=10, time_budget=None):
    """Try to find a rank-target_rank decomposition."""
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    if time_budget is None:
        time_budget = 15  # seconds per rank
    
    for restart in range(num_restarts):
        if time.time() - _START_TIME > _TIMEOUT:
            break
        
        U, V, W = _generate_init(target_rank, n, m, p, restart)
        U_sol, V_sol, W_sol, err = _als_with_polish(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_sol, V_sol, W_sol
        
        # Early exit if we found exact solution
        if best_err < 1e-12:
            break
    
    return best_U, best_V, best_W, best_err

def _round_to_half(X):
    """Round matrix to nearest 0.5."""
    return np.round(2 * X) / 2

def _verify_rounded(U, V, W, n, m, p):
    """Verify if rounded matrices give exact decomposition."""
    U_r = _round_to_half(U)
    V_r = _round_to_half(V)
    W_r = _round_to_half(W)
    T = _compute_tensor(U_r, V_r, W_r, n, m, p)
    err = _error(T)
    if err < 1e-10:
        return U_r, V_r, W_r
    return None

def construct():
    n, m, p = DIMENSIONS
    max_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_sb = np.zeros((n * m, max_rank))
    V_sb = np.zeros((m * p, max_rank))
    W_sb = np.zeros((p * n, max_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = max_rank
    
    # Try ranks from 24 to 30
    # 27 is the known rank for 4x4x4, so 3x4x4 should be <= 27
    candidate_ranks = [27, 26, 25, 24, 28, 29, 30]
    
    for target_rank in candidate_ranks:
        if time.time() - _START_TIME > _TIMEOUT - 5:
            break
        
        num_restarts = 15 if target_rank <= 27 else 10
        U_try, V_try, W_try, err = _try_rank(target_rank, n, m, p, num_restarts=num_restarts)
        
        if U_try is not None:
            # Try to round to half-integers
            rounded = _verify_rounded(U_try, V_try, W_try, n, m, p)
            if rounded is not None:
                best_U, best_V, best_W = rounded
                best_rank = target_rank
                # If we found a low rank, we can stop or try even lower
                if target_rank <= 24:
                    break
            elif err < 1e-10:
                # If not rounded but exact, use it (though contract prefers half-int)
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
    
    # Convert to lists for return
    U_list = [[float(U_list_val) for U_list_val in row] for row in best_U.tolist()]
    V_list = [[float(V_list_val) for V_list_val in row] for row in best_V.tolist()]
    W_list = [[float(W_list_val) for W_list_val in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
