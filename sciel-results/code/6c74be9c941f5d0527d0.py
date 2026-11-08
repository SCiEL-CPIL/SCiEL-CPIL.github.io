"""Combined program: structural schoolbook base with gradient-based low-rank search."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]
_DEADLINE = time.time() + 115.0  # Safety margin before 120s limit

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return U, V, W

def _compute_error_vectorized(theta, n, m, p, rank):
    """
    Compute the error and gradient for a given flattened parameter vector.
    theta: array of shape (47 * rank,)
    """
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    
    U_flat = theta[:size_U]
    V_flat = theta[size_U:size_U+size_V]
    W_flat = theta[size_U+size_V:]
    
    U = U_flat.reshape(n, m, rank)
    V = V_flat.reshape(m, p, rank)
    W = W_flat.reshape(p, n, rank)
    
    T_recon = np.einsum('ijr,jkr,kin->ijk', U, V, W)
    
    error = T_recon - 1.0
    squared_error = np.sum(error**2)
    
    grad_U = 2.0 * np.einsum('ijk,jkr,kin->ijr', error, V, W)
    grad_V = 2.0 * np.einsum('ijk,ijr,kin->jkr', error, U, W)
    grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kin', error, U, V)
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    return squared_error, grad

def _try_rank_optimization(n, m, p, rank, num_starts=5, max_iters=100):
    """
    Try to find a decomposition of the given rank using gradient descent.
    """
    best_result = None
    best_error = float('inf')
    
    total_vars = 47 * rank
    
    for start_idx in range(num_starts):
        if time.time() > _DEADLINE:
            break
            
        rng = np.random.RandomState(start_idx * 100 + rank)
        theta0 = rng.randn(total_vars) * 0.1
        
        res = minimize(
            _compute_error_vectorized, 
            theta0, 
            args=(n, m, p, rank), 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-9}
        )
        
        if res.fun < best_error:
            best_error = res.fun
            best_theta = res.x
            
            if best_error < 1e-9:
                size_U = n * m * rank
                size_V = m * p * rank
                size_W = p * n * rank
                
                U_flat = best_theta[:size_U]
                V_flat = best_theta[size_U:size_U+size_V]
                W_flat = best_theta[size_U+size_V:]
                
                U = U_flat.reshape(n, m, rank)
                V = V_flat.reshape(m, p, rank)
                W = W_flat.reshape(p, n, rank)
                
                U_list = [[float(U[i, j, r]) for r in range(rank)] for i in range(n) for j in range(m)]
                V_list = [[float(V[j, k, r]) for r in range(rank)] for j in range(m) for k in range(p)]
                W_list = [[float(W[k, i, r]) for r in range(rank)] for k in range(p) for i in range(n)]
                
                # Try to snap to 0.5
                U_arr = np.array(U_list)
                V_arr = np.array(V_list)
                W_arr = np.array(W_list)
                
                U_snapped = np.round(U_arr * 2.0) / 2.0
                V_snapped = np.round(V_arr * 2.0) / 2.0
                W_snapped = np.round(W_arr * 2.0) / 2.0
                
                U_s_list = U_snapped.tolist()
                V_s_list = V_snapped.tolist()
                W_s_list = W_snapped.tolist()
                
                if _verify(U_s_list, V_s_list, W_s_list, n, m, p, rank):
                    return U_s_list, V_s_list, W_s_list, rank, best_error
                
                if _verify(U_list, V_list, W_list, n, m, p, rank):
                    return U_list, V_list, W_list, rank, best_error

    if best_theta is not None and best_error < 1e-6:
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U_flat = best_theta[:size_U]
        V_flat = best_theta[size_U:size_U+size_V]
        W_flat = best_theta[size_U+size_V:]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        U_list = [[float(U[i, j, r]) for r in range(rank)] for i in range(n) for j in range(m)]
        V_list = [[float(V[j, k, r]) for r in range(rank)] for j in range(m) for k in range(p)]
        W_list = [[float(W[k, i, r]) for r in range(rank)] for k in range(p) for i in range(n)]
        
        return U_list, V_list, W_list, rank, best_error
        
    return None

def construct():
    n, m, p = DIMENSIONS
    
    for rank in range(20, 60):
        if time.time() > _DEADLINE:
            break
            
        num_starts = 5 if rank < 30 else 2
        
        result = _try_rank_optimization(n, m, p, rank, num_starts=num_starts, max_iters=100)
        
        if result is not None:
            U, V, W, r, err = result
            if err < 1e-9:
                return {"U": U, "V": V, "W": W}
            
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
