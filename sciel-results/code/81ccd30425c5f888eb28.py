"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid product index."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose T."""
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    err = np.sum((recon - T) ** 2)
    return err < tol

def _snap_matrix(M, tol=1e-6):
    """Snap matrix entries to nearest integer or half-integer."""
    M_int = np.round(M)
    if np.allclose(M, M_int, atol=tol):
        return M_int
    M_half = np.round(M * 2) / 2.0
    if np.allclose(M, M_half, atol=tol):
        return M_half
    return None

def _optimize_decomposition(n, m, p, rank, T, time_budget):
    """
    Use scipy.optimize to find a rank-decomposition.
    Variables: flattened U, V, W.
    Objective: Reconstruction error + Penalty for non-half-integers.
    """
    start_time = time.time()
    
    # Sizes
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    def objective(x):
        if time.time() - start_time > time_budget:
            return 1e10
        
        U = x[:size_U].reshape(n * m, rank)
        V = x[size_U:size_U + size_V].reshape(m * p, rank)
        W = x[size_U + size_V:].reshape(p * n, rank)
        
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        err = np.sum((recon - T) ** 2)
        
        # Penalty for not being close to half-integers
        # x - 0.5 * round(2x)
        penalty = 0.0
        # We can compute this efficiently on the whole vector x
        # But x contains U, V, W.
        # Let's just compute it for all x
        x_half = np.round(x * 2) / 2.0
        penalty = np.sum((x - x_half) ** 2)
        
        return err + 10.0 * penalty

    # Multi-start
    np.random.seed(42)
    best_result = None
    best_score = float('inf')
    
    # Number of restarts depends on time
    # Each run takes ~0.1-1s. Let's allow up to 50 restarts.
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
            
        # Initialize
        # Sparse random initialization with values in {-1, 0, 1, 0.5, -0.5}
        x0 = np.random.choice([-1, 0, 1, 0.5, -0.5], size=total_size)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 1000})
            
            if res.success or (res.fun < best_score):
                best_score = res.fun
                best_result = res.x
                
                # Check if we found a perfect solution
                U = best_result[:size_U].reshape(n * m, rank)
                V = best_result[size_U:size_U + size_V].reshape(m * p, rank)
                W = best_result[size_U + size_V:].reshape(p * n, rank)
                
                U_snap = _snap_matrix(U)
                V_snap = _snap_matrix(V)
                W_snap = _snap_matrix(W)
                
                if U_snap is not None and V_snap is not None and W_snap is not None:
                    if _check_decomposition(U_snap, V_snap, W_snap, T):
                        return U_snap, V_snap, W_snap
                        
        except Exception:
            continue
            
    # If we have a good candidate, try to polish it
    if best_result is not None:
        U = best_result[:size_U].reshape(n * m, rank)
        V = best_result[size_U:size_U + size_V].reshape(m * p, rank)
        W = best_result[size_U + size_V:].reshape(p * n, rank)
        
        U_snap = _snap_matrix(U)
        V_snap = _snap_matrix(V)
        W_snap = _snap_matrix(W)
        
        if U_snap is not None and V_snap is not None and W_snap is not None:
            if _check_decomposition(U_snap, V_snap, W_snap, T):
                return U_snap, V_snap, W_snap
                
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    total_start = time.time()
    time_budget = 140.0
    
    # Fallback: Schoolbook
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Try ranks
    # Known lower bound for 2x3x4 is 19.
    for rank in [19, 20, 21, 22, 23]:
        if rank > rank_fb:
            continue
            
        elapsed = time.time() - total_start
        remaining = time_budget - elapsed
        if remaining < 2.0:
            break
        
        # Allocate time
        if rank == 19:
            rank_budget = remaining * 0.7
        elif rank == 20:
            rank_budget = remaining * 0.2
        else:
            rank_budget = remaining * 0.1
            
        result = _optimize_decomposition(n, m, p, rank, T, rank_budget)
        
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
