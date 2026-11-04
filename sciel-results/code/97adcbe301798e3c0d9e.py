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

def _optimize_decomposition(n, m, p, rank, T, time_budget, seed=42):
    """
    Use scipy.optimize to find a rank-decomposition.
    """
    start_time = time.time()
    
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
        x_half = np.round(x * 2) / 2.0
        penalty = np.sum((x - x_half) ** 2)
        
        return err + 50.0 * penalty

    np.random.seed(seed)
    best_result = None
    best_score = float('inf')
    
    max_restarts = 100 # Increased restarts
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
            
        # Initialization: Sparse random
        # Use a mix of 0, 0.5, 1.0, -0.5, -1.0
        x0 = np.random.choice([-1, 0, 1, 0.5, -0.5], size=total_size)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 2000})
            
            if res.fun < best_score:
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
            
    # If we have a good candidate, try to polish it with local search on lattice
    if best_result is not None:
        U = best_result[:size_U].reshape(n * m, rank)
        V = best_result[size_U:size_U + size_V].reshape(m * p, rank)
        W = best_result[size_U:size_U + size_V:].reshape(p * n, rank)
        
        # Snap to nearest half-integers as a starting point for local search
        U_snap = _snap_matrix(U, tol=0.5) # Loose snap
        V_snap = _snap_matrix(V, tol=0.5)
        W_snap = _snap_matrix(W, tol=0.5)
        
        if U_snap is not None and V_snap is not None and W_snap is not None:
            # Local search: flip individual entries to improve error
            # This is expensive, so we only do it if error is already low
            recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
            err = np.sum((recon - T) ** 2)
            if err < 1.0: # Threshold for local search
                # Simple local search: try changing each entry to its neighbors (half-integer steps)
                # This is O(N * 3) where N is number of variables
                # N = 494 for rank 19. 494 * 3 * 100 iterations = 150k ops. Feasible.
                # We will do a few passes.
                for _ in range(50):
                    if time.time() - start_time > time_budget:
                        break
                    improved = False
                    # Flatten for easier indexing
                    x_flat = np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()])
                    # We can't just flatten and search easily because of the tensor structure
                    # We'll iterate over U, V, W separately
                    
                    for mat_name, mat in [('U', U_snap), ('V', V_snap), ('W', W_snap)]:
                        for i in range(mat.shape[0]):
                            for j in range(mat.shape[1]):
                                old_val = mat[i, j]
                                # Try neighbors: old_val - 0.5, old_val + 0.5
                                for delta in [-0.5, 0.5]:
                                    new_val = old_val + delta
                                    # Only allow half-integers
                                    if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                                        continue
                                    mat[i, j] = new_val
                                    recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
                                    new_err = np.sum((recon - T) ** 2)
                                    if new_err < err - 1e-6:
                                        err = new_err
                                        improved = True
                                    else:
                                        mat[i, j] = old_val
                    if not improved:
                        break
                    
                    if err < 1e-9:
                        break
            
            if _check_decomposition(U_snap, V_snap, W_snap, T):
                return U_snap, V_snap, W_snap

    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    total_start = time.time()
    time_budget = 145.0 # Safety margin
    
    # Fallback: Schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    # Try ranks
    # Known lower bound for 2x3x4 is 19.
    for rank in [19, 20, 21, 22]:
        if rank > rank_fb:
            continue
            
        elapsed = time.time() - total_start
        remaining = time_budget - elapsed
        if remaining < 5.0:
            break
        
        # Allocate time
        if rank == 19:
            rank_budget = remaining * 0.8
        else:
            rank_budget = remaining * 0.2
            
        result = _optimize_decomposition(n, m, p, rank, T, rank_budget)
        
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
