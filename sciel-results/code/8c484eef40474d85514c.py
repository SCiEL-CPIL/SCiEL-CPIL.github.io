"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W form an exact decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    T_target = np.ones((n, m, p))
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        u = np.array(U[:, c]).reshape(n, m)
        v = np.array(V[:, c]).reshape(m, p)
        w = np.array(W[:, c]).reshape(p, n)
        T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _compute_tensor(U, V, W, n, m, p, rank):
    """Compute the reconstructed tensor."""
    T = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return T

def _snap_to_half_integers(arr, tol=1e-4):
    """Snap values to nearest half-integer if close enough."""
    snapped = np.round(arr * 2) / 2.0
    if np.max(np.abs(arr - snapped)) > tol:
        return None
    return snapped

def _try_rank(n, m, p, target_rank, seed, time_budget):
    """Try to find a decomposition of the given rank using random init + local search."""
    rng = np.random.RandomState(seed)
    T_target = np.ones((n, m, p))
    
    best_error = float('inf')
    best_U = best_V = best_W = None
    
    # Multiple random initializations
    for init in range(10):
        if time_budget is not None and time.time() > time_budget:
            break
        
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        # Local search: gradient-like steps
        current_error = np.linalg.norm(_compute_tensor(U, V, W, n, m, p, target_rank) - T_target)
        
        for step in range(200):
            if time_budget is not None and time.time() > time_budget:
                break
            
            # Pick which factor to update
            comp = step % 3
            arr = [U, V, W][comp]
            row = rng.randint(arr.shape[0])
            col = rng.randint(arr.shape[1])
            
            old_val = arr[row, col]
            # Adaptive step size
            step_size = 0.1 / (1 + step * 0.01)
            new_val = old_val + rng.randn() * step_size
            arr[row, col] = new_val
            
            new_error = np.linalg.norm(_compute_tensor(U, V, W, n, m, p, target_rank) - T_target)
            
            if new_error < current_error:
                current_error = new_error
            else:
                arr[row, col] = old_val
        
        if current_error < best_error:
            best_error = current_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to snap to half-integers
    if best_U is not None:
        U_s = _snap_to_half_integers(best_U, tol=1e-3)
        V_s = _snap_to_half_integers(best_V, tol=1e-3)
        W_s = _snap_to_half_integers(best_W, tol=1e-3)
        
        if U_s is not None and V_s is not None and W_s is not None:
            U_list = [[U_s[i, c] for c in range(target_rank)] for i in range(n * m)]
            V_list = [[V_s[i, c] for c in range(target_rank)] for i in range(m * p)]
            W_list = [[W_s[i, c] for c in range(target_rank)] for i in range(p * n)]
            
            if _validate(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    # Also try without snapping but with exact validation
    if best_U is not None and best_error < 1e-6:
        U_list = [[best_U[i, c] for c in range(target_rank)] for i in range(n * m)]
        V_list = [[best_V[i, c] for c in range(target_rank)] for i in range(m * p)]
        W_list = [[best_W[i, c] for c in range(target_rank)] for i in range(p * n)]
        
        if _validate(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook baseline (rank 24)
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
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find lower rank decompositions
    # Lower bound is max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    for target_rank in range(12, best_rank):
        if time.time() > deadline - 5:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 3:
                break
            
            result = _try_rank(n, m, p, target_rank, 42 + seed, deadline - 2)
            if result is not None:
                U_list, V_list, W_list = result
                best_rank = target_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
