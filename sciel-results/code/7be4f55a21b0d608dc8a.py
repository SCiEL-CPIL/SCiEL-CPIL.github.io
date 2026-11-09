"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    rank = Ua.shape[1]
    recon = np.zeros((n, m, p))
    for c in range(rank):
        U_c = Ua[:, c].reshape(n, m)
        V_c = Va[:, c].reshape(m, p)
        W_c = Wa[:, c].reshape(p, n)
        recon += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[np.newaxis, :, :]
    target = np.ones((n, m, p))
    diff = recon - target
    return np.linalg.norm(diff) < tol

def _snap_to_half_int(arr):
    """Snap values close to half-integers."""
    snapped = np.round(arr * 2) / 2
    return snapped

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual norm for the decomposition."""
    recon = np.zeros((n, m, p))
    rank = U.shape[1]
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        recon += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[np.newaxis, :, :]
    target = np.ones((n, m, p))
    return np.linalg.norm(recon - target)

def _als_optimize(U, V, W, n, m, p, max_iter, deadline, lr_start=0.5, lr_decay=0.995, lr_min=0.01):
    """Run ALS optimization with gradient descent. Returns (U, V, W, residual, converged)."""
    rank = U.shape[1]
    lr = lr_start
    residual = _compute_residual(U, V, W, n, m, p)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            return U, V, W, residual, False
        
        # Compute reconstruction and diff
        recon = np.zeros((n, m, p))
        for c in range(rank):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            recon += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[np.newaxis, :, :]
        
        target = np.ones((n, m, p))
        diff = recon - target
        residual = np.linalg.norm(diff)
        
        if residual < 1e-10:
            return U, V, W, residual, True
        
        # Vectorized gradient updates for all factors simultaneously
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        for c in range(rank):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            
            # Gradient for U_c[i,j] = sum_k diff[i,j,k] * V_c[j,k] * W_c[k,i]
            U_grad_c = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    U_grad_c[i, j] = np.sum(diff[i, j, :] * V_c[j, :] * W_c[:, i])
            U_new[:, c] = U[:, c] - lr * U_grad_c.flatten()
            
            # Gradient for V_c[j,k] = sum_i diff[i,j,k] * U_c[i,j] * W_c[k,i]
            V_grad_c = np.zeros((m, p))
            for j in range(m):
                for k in range(p):
                    V_grad_c[j, k] = np.sum(diff[:, j, k] * U_c[:, j] * W_c[k, :])
            V_new[:, c] = V[:, c] - lr * V_grad_c.flatten()
            
            # Gradient for W_c[k,i] = sum_j diff[i,j,k] * U_c[i,j] * V_c[j,k]
            W_grad_c = np.zeros((p, n))
            for k in range(p):
                for i in range(n):
                    W_grad_c[k, i] = np.sum(diff[i, :, k] * U_c[i, :] * V_c[:, k])
            W_new[:, c] = W[:, c] - lr * W_grad_c.flatten()
        
        U, V, W = U_new, V_new, W_new
        lr *= lr_decay
        if lr < lr_min:
            lr = lr_min
    
    residual = _compute_residual(U, V, W, n, m, p)
    return U, V, W, residual, residual < 1e-8

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Theoretical lower bound
    lower_bound = max(n * m, m * p, p * n)  # max(9, 15, 15) = 15
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try ranks from lower bound upward
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 30:
            break
        
        found = False
        max_restarts = 30
        
        for restart in range(max_restarts):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(42 + restart * 1000)
            
            # Random initialization
            scale = 0.5
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            W = np.random.randn(p * n, target_rank) * scale
            
            # Run ALS optimization
            U, V, W, residual, converged = _als_optimize(
                U, V, W, n, m, p,
                max_iter=500,
                deadline=deadline - 20,
                lr_start=0.5,
                lr_decay=0.995,
                lr_min=0.01
            )
            
            if converged or residual < 1e-8:
                # Found a valid decomposition
                # Try to snap to half-integers
                U_snapped = _snap_to_half_int(U)
                V_snapped = _snap_to_half_int(V)
                W_snapped = _snap_to_half_int(W)
                
                if _verify(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-6):
                    best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                else:
                    best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
