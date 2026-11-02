"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_schoolbook(n, m, p):
    """Return (U, V, W) as numpy arrays for schoolbook rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W correctly decompose the multiplication tensor.
    
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _gradient_descent(n, m, p, rank, max_iters=500, lr_init=0.05, deadline=None):
    """
    Attempt to find a rank-decomposition using gradient descent.
    Returns (U, V, W, success)
    """
    # Initialize
    # Scale initialization to be in a reasonable range
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    lr = lr_init
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Reshape for einsum
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Forward pass
        # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Error
        E = T - 1.0
        
        # Check convergence
        max_err = np.max(np.abs(E))
        if max_err < 1e-6:
            return U, V, W, True
            
        # Gradients
        dU_3d = 2 * np.einsum('ijk,jkr,kir->ijr', E, V_3d, W_3d)
        dV_3d = 2 * np.einsum('ijk,ijr,kir->jkr', E, U_3d, W_3d)
        dW_3d = 2 * np.einsum('ijk,ijr,jkr->kir', E, U_3d, V_3d)
        
        # Update
        U -= lr * dU_3d.reshape(n*m, rank)
        V -= lr * dV_3d.reshape(m*p, rank)
        W -= lr * dW_3d.reshape(p*n, rank)
        
        # Decay learning rate
        lr *= 0.995
        
    return U, V, W, False

def _snap_to_half(U, V, W):
    """Snap matrices to nearest half-integers and validate."""
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0
    return U_s, V_s, W_s

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook (rank 72)
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p  # 72
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank decompositions using Gradient Descent
    # We expect the rank to be significantly lower than 72.
    # Let's try ranks from 45 to 71.
    
    for target_rank in range(45, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few restarts
        for restart in range(10):
            if time.time() > deadline:
                break
                
            U_try, V_try, W_try, success = _gradient_descent(n, m, p, target_rank, max_iters=300, lr_init=0.1, deadline=deadline)
            
            if success:
                # Try to snap
                U_snap, V_snap, W_snap = _snap_to_half(U_try, V_try, W_try)
                if _validate(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                    best_rank = target_rank
                    best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    found = True
                    break
                else:
                    # If snapping fails, try to refine with smaller steps?
                    # For now, skip this rank if we can't snap.
                    pass
                    
        if found:
            break
            
    return best_result

# EVOLVE-BLOCK-END
