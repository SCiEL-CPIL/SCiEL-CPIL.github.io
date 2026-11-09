import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """
    Build the dense matrix multiplication tensor T of shape (n*m, m*p, n*p).
    T[a, b, c] = 1 if indices match, 0 otherwise.
    a = i*m + j
    b = j*p + k
    c = k*n + i  (Note: task says W[k*n+i], so c index is k*n+i)
    """
    T = np.zeros((n * m, m * p, n * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _compute_error(U, V, W, T):
    """
    Compute the squared error of the decomposition.
    U: (n*m, r)
    V: (m*p, r)
    W: (n*p, r)
    T: (n*m, m*p, n*p)
    """
    # pred[a, b, c] = sum_t U[a, t] * V[b, t] * W[c, t]
    # This is a batched matrix product or einsum
    # U[:, :, None] * V[None, :, :] -> (n*m, m*p, r)
    # Then dot with W.T?
    # Let's use einsum
    # pred = np.einsum('at,bt,ct->abc', U, V, W)
    # This might be slow for large r, but r is ~72.
    # Dimensions: 12, 24, 18, 72.
    # 12*24*18*72 operations ~ 370k. Very fast.
    pred = np.einsum('at,bt,ct->abc', U, V, W)
    err = np.sum((pred - T) ** 2)
    return err

def _try_reduce_rank(T, target_rank, max_time=10.0):
    """
    Try to find a decomposition of rank target_rank.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = n * p
    
    start_time = time.time()
    
    # Try a few random restarts
    for attempt in range(5):
        if time.time() - start_time > max_time:
            break
            
        # Initialize with random values
        U = np.random.randn(dim_u, target_rank) * 0.1
        V = np.random.randn(dim_v, target_rank) * 0.1
        W = np.random.randn(dim_w, target_rank) * 0.1
        
        # Simple gradient descent
        lr = 0.01
        prev_err = _compute_error(U, V, W, T)
        
        for step in range(1000):
            if time.time() - start_time > max_time:
                return None
            
            err = _compute_error(U, V, W, T)
            if err < 1e-10:
                return U, V, W
            
            # Compute gradients
            # dE/dU = 2 * (pred - T) contracted with V, W
            # pred = einsum('at,bt,ct->abc', U, V, W)
            # Let diff = pred - T
            # dU[a,t] = 2 * sum_b,c diff[a,b,c] V[b,t] W[c,t]
            
            pred = np.einsum('at,bt,ct->abc', U, V, W)
            diff = pred - T
            
            grad_U = 2 * np.einsum('abc,bt,ct->at', diff, V, W)
            grad_V = 2 * np.einsum('abc,at,ct->bt', diff, U, W)
            grad_W = 2 * np.einsum('abc,at,bt->ct', diff, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Optional: adjust learning rate if error increases
            new_err = _compute_error(U, V, W, T)
            if new_err > prev_err:
                lr *= 0.5
                U += (lr / (lr * 0.5)) * grad_U # Undo and re-apply with smaller lr? 
                # Simpler: just keep lr small
            else:
                prev_err = new_err
                
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for matrix multiplication 3x4 by 4x6.
    """
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Try to find a lower rank
    # Start from 71 down to 65
    for r in range(71, 64, -1):
        result = _try_reduce_rank(T, r, max_time=10.0)
        if result is not None:
            U, V, W = result
            # Verify
            err = _compute_error(U, V, W, T)
            if err < 1e-10:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, column] = 1
                V[v_idx, column] = 1
                W[w_idx, column] = 1
                column += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
