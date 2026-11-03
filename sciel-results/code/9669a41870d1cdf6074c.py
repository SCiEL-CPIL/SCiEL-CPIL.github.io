"""Matrix multiplication 3x3x6 tensor decomposition."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors for the decomposition (vectorized)."""
    # Build tensor of values: val[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    rank = U.shape[1]
    
    # Reshape for broadcasting
    # U[i*m+j, c] -> U_r[i, j, c]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # val[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # Use einsum
    val = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    error = np.sum((val - 1.0) ** 2)
    return error, val

def _compute_gradients(U, V, W, n, m, p, val=None):
    """Compute gradients of the error with respect to U, V, W (vectorized)."""
    rank = U.shape[1]
    
    if val is None:
        _, val = _compute_error(U, V, W, n, m, p)
    
    err = 2.0 * (val - 1.0)  # shape (n, m, p)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # dU[i,j,c] = sum_k err[i,j,k] * V_r[j,k,c] * W_r[k,i,c]
    dU_r = np.einsum('ijk,jkc,kic->ijc', err, V_r, W_r)
    # dV[j,k,c] = sum_i err[i,j,k] * U_r[i,j,c] * W_r[k,i,c]
    dV_r = np.einsum('ijk,ijc,kic->jkc', err, U_r, W_r)
    # dW[k,i,c] = sum_j err[i,j,k] * U_r[i,j,c] * V_r[j,k,c]
    dW_r = np.einsum('ijk,ijc,jkc->kic', err, U_r, V_r)
    
    dU = dU_r.reshape(n*m, rank)
    dV = dV_r.reshape(m*p, rank)
    dW = dW_r.reshape(p*n, rank)
    
    return dU, dV, dW

def _optimize_decomposition(U, V, W, n, m, p, max_iter=500, tol=1e-12, lr=0.05, time_limit=None, start_time=None):
    """Optimize the decomposition using gradient descent with adaptive learning rate (vectorized)."""
    if start_time is None:
        start_time = time.time()
    
    current_error, _ = _compute_error(U, V, W, n, m, p)
    best_error = current_error
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    effective_lr = lr
    prev_error = current_error
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        _, val = _compute_error(U, V, W, n, m, p)
        dU, dV, dW = _compute_gradients(U, V, W, n, m, p, val=val)
        
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm < 1e-14:
            break
        
        # Try the step
        U_new = U - effective_lr * dU
        V_new = V - effective_lr * dV
        W_new = W - effective_lr * dW
        
        new_error, _ = _compute_error(U_new, V_new, W_new, n, m, p)
        
        if new_error < prev_error:
            U, V, W = U_new, V_new, W_new
            # Increase lr slightly if making good progress
            if new_error < prev_error * 0.9:
                effective_lr = min(effective_lr * 1.02, 0.5)
            prev_error = new_error
            if new_error < best_error:
                best_error = new_error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            if best_error < tol:
                break
        else:
            # Decrease lr and retry
            effective_lr *= 0.5
            if effective_lr < 1e-10:
                break
    
    return best_U, best_V, best_W, best_error

def _structured_init(n, m, p, target_rank, seed):
    """
    Create a structured initialization for the decomposition.
    
    For p=6, view B as [B1 | B2] where B1, B2 are 3x3.
    C = [A*B1 | A*B2]
    
    We initialize with a pattern that shares some structure between the two halves.
    """
    np.random.seed(seed)
    
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    # Add some structure: make certain columns related
    # For the two halves of V (first 3*p and last 3*p rows), correlate them
    for c in range(min(target_rank // 2, p * n // 2)):
        # Make V and W have some shared structure
        V[0:3*p, c] = 0.5 * (V[0:3*p, c] + V[3*p:6*p, c])
        V[3*p:6*p, c] = 0.5 * (V[0:3*p, c] + V[3*p:6*p, c])
    
    return U, V, W

def _random_search(n, m, p, target_rank, num_restarts=20, time_limit=30, start_time=None):
    """Random search with multiple restarts and gradient optimization."""
    if start_time is None:
        start_time = time.time()
    
    best_result = None
    best_error = float('inf')
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        seed = 12345 + restart * 7919 + target_rank * 100003
        
        # Use structured initialization
        U, V, W = _structured_init(n, m, p, target_rank, seed)
        
        # Optimize
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining < 1:
            break
        
        U_opt, V_opt, W_opt, error = _optimize_decomposition(
            U, V, W, n, m, p,
            max_iter=300,
            tol=1e-12,
            lr=0.1,
            time_limit=time_remaining,
            start_time=start_time
        )
        
        if error < best_error:
            best_error = error
            best_result = (U_opt, V_opt, W_opt)
        
        if best_error < 1e-10:
            break
    
    return best_result, best_error

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = result
    
    # Deadline guard: 150s wall-clock, use 145s as safety margin
    deadline = time.time() + 145
    start_time = time.time()
    
    # Try various ranks from low to high
    # Theoretical lower bound is around n*p = 18, but likely higher
    # Try ranks from 30 to 53
    for target_rank in range(30, 54):
        if time.time() > deadline - 5:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 3:
            break
        
        # Allocate time based on rank
        # Lower ranks get more time since they're harder
        time_for_rank = min(time_remaining * 0.35, 15)
        
        search_result, error = _random_search(
            n, m, p, target_rank,
            num_restarts=50,
            time_limit=time_for_rank,
            start_time=start_time
        )
        
        if search_result is not None and error < 1e-8:
            # Verify
            U_list = search_result[0].tolist()
            V_list = search_result[1].tolist()
            W_list = search_result[2].tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
                best_rank = target_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found the lowest rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
