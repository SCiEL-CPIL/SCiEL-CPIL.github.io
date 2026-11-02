import time
import numpy as np
import random

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _compute_residual(U, V, W, tensor_a, tensor_b, tensor_c):
    """Compute sum of squared residuals."""
    UA = U[tensor_a, :]
    VB = V[tensor_b, :]
    WC = W[tensor_c, :]
    Pred = (UA * VB * WC).sum(axis=1)
    return np.sum((Pred - 1.0)**2)

def _compute_gradients(U, V, W, tensor_a, tensor_b, tensor_c):
    """Compute full-batch gradients for U, V, W."""
    rank = U.shape[1]
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    UA = U[tensor_a, :]
    VB = V[tensor_b, :]
    WC = W[tensor_c, :]
    
    Prod = UA * VB * WC
    Pred = Prod.sum(axis=1)
    Err = Pred - 1.0
    
    # For each tensor entry e, the gradient contribution to U[a_e, r] is:
    # 2 * Err_e * V[b_e, r] * W[c_e, r]
    # We accumulate over all entries
    
    # Vectorized approach: for each row index, accumulate contributions
    # Use bincount-like accumulation
    
    # grad_U[a, r] = 2 * sum_{e: a_e=a} Err_e * VB[e, r] * WC[e, r]
    # This is: 2 * sum_e Err_e * [a_e==a] * VB[e, r] * WC[e, r]
    
    # We can compute this by: for each r, grad_U[:, r] = 2 * (one_hot(a_e) @ (Err_e * VB[e, r] * WC[e, r]))
    # But that's expensive. Instead, loop over the small number of unique indices.
    
    # Since nm=4, mp=18, pn=18, and N=36 entries, we can afford a simple loop
    for e in range(len(tensor_a)):
        a, b, c = tensor_a[e], tensor_b[e], tensor_c[e]
        err_e = Err[e]
        # grad_U[a, :] += 2 * err_e * VB[e, :] * WC[e, :]
        grad_U[a, :] += 2 * err_e * VB[e, :] * WC[e, :]
        # grad_V[b, :] += 2 * err_e * UA[e, :] * WC[e, :]
        grad_V[b, :] += 2 * err_e * UA[e, :] * WC[e, :]
        # grad_W[c, :] += 2 * err_e * UA[e, :] * VB[e, :]
        grad_W[c, :] += 2 * err_e * UA[e, :] * VB[e, :]
    
    return grad_U, grad_V, grad_W

def _try_rank31_from_drop(U0, V0, W0, drop_col, tensor_a, tensor_b, tensor_c, time_budget, max_iter=2000):
    """Try to find rank-31 by dropping column and optimizing."""
    start_time = time.time()
    
    U = np.delete(U0, drop_col, axis=1).copy()
    V = np.delete(V0, drop_col, axis=1).copy()
    W = np.delete(W0, drop_col, axis=1).copy()
    
    rank = U.shape[1]
    
    current_res = _compute_residual(U, V, W, tensor_a, tensor_b, tensor_c)
    best_res = current_res
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Adaptive learning rate
    lr = 0.1
    patience = 0
    max_patience = 200
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        grad_U, grad_V, grad_W = _compute_gradients(U, V, W, tensor_a, tensor_b, tensor_c)
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        new_res = _compute_residual(U, V, W, tensor_a, tensor_b, tensor_c)
        
        if new_res < best_res:
            best_res = new_res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            patience = 0
            if best_res < 1e-10:
                break
        else:
            patience += 1
            if patience > max_patience:
                break
            lr *= 0.95
        
        if new_res < 1e-10:
            break
    
    return best_U, best_V, best_W, best_res

def _try_rank31_random_init(tensor_a, tensor_b, tensor_c, time_budget, max_iter=1000):
    """Try random initialization for rank-31."""
    start_time = time.time()
    
    nm, mp, pn = 4, 18, 18
    rank = 31
    
    best_res = float('inf')
    best_U, best_V, best_W = None, None, None
    
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget * 0.8:
            break
        
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        current_res = _compute_residual(U, V, W, tensor_a, tensor_b, tensor_c)
        best_local_res = current_res
        best_local_U, best_local_V, best_local_W = U.copy(), V.copy(), W.copy()
        
        lr = 0.05
        patience = 0
        max_patience = 100
        
        for iteration in range(max_iter):
            if time.time() - start_time > time_budget * 0.9:
                break
            
            grad_U, grad_V, grad_W = _compute_gradients(U, V, W, tensor_a, tensor_b, tensor_c)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            new_res = _compute_residual(U, V, W, tensor_a, tensor_b, tensor_c)
            
            if new_res < best_local_res:
                best_local_res = new_res
                best_local_U, best_local_V, best_local_W = U.copy(), V.copy(), W.copy()
                patience = 0
                if best_local_res < 1e-10:
                    break
            else:
                patience += 1
                if patience > max_patience:
                    break
                lr *= 0.95
            
            if new_res < 1e-10:
                break
        
        if best_local_res < best_res:
            best_res = best_local_res
            best_U, best_V, best_W = best_local_U, best_local_V, best_local_W
    
    return best_U, best_V, best_W, best_res

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Build rank-32 solution
    U0, V0, W0 = _build_rank32()
    
    # Build tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    tensor_a = np.array([e[0] for e in tensor_entries])
    tensor_b = np.array([e[1] for e in tensor_entries])
    tensor_c = np.array([e[2] for e in tensor_entries])
    
    # First, try dropping each column from rank-32 and optimizing
    target_rank = 31
    best_U, best_V, best_W = None, None, None
    best_res = float('inf')
    
    # Try dropping each column, but limit the number of attempts
    cols_to_try = list(range(32))
    random.shuffle(cols_to_try)
    
    time_per_attempt = 2.0  # seconds per attempt
    
    for drop_col in cols_to_try:
        if time.time() > deadline - 30:
            break
        
        U_trial, V_trial, W_trial, res = _try_rank31_from_drop(
            U0, V0, W0, drop_col, tensor_a, tensor_b, tensor_c, 
            time_budget=time_per_attempt
        )
        
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U_trial, V_trial, W_trial
        
        if best_res < 1e-10:
            break
    
    # If we found a good solution, return it
    if best_U is not None and best_res < 1e-10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try random initialization if we have time
    if time.time() < deadline - 20:
        U_rand, V_rand, W_rand, res_rand = _try_rank31_random_init(
            tensor_a, tensor_b, tensor_c, 
            time_budget=min(15, deadline - time.time() - 10)
        )
        
        if U_rand is not None and res_rand < best_res:
            best_U, best_V, best_W = U_rand, V_rand, W_rand
            best_res = res_rand
        
        if best_res < 1e-10:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # If no rank-31 found, return the rank-32 solution
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
