"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W, "rank": rank}

def _verify(decomp, n, m, p):
    """Verify that the tensor decomposition exactly multiplies the matrices."""
    U = np.array(decomp["U"])
    V = np.array(decomp["V"])
    W = np.array(decomp["W"])
    r = U.shape[1]
    
    # Reshape to (n, m, r), (m, p, r), (p, n, r)
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # einsum: i j r, j k r, k i r -> i j k
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    # Check if T is close to 1 for all i,j,k
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _compute_loss_and_grads(U, V, W, n, m, p):
    """Compute loss and gradients for the tensor decomposition using vectorized numpy operations."""
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    r = U.shape[1]
    
    # Reshape to 3D
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    
    # Compute T
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    diff = T - 1.0
    loss = np.sum(diff * diff)
    
    # Gradients
    # dL/dU[i,j,r] = 2 * sum_k diff[i,j,k] * V[j,k,r] * W[k,i,r]
    grad_U_t = 2.0 * np.einsum('ijk,jkr,kir->ijr', diff, V_t, W_t)
    # dL/dV[j,k,r] = 2 * sum_i diff[i,j,k] * U[i,j,r] * W[k,i,r]
    grad_V_t = 2.0 * np.einsum('ijk,ijr,kir->jkr', diff, U_t, W_t)
    # dL/dW[k,i,r] = 2 * sum_j diff[i,j,k] * U[i,j,r] * V[j,k,r]
    grad_W_t = 2.0 * np.einsum('ijk,ijr,jkr->kir', diff, U_t, V_t)
    
    # Reshape gradients back to 2D
    grad_U = grad_U_t.reshape(n * m, r)
    grad_V = grad_V_t.reshape(m * p, r)
    grad_W = grad_W_t.reshape(p * n, r)
    
    return loss, grad_U, grad_V, grad_W

def _try_rank(n, m, p, rank, max_iter=1000, seed=42):
    """Try to find a rank-r decomposition using vectorized gradient descent."""
    rng = np.random.RandomState(seed)
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    best_loss = np.inf
    best_U, best_V, best_W = None, None, None
    
    # Try a few initializations
    for init_seed in range(2):
        init_rng = np.random.RandomState(seed + init_seed * 1000)
        
        # Scale initialization
        U = init_rng.randn(num_u, rank) * 0.5
        V = init_rng.randn(num_v, rank) * 0.5
        W = init_rng.randn(num_w, rank) * 0.5
        
        lr = 0.01
        for it in range(max_iter):
            loss, grad_U, grad_V, grad_W = _compute_loss_and_grads(U, V, W, n, m, p)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-12:
                break
            
            # Adaptive learning rate: simple decay or constant
            # For now, keep it simple
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Optional: reduce lr slightly
            if it % 100 == 99:
                lr *= 0.9
        
        if best_loss < 1e-12:
            break
    
    if best_U is not None and best_loss < 1e-12:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist(), "rank": rank}
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    best_decomp = _schoolbook()
    best_rank = best_decomp["rank"]
    
    # Try ranks from 14 upwards (known minimum for 2x5x6 is 14)
    # If 14 fails, try 15, 16... up to a reasonable limit
    for rank in range(14, 30):
        if time.time() > deadline:
            break
        
        # Try a few seeds for each rank
        for seed in range(3):
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, rank, max_iter=500, seed=seed + 100)
            if result is not None:
                if _verify(result, n, m, p):
                    best_decomp = result
                    best_rank = rank
                    break
        if best_rank == rank:
            break
    
    return {"U": best_decomp["U"], "V": best_decomp["V"], "W": best_decomp["W"]}
# EVOLVE-BLOCK-END
