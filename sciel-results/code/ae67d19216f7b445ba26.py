import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _compute_error(U, V, W, n, m, p):
    """Compute the squared error of the decomposition."""
    rank = U.shape[2]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.sum((T - 1.0)**2)

def _compute_gradient(U, V, W, n, m, p):
    """Compute the gradient of the squared error."""
    rank = U.shape[2]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    err = T - 1.0
    
    dU = np.zeros_like(U_r)
    dV = np.zeros_like(V_r)
    dW = np.zeros_like(W_r)
    
    # dU[i,j,c] = sum_k err[i,j,k] * V[j,k,c] * W[k,i,c]
    # dV[j,k,c] = sum_i err[i,j,k] * U[i,j,c] * W[k,i,c]
    # dW[k,i,c] = sum_j err[i,j,k] * U[i,j,c] * V[j,k,c]
    
    dU = np.einsum('ijk,jkc,kic->ijc', err, V_r, W_r)
    dV = np.einsum('ijk,ijc,kic->jkc', err, U_r, W_r)
    dW = np.einsum('ijk,ijc,jkc->kic', err, U_r, V_r)
    
    # Factor of 2 for squared error
    return 2.0 * dU, 2.0 * dV, 2.0 * dW

def _gd_decomposition(n, m, p, rank, iters=2000, lr=0.01, seed=0):
    """Gradient descent to find decomposition."""
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.reshape(p * n, rank) * 0.1
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    current_err = _compute_error(U_r, V_r, W_r, n, m, p)
    
    for it in range(iters):
        dU, dV, dW = _compute_gradient(U_r, V_r, W_r, n, m, p)
        
        U_r -= lr * dU
        V_r -= lr * dV
        W_r -= lr * dW
        
        # Resync
        U = U_r.reshape(n * m, rank)
        V = V_r.reshape(m * p, rank)
        W = W_r.reshape(p * n, rank)
        
        if it % 50 == 0:
            new_err = _compute_error(U_r, V_r, W_r, n, m, p)
            if new_err < 1e-6:
                return U.tolist(), V.tolist(), W.tolist()
            # Simple learning rate decay
            lr *= 0.95
            
    new_err = _compute_error(U_r, V_r, W_r, n, m, p)
    if new_err < 1e-6:
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using NumPy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    return np.allclose(T, 1.0, atol=1e-6)

def _verify_slow(U, V, W, n, m, p):
    """Slow but reliable verification."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1) > 1e-6:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from lower bound upward
    # Lower bound for 4x5x5 is roughly 36-40
    ranks_to_try = list(range(36, 60))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        max_seeds = 5
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
            
            result = _gd_decomposition(n, m, p, target_rank, iters=1000, lr=0.05, seed=seed)
            if result is not None:
                U_list, V_list, W_list = result
                # Round to half-integers if possible, but keep floats for now
                # The task allows integer/half-integer. 
                # We can try to snap to half-integers.
                
                U_rounded = [[round(2 * x) / 2 for x in row] for row in U_list]
                V_rounded = [[round(2 * x) / 2 for x in row] for row in V_list]
                W_rounded = [[round(2 * x) / 2 for x in row] for row in W_list]
                
                if _verify_slow(U_rounded, V_rounded, W_rounded, n, m, p):
                    if target_rank < best_rank:
                        best = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                        best_rank = target_rank
                        if best_rank <= 40:
                            return best
                            
    return best
# EVOLVE-BLOCK-END
