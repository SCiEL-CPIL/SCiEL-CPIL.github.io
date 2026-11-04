"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _gradient_descent(n, m, p, rank, max_iters=5000, lr_init=0.01, seed=42):
    """Try to find a rank-`rank` decomposition via gradient descent."""
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Scale to make products of order 1
    scale = (1.0 / rank) ** (1.0/3.0)
    U *= scale
    V *= scale
    W *= scale
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = lr_init
    for it in range(max_iters):
        # Compute error and gradients
        err_sq = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # s = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    prod = U[i * m + j] * V[j * p + k] * W[k * n + i]
                    s = np.sum(prod)
                    grad = (s - 1.0)
                    err_sq += grad * grad
                    dU[i * m + j] += grad * V[j * p + k] * W[k * n + i]
                    dV[j * p + k] += grad * U[i * m + j] * W[k * n + i]
                    dW[k * n + i] += grad * U[i * m + j] * V[j * p + k]
        
        err = np.sqrt(err_sq / (n * m * p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-8:
            break
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        if it % 500 == 499:
            lr *= 0.5
    
    if best_err < 1e-8:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return U_list, V_list, W_list, best_err
    return None, None, None, best_err

def _try_rank(n, m, p, rank, max_trials=10, max_iters=3000):
    """Try multiple random seeds to find a valid decomposition of given rank."""
    for trial in range(max_trials):
        seed = 42 + trial * 7
        U, V, W, err = _gradient_descent(n, m, p, rank, max_iters=max_iters, seed=seed)
        if U is not None:
            return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Best solution starts as schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 90
    
    # Try to find lower ranks using gradient descent
    # For 3x5x6, try ranks from 45 upward (lower bound is roughly n*m*p/2 for some cases)
    for rank in range(45, best_rank):
        if time.time() > deadline - 5:
            break
        result = _try_rank(n, m, p, rank, max_trials=5, max_iters=2000)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = rank
                break  # Found a valid decomposition, use it
    
    return best
# EVOLVE-BLOCK-END
