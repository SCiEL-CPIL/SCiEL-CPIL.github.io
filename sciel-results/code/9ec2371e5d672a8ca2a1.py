"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _compute_error_and_grad(U, V, W, n, m, p):
    """Compute sum of squared errors and gradients for U, V, W."""
    rank = U.shape[1]
    err = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for c in range(rank):
                    s += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                e = s - 1.0
                err += e * e
                for c in range(rank):
                    g = 2.0 * e
                    grad_U[idx_u, c] += g * V[idx_v, c] * W[idx_w, c]
                    grad_V[idx_v, c] += g * U[idx_u, c] * W[idx_w, c]
                    grad_W[idx_w, c] += g * U[idx_u, c] * V[idx_v, c]
    
    return err, grad_U, grad_V, grad_W

def _gradient_descent(U, V, W, n, m, p, max_iters=2000, lr=0.01, tol=1e-10):
    """Run gradient descent to minimize the error."""
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    beta = 0.9
    
    for it in range(max_iters):
        err, gU, gV, gW = _compute_error_and_grad(U, V, W, n, m, p)
        if err < tol:
            break
        
        vU = beta * vU + (1 - beta) * gU
        vV = beta * vV + (1 - beta) * gV
        vW = beta * vW + (1 - beta) * gW
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
    
    return U, V, W, err

def _try_rank(rank, n, m, p, deadline, seed=42):
    """Try to find a valid decomposition of given rank."""
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    # Run gradient descent with multiple learning rates
    for lr in [0.001, 0.005, 0.01, 0.02, 0.05]:
        if time.time() > deadline:
            break
        U_try, V_try, W_try, err = _gradient_descent(
            U.copy(), V.copy(), W.copy(), n, m, p, 
            max_iters=500, lr=lr, tol=1e-10
        )
        if err < 1e-9:
            return U_try, V_try, W_try
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin (150s limit)
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try to reduce rank from high to low
    for r in range(best_rank - 1, 25, -1):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank(r, n, m, p, deadline - 5, seed=r * 7 + 13)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                print(f"Found valid decomposition of rank {r}")
                # Continue trying lower ranks
                continue
            else:
                print(f"Rank {r} result failed verification")
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
