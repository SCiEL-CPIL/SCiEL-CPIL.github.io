"""Matrix multiplication 3x3x5 rank search with gradient-based optimization and deadline guard."""
import time
import random
import math
import numpy as np

DIMENSIONS = [3, 3, 5]
_START = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _validate(U, V, W):
    """Check exact tensor decomposition. Returns (rank, U, V, W) if valid, else None."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wk = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wk[c]
                if s != 1:
                    return None
    return (r, U, V, W)

def _schoolbook():
    """Schoolbook rank-45 decomposition."""
    n, m, p = DIMENSIONS
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
    return (rank, U, V, W)

def _compute_residual(U, V, W):
    """Compute sum of squared residuals for the tensor decomposition."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                res += (s - 1.0) ** 2
    return res

def _gradient_descent(rank, seed, max_steps=5000, lr_init=0.01):
    """
    Use gradient descent to find a rank-r decomposition.
    Returns (U, V, W) as lists of lists if found, else None.
    """
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    
    # Initialize with small random values
    def rand_init():
        return [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
    
    U = rand_init()
    V = rand_init()
    W = rand_init()
    
    # Scale to encourage non-zero solutions
    for i in range(len(U)):
        for j in range(rank):
            U[i][j] *= 0.5
            V[i][j] *= 0.5
            W[i][j] *= 0.5
    
    val = _compute_residual(U, V, W)
    lr = lr_init
    
    for step in range(max_steps):
        if _time_left() < 10.0:
            break
        
        # Compute gradients
        grad_U = [[0.0] * rank for _ in range(n * m)]
        grad_V = [[0.0] * rank for _ in range(m * p)]
        grad_W = [[0.0] * rank for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute s and its partial derivatives
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    ds = 2.0 * (s - 1.0)
                    
                    for c in range(rank):
                        grad_U[i * m + j][c] += ds * V[j * p + k][c] * W[k * n + i][c]
                        grad_V[j * p + k][c] += ds * U[i * m + j][c] * W[k * n + i][c]
                        grad_W[k * n + i][c] += ds * U[i * m + j][c] * V[j * p + k][c]
        
        # Update parameters
        for i in range(n * m):
            for c in range(rank):
                U[i][c] -= lr * grad_U[i][c]
        for i in range(m * p):
            for c in range(rank):
                V[i][c] -= lr * grad_V[i][c]
        for i in range(p * n):
            for c in range(rank):
                W[i][c] -= lr * grad_W[i][c]
        
        new_val = _compute_residual(U, V, W)
        
        if new_val < val:
            val = new_val
            lr *= 0.99
        else:
            # Roll back
            lr *= 0.5
        
        # Check for convergence
        if val < 1e-12:
            # Round to nearest half-integer
            U_rounded = [[(int(round(x * 2)) / 2) for x in row] for row in U]
            V_rounded = [[(int(round(x * 2)) / 2) for x in row] for row in V]
            W_rounded = [[(int(round(x * 2)) / 2) for x in row] for row in W]
            if _validate(U_rounded, V_rounded, W_rounded) is not None:
                return (U_rounded, V_rounded, W_rounded)
        
        # Early stop if stuck
        if step > 100 and val > 1.0:
            break
    
    return None

def _try_rank(rank, num_trials=5):
    """Try to find a rank-r decomposition using gradient descent."""
    for trial in range(num_trials):
        if _time_left() < 15.0:
            break
        
        result = _gradient_descent(rank, seed=42 + trial * 7, max_steps=3000, lr_init=0.005)
        if result is not None:
            U, V, W = result
            if _validate(U, V, W) is not None:
                return (rank, U, V, W)
    
    return None

def _strassen_based_3x3x3():
    """
    Try to use Strassen-like structure for the 3x3x3 part.
    This is a heuristic construction.
    """
    n, m, p = DIMENSIONS
    # For now, return None - this is a placeholder
    return None

def construct():
    """Main entry point. Returns {'U', 'V', 'W'} with the best valid decomposition found."""
    # Start with schoolbook as fallback
    best_rank, best_U, best_V, best_W = _schoolbook()
    
    # Try lower ranks first
    for rank in [15, 16, 17, 18, 19, 20]:
        if _time_left() < 20.0:
            break
        
        result = _try_rank(rank, num_trials=3)
        if result is not None:
            r, U, V, W = result
            if _validate(U, V, W) is not None:
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
