"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Create the standard schoolbook decomposition."""
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
    return U, V, W

def _gradient_descent_search(n, m, p, target_rank, max_time=60.0, seed=42):
    """Use gradient descent to find a low-rank decomposition."""
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # Add some structure: ensure each (i,j,k) triple has at least one contributing term
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = random.randint(0, target_rank - 1)
                U[i*m+j, c] += 1.0
                V[j*p+k, c] += 1.0
                W[k*n+i, c] += 1.0
    
    lr = 0.01
    best_loss = np.inf
    best_params = (U.copy(), V.copy(), W.copy())
    
    for iteration in range(100000):
        if time.time() - start_time > max_time:
            break
        
        # Compute loss: sum over all (i,j,k) of (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    diff = s - 1.0
                    loss += diff * diff
                    for c in range(target_rank):
                        grad_U[i*m+j, c] += 2 * diff * V[j*p+k, c] * W[k*n+i, c]
                        grad_V[j*p+k, c] += 2 * diff * U[i*m+j, c] * W[k*n+i, c]
                        grad_W[k*n+i, c] += 2 * diff * U[i*m+j, c] * V[j*p+k, c]
        
        # Update parameters
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if loss < best_loss:
            best_loss = loss
            best_params = (U.copy(), V.copy(), W.copy())
        
        if best_loss < 1e-10:
            break
        
        # Adaptive learning rate
        if iteration % 1000 == 0:
            lr *= 0.95
    
    # Check if we found a valid solution
    U_best, V_best, W_best = best_params
    valid = _verify(U_best.tolist(), V_best.tolist(), W_best.tolist(), n, m, p, tol=1e-6)
    
    if valid:
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    
    return None

def _structured_search(n, m, p, max_rank, max_time=100.0):
    """Structured search using tensor decomposition techniques."""
    start_time = time.time()
    
    # Try different rank values
    for target_rank in range(60, max_rank + 1):
        if time.time() - start_time > max_time:
            break
        
        # Try multiple random seeds
        for seed in range(20):
            if time.time() - start_time > max_time:
                break
            
            result = _gradient_descent_search(n, m, p, target_rank, max_time=5.0, seed=seed)
            if result is not None:
                return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Leave 5s safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # Try structured search
    result = _structured_search(n, m, p, max_rank=84, max_time=100.0)
    if result is not None:
        best_result = result
        best_rank = len(result["U"][0])
    
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
