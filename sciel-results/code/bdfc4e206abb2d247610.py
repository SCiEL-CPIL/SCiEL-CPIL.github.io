"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Pre-compute the index mappings for the matrix multiplication tensor
    # The tensor T has indices (a, b, c) where:
    # a = i*m + j (input row, col)
    # b = j*p + k (input col, output col)
    # c = k*n + i (output row, col)
    # T[a, b, c] = 1 if the indices are consistent, 0 otherwise
    
    # For a decomposition of rank R:
    # T[a,b,c] = sum_{r=0}^{R-1} U[a,r] * V[b,r] * W[c,r]
    
    # The naive decomposition has rank n*m*p = 24
    # We want to find a lower rank decomposition
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Start with naive decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    
    # Try to reduce rank by using a structured approach
    # For 2x3x4 matrix multiplication, we can try to find a rank-19 or lower decomposition
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Strategy: Use a randomized approach with structured initialization
    # and then refine using gradient-based methods
    
    np.random.seed(42)
    
    # Try various target ranks from low to high
    for target_rank in range(19, min(24, rank + 1)):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            U_try = np.zeros((n * m, target_rank))
            V_try = np.zeros((m * p, target_rank))
            W_try = np.zeros((p * n, target_rank))
            
            # Initialize with random values
            U_try[:, :] = np.random.randn(n * m, target_rank) * 0.5
            V_try[:, :] = np.random.randn(m * p, target_rank) * 0.5
            W_try[:, :] = np.random.randn(p * n, target_rank) * 0.5
            
            # Use Adam-like optimization with adaptive learning rate
            lr = 0.05
            for step in range(2000):
                if time.time() > deadline - 15:
                    break
                
                # Compute gradients efficiently
                grad_U = np.zeros_like(U_try)
                grad_V = np.zeros_like(V_try)
                grad_W = np.zeros_like(W_try)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            # Current value
                            val = np.sum(U_try[a, :] * V_try[b, :] * W_try[c, :])
                            err = val - 1.0
                            # Gradients
                            grad_U[a, :] += err * V_try[b, :] * W_try[c, :]
                            grad_V[b, :] += err * U_try[a, :] * W_try[c, :]
                            grad_W[c, :] += err * U_try[a, :] * V_try[b, :]
                
                # Update
                U_try -= lr * grad_U
                V_try -= lr * grad_V
                W_try -= lr * grad_W
                
                # Check if converged
                max_grad = max(np.max(np.abs(grad_U)), np.max(np.abs(grad_V)), np.max(np.abs(grad_W)))
                if max_grad < 1e-7:
                    break
            
            # Check if we found a valid decomposition
            U_list = U_try.tolist()
            V_list = V_try.tolist()
            W_list = W_try.tolist()
            
            if check_decomposition(U_list, V_list, W_list, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_list, V_list, W_list
                break
    
    # Convert to lists if needed
    if isinstance(best_U, np.ndarray):
        best_U = best_U.tolist()
    if isinstance(best_V, np.ndarray):
        best_V = best_V.tolist()
    if isinstance(best_W, np.ndarray):
        best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
