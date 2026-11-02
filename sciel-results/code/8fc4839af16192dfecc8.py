"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    def is_exact(U, V, W, n, m, p):
        """Check if the given factors exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_gradient_descent(rank, max_iter=500, lr=0.01, num_restarts=5):
        """Use gradient descent to find a rank-r decomposition."""
        best_result = None
        
        for restart in range(num_restarts):
            # Initialize with small random values
            np.random.seed(42 + restart)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Scale to help with convergence
            scale = 1.0 / (rank ** 0.5)
            U *= scale
            V *= scale
            W *= scale
            
            for iteration in range(max_iter):
                # Compute the tensor values
                tensor = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            tensor[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            target = 1.0
                            error = tensor[i, j, k] - target
                            for t in range(rank):
                                grad_U[i*m+j, t] += error * V[j*p+k, t] * W[k*n+i, t]
                                grad_V[j*p+k, t] += error * U[i*m+j, t] * W[k*n+i, t]
                                grad_W[k*n+i, t] += error * U[i*m+j, t] * V[j*p+k, t]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if np.max(np.abs(tensor - 1.0)) < 1e-6:
                    # Round to nearest half-integer
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if is_exact(U_rounded, V_rounded, W_rounded, n, m, p):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    break
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound is max(n*m, m*p, p*n) = 30
    lower_bound = max(n*m, m*p, p*n)
    
    for rank in range(lower_bound, n*m*p + 1):
        result = try_gradient_descent(rank, max_iter=200, lr=0.05, num_restarts=3)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
