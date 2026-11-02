"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
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
        r = U.shape[1] if hasattr(U, 'shape') else len(U[0])
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
    
    def round_to_half_integer(x):
        """Round a float to the nearest half-integer."""
        return 2.0 * round(x * 2.0) / 2.0
    
    def try_gradient_descent(rank, num_restarts=5, max_iter=500, lr=0.01):
        """
        Try to find a rank-r decomposition using gradient descent on the 
        non-convex QCQP formulation. We minimize the sum of squared errors
        of the tensor reconstruction.
        """
        best_result = None
        best_error = float('inf')
        
        for restart in range(num_restarts):
            # Initialize U, V, W with random values
            np.random.seed(42 + restart)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            for iteration in range(max_iter):
                # Compute the tensor reconstruction error
                # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                # We want T[i,j,k] = 1 for all i,j,k
                
                # Compute gradients using automatic differentiation-like approach
                # For efficiency, compute the error and gradients numerically or analytically
                
                # Compute current tensor values
                T = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for t in range(rank):
                                s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                            T[i, j, k] = s
                
                error = np.sum((T - 1.0) ** 2)
                
                if error < best_error:
                    best_error = error
                    # Check if we're close enough to consider it a solution
                    if error < 1e-10:
                        # Round to half-integers and verify
                        U_rounded = np.vectorize(round_to_half_integer)(U)
                        V_rounded = np.vectorize(round_to_half_integer)(V)
                        W_rounded = np.vectorize(round_to_half_integer)(W)
                        
                        if is_exact(U_rounded, V_rounded, W_rounded, n, m, p):
                            return {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = T[i, j, k] - 1.0
                            for t in range(rank):
                                u_val = U[i * m + j, t]
                                v_val = V[j * p + k, t]
                                w_val = W[k * n + i, t]
                                dU[i * m + j, t] += 2 * err * v_val * w_val
                                dV[j * p + k, t] += 2 * err * u_val * w_val
                                dW[k * n + i, t] += 2 * err * u_val * v_val
                
                # Update parameters
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if error < 1e-10:
                    break
        
        return None
    
    # Try to find a lower-rank decomposition
    # The trivial lower bound is max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Let's try ranks from 30 up to 60 (schoolbook)
    
    for rank in range(30, 61):
        result = try_gradient_descent(rank, num_restarts=3, max_iter=200, lr=0.005)
        if result is not None:
            return result
    
    # If no lower-rank solution found, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
