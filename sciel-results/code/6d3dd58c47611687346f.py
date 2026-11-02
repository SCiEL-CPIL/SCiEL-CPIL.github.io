"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    def verify(U, V, W):
        """Verify exact decomposition with tolerance for floating point."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def round_to_halfint(arr, tol=1e-4):
        """Round array to nearest half-integer if close enough."""
        arr = np.asarray(arr, dtype=float)
        rounded = np.round(arr * 2) / 2
        if np.allclose(arr, rounded, atol=tol):
            return rounded
        return None
    
    # Try structured decompositions using known patterns
    # For small matrices, we can try block-based approaches
    
    def try_structured_decomposition():
        """Try various structured low-rank decompositions."""
        best = None
        
        # Try rank from lower bound to schoolbook
        # Lower bound is max(n*m, m*p, p*n) = max(10, 30, 12) = 30
        for rank in range(30, n * m * p + 1):
            # Try random structured initializations
            for trial in range(50):
                U = np.zeros((n * m, rank), dtype=float)
                V = np.zeros((m * p, rank), dtype=float)
                W = np.zeros((p * n, rank), dtype=float)
                
                # Initialize with random sparse patterns
                for t in range(rank):
                    # Each factor gets a random sparse pattern
                    U[:, t] = np.random.choice([-1, 0, 0, 0, 1], size=n * m)
                    V[:, t] = np.random.choice([-1, 0, 0, 0, 1], size=m * p)
                    W[:, t] = np.random.choice([-1, 0, 0, 0, 1], size=p * n)
                
                # Quick check before full verification
                if verify(U, V, W):
                    U_h = round_to_halfint(U)
                    V_h = round_to_halfint(V)
                    W_h = round_to_halfint(W)
                    if U_h is not None and V_h is not None and W_h is not None:
                        if verify(U_h, V_h, W_h):
                            result = {
                                "U": U_h.tolist(),
                                "V": V_h.tolist(),
                                "W": W_h.tolist()
                            }
                            if best is None or rank < best[0]:
                                best = (rank, result)
                            return best[1]
            
            # If we found a valid decomposition at this rank, return it
            if best is not None and best[0] == rank:
                return best[1]
        
        return None
    
    # Try gradient-based optimization with Lagrangian relaxation
    def try_gradient_optimization(max_rank=45):
        """Use gradient descent to find low-rank decompositions."""
        best = None
        
        for rank in range(30, max_rank + 1):
            for trial in range(20):
                # Random initialization
                U = np.random.randn(n * m, rank) * 0.5
                V = np.random.randn(m * p, rank) * 0.5
                W = np.random.randn(p * n, rank) * 0.5
                
                # Optimize using simple gradient descent on the error
                lr = 0.01
                for step in range(500):
                    # Compute error tensor
                    error = np.zeros((n, m, p), dtype=float)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for t in range(rank):
                                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                                error[i, j, k] = s - 1.0
                    
                    err_norm = np.linalg.norm(error)
                    if err_norm < 1e-5:
                        break
                    
                    # Compute gradients
                    dU = np.zeros_like(U)
                    dV = np.zeros_like(V)
                    dW = np.zeros_like(W)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                e = error[i, j, k]
                                for t in range(rank):
                                    dU[i * m + j, t] += e * V[j * p + k, t] * W[k * n + i, t]
                                    dV[j * p + k, t] += e * U[i * m + j, t] * W[k * n + i, t]
                                    dW[k * n + i, t] += e * U[i * m + j, t] * V[j * p + k, t]
                    
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                
                # Check if we found a valid solution
                if verify(U, V, W):
                    U_h = round_to_halfint(U, tol=1e-3)
                    V_h = round_to_halfint(V, tol=1e-3)
                    W_h = round_to_halfint(W, tol=1e-3)
                    if U_h is not None and V_h is not None and W_h is not None:
                        if verify(U_h, V_h, W_h):
                            result = {
                                "U": U_h.tolist(),
                                "V": V_h.tolist(),
                                "W": W_h.tolist()
                            }
                            if best is None or rank < best[0]:
                                best = (rank, result)
                            return best[1]
        
        return None
    
    # First try structured approaches
    result = try_structured_decomposition()
    if result is not None:
        return result
    
    # Then try gradient optimization
    result = try_gradient_optimization()
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END

