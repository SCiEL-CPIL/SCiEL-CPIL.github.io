"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 36
    
    def verify_decomposition(U, V, W, n, m, p, rank, atol=1e-9):
        """
        Verify that T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
                    if not np.isclose(val, 1.0, atol=atol):
                        return False
        return True
    
    def trivial_decomposition(n, m, p):
        """Return the trivial rank n*m*p decomposition."""
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
    
    # Strategy: Use a structured ansatz inspired by Strassen-like algorithms.
    # For 3x3x3, we know rank 23 is achievable. For 3x3x4, we can try to extend this.
    # The key insight is that the 4th column of B can be treated specially.
    
    # Let's try a direct optimization approach using gradient descent on a 
    # smooth surrogate objective, followed by rounding to half-integers.
    
    def try_optimization(target_rank, n, m, p, max_iters=5000, seed=42):
        """
        Try to find a rank-r decomposition via optimization.
        We minimize the sum of squared errors: sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        """
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        lr = 0.01
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute loss and gradients
            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # val = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        uvw = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                        val = np.sum(uvw)
                        err = val - 1.0
                        loss += err * err
                        
                        # d(loss)/d(U[i*m+j, r]) = 2*err * V[j*p+k, r] * W[k*n+i, r]
                        dU[i * m + j, :] += 2.0 * err * V[j * p + k, :] * W[k * n + i, :]
                        dV[j * p + k, :] += 2.0 * err * U[i * m + j, :] * W[k * n + i, :]
                        dW[k * n + i, :] += 2.0 * err * U[i * m + j, :] * V[j * p + k, :]
            
            # Gradient descent step
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-12:
                break
        
        # Check if we found a valid solution
        if best_loss < 1e-8:
            # Try to round to half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, target_rank):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
            
            # If rounding fails, try without rounding (but entries must be half-integers)
            # This means we need to find a solution that's already at half-integer values
            # For now, return the continuous solution as a fallback (it won't be valid)
            return None
        
        return None
    
    # Try various ranks from low to high
    # Known: 3x3x3 has rank 23. For 3x3x4, the minimum rank is at least 23 (since it contains 3x3x3)
    # and at most 36 (trivial). Let's try ranks from 23 to 35.
    
    best_result = None
    
    for target_rank in range(23, 36):
        if time.time() > deadline:
            break
        
        result = try_optimization(target_rank, n, m, p, max_iters=2000)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            break
    
    # If no better decomposition found, return the trivial one
    if best_result is not None:
        return best_result
    
    # Fallback: trivial decomposition
    return trivial_decomposition(n, m, p)
# EVOLVE-BLOCK-END
