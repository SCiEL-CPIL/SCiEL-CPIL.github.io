"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * range(p * n)]
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
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # The tensor is all-ones 2x2x8
    
    # For rank r, we need to solve:
    # Sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Let's try a structured approach.
    # Note that T[i,j,k] = 1 = 1*1*1, so one trivial factorization is rank 1
    # but that would require U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 for all i,j,k
    # which is impossible since U depends on (i,j), V on (j,k), W on (k,i)
    
    # Let's try rank 2, 3, ... up to some limit
    # Use random initialization and gradient-free optimization
    
    def try_rank(r, max_iter=5000, seed=42):
        """Try to find a rank-r decomposition using optimization."""
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        def compute_error():
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        err += (val - 1.0) ** 2
            return err
        
        # Simple gradient descent
        lr = 0.01
        prev_err = compute_error()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        err_term = 2 * (val - 1.0)
                        for c in range(r):
                            dU[i*m+j, c] += err_term * V[j*p+k, c] * W[k*n+i, c]
                            dV[j*p+k, c] += err_term * U[i*m+j, c] * W[k*n+i, c]
                            dW[k*n+i, c] += err_term * U[i*m+j, c] * V[j*p+k, c]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            curr_err = compute_error()
            if curr_err < prev_err * 0.99:
                lr = min(lr * 1.1, 0.1)
            elif curr_err > prev_err:
                lr *= 0.5
            prev_err = curr_err
            
            if curr_err < 1e-12:
                break
        
        if prev_err < 1e-9:
            # Round to half-integers and verify
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from low to high
    for r in range(1, 25):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iter=3000, seed=r)
        if result is not None:
            return result
        
        # Try a few more seeds
        for seed in [r+100, r+200, r+300]:
            if time.time() > deadline - 10:
                break
            result = try_rank(r, max_iter=2000, seed=seed)
            if result is not None:
                return result
    
    # If no lower rank found, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
