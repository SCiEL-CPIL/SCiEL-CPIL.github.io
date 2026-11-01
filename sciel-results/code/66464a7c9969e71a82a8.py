"""Matrix multiplication 2x5x5 with rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Try to find lowest rank decomposition
    # Start from rank 49 and try to go lower
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Precompute the tensor T[i,j,k] = 1 if C[i,k] depends on A[i,j]*B[j,k]
    # T has shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W with given rank exactly decompose the tensor."""
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Check for all a, b, c
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Extract indices
                    i = a // m
                    j = a % m
                    jj = b // p
                    k = b % p
                    kk = c // n
                    ii = c % n
                    
                    # The tensor entry T[a,b,c] is 1 if i==ii and j==jj and k==kk
                    expected = 1.0 if (i == ii and j == jj and k == kk) else 0.0
                    
                    # Compute sum
                    total = 0.0
                    for r in range(rank):
                        total += U[a][r] * V[b][r] * W[c][r]
                    
                    if abs(total - expected) > 1e-10:
                        return False
        return True
    
    def try_rank(rank):
        """Try to find a decomposition of given rank."""
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Gradient descent to minimize error
        lr = 0.01
        for step in range(5000):
            if time.time() > deadline:
                break
            
            # Compute error and gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            error = 0.0
            for a in range(n * m):
                i = a // m
                j = a % m
                for b in range(m * p):
                    jj = b // p
                    k = b % p
                    for c in range(p * n):
                        kk = c // n
                        ii = c % n
                        
                        expected = 1.0 if (i == ii and j == jj and k == kk) else 0.0
                        
                        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                        val = 0.0
                        for r in range(rank):
                            val += U[a, r] * V[b, r] * W[c, r]
                        
                        diff = val - expected
                        error += diff * diff
                        
                        # Gradients
                        for r in range(rank):
                            grad_U[a, r] += 2 * diff * V[b, r] * W[c, r]
                            grad_V[b, r] += 2 * diff * U[a, r] * W[c, r]
                            grad_W[c, r] += 2 * diff * U[a, r] * V[b, r]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if error < 1e-20:
                break
        
        # Check if we found a valid decomposition
        if error < 1e-15:
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U.tolist()]
            V_list = [[float(x) for x in row] for row in V.tolist()]
            W_list = [[float(x) for x in row] for row in W.tolist()]
            
            if verify_decomposition(U_list, V_list, W_list, rank):
                return (U_list, V_list, W_list)
        
        return None
    
    # Try ranks from 49 down to 40
    for rank in range(49, 39, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(rank)
        if result is not None:
            U_list, V_list, W_list = result
            best_rank = rank
            best_U = U_list
            best_V = V_list
            best_W = W_list
            break
    
    # If no better rank found, use schoolbook
    if best_U is None:
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
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
