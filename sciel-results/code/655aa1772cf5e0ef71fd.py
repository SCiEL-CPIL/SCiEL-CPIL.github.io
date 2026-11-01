"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import sys

def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    
    We start with the schoolbook rank (42) and attempt to reduce it using numerical methods and randomized approaches.
    If we cannot find a valid lower-rank decomposition, we return the schoolbook decomposition.
    
    The score is the rank, and we aim to minimize it.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # Function to create the schoolbook decomposition
    def schoolbook_decomposition():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Function to check if a decomposition is valid
    def check_decomposition(decomp, tol=1e-8):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        
        # Check shapes
        if U.shape[0] != n * m:
            return False
        if V.shape[0] != m * p:
            return False
        if W.shape[0] != p * n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        
        rank = U.shape[1]
        
        # Efficient check using tensor contraction
        # The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
        # We need: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = 1 if (i,j,k)==(i',j',k') else 0
        
        # Let's verify by checking the matrix multiplication property
        # For random matrices A (n x m) and B (m x p), we should have C = A @ B
        # And C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # Actually, the correct check is:
        # For the tensor decomposition, we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
        # This is because the matrix multiplication tensor is T[i,j,k] = 1 for all valid indices
        
        # Let's verify this with a direct check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    # Function to try to reduce rank using random initialization and optimization
    def try_reduce_rank(target_rank, max_iterations=5000, tol=1e-6):
        # Initialize U, V, W randomly
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Simple gradient-free optimization: try to minimize the error
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        actual = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                        error += (actual - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_error < tol:
                break
            
            # Random perturbation
            U += 0.1 * np.random.randn(n * m, target_rank)
            V += 0.1 * np.random.randn(m * p, target_rank)
            W += 0.1 * np.random.randn(p * n, target_rank)
        
        decomp = {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        if check_decomposition(decomp, tol):
            return decomp
        return None
    
    # Start with schoolbook
    best_decomp = schoolbook_decomposition()
    best_rank = rank_schoolbook
    
    # Try to reduce rank
    for target_rank in range(rank_schoolbook - 1, max(10, rank_schoolbook - 10), -1):
        if time.time() > deadline:
            break
        result = try_reduce_rank(target_rank)
        if result is not None:
            best_decomp = result
            best_rank = target_rank
            break
    
    return best_decomp
# EVOLVE-BLOCK-END
