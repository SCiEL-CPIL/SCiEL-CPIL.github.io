"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard: 120s wall-clock limit
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][column] = 1
                V_schoolbook[j * p + k][column] = 1
                W_schoolbook[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_schoolbook, V_schoolbook, W_schoolbook
    
    # Try to find a lower-rank decomposition using optimization
    # Strategy: minimize sparsity-regularized objective with continuous variables
    # then snap to half-integers
    
    # Target ranks to try (in increasing order)
    target_ranks = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:  # Need 30s remaining for verification
            break
        
        # Try to find a rank-`target_rank` decomposition
        found = _find_decomposition(n, m, p, target_rank, deadline)
        if found is not None:
            U, V, W = found
            # Verify the decomposition
            if _verify_decomposition(n, m, p, U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break  # Found the lowest rank we were trying
    
    return {"U": best_U, "V": best_V, "W": best_W}


def _find_decomposition(n, m, p, target_rank, deadline):
    """Try to find a rank-target_rank decomposition using optimization."""
    
    # Number of entries in each factor matrix
    nU = n * m  # 10
    nV = m * p  # 30
    nW = p * n  # 12
    
    # Initialize with random values
    np.random.seed(42)
    
    # Multiple restarts
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        # Initialize factor matrices
        U = np.random.randn(nU, target_rank) * 0.5
        V = np.random.randn(nV, target_rank) * 0.5
        W = np.random.randn(nW, target_rank) * 0.5
        
        # Gradient descent with sparsity regularization
        lr = 0.01
        sparsity_weight = 0.01
        
        # Precompute indices for faster residual computation
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        for iteration in range(200):
            if time.time() > deadline - 15:
                break
            
            # Compute residual: T_residual[i,j,k] = 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # For efficiency, compute in batches
            
            # Compute the current tensor product
            # This is expensive, so we'll use a simplified gradient approach
            
            # Gradient w.r.t. U[i,j,r]: sum_k V[j*p+k,r] * W[k*n+i,r] * (1 - current[i,j,k])
            # But we need current[i,j,k] for all i,j,k
            
            # Let's compute the full residual tensor
            # Shape: (n, m, p)
            residual = np.ones((n, m, p))
            
            for r in range(target_rank):
                # Compute outer product contribution for rank r
                # U[:, r] is (n*m,), V[:, r] is (m*p,), W[:, r] is (p*n,)
                # Reshape and compute
                Ur = U[:, r].reshape(n, m)
                Vr = V[:, r].reshape(m, p)
                Wr = W[:, r].reshape(p, n)
                
                # For each (i,j,k): add Ur[i,j] * Vr[j,k] * Wr[k,i]
                # This is a 3D operation
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual[i, j, k] -= Ur[i, j] * Vr[j, k] * Wr[k, i]
            
            # Check if residual is small enough
            max_residual = np.max(np.abs(residual))
            if max_residual < 1e-6:
                # Snap to half-integers
                U_snapped = _snap_to_half_integer(U)
                V_snapped = _snap_to_half_integer(V)
                W_snapped = _snap_to_half_integer(W)
                
                # Convert to lists
                U_list = U_snapped.tolist()
                V_list = V_snapped.tolist()
                W_list = W_snapped.tolist()
                
                return (U_list, V_list, W_list)
            
            # Compute gradients
            # grad_U[i,j,r] = sum_k residual[i,j,k] * V[j*p+k,r] * W[k*n+i,r]
            # grad_V[j,k,r] = sum_i residual[i,j,k] * U[i*m+j,r] * W[k*n+i,r]
            # grad_W[k,i,r] = sum_j residual[i,j,k] * U[i*m+j,r] * V[j*p+k,r]
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res_ijk = residual[i, j, k]
                        for r in range(target_rank):
                            # Gradient for U[i,j,r]
                            grad_U[i * m + j, r] += res_ijk * V[j * p + k, r] * W[k * n + i, r]
                            # Gradient for V[j,k,r]
                            grad_V[j * p + k, r] += res_ijk * U[i * m + j, r] * W[k * n + i, r]
                            # Gradient for W[k,i,r]
                            grad_W[k * n + i, r] += res_ijk * U[i * m + j, r] * V[j * p + k, r]
            
            # Add sparsity regularization
            grad_U += sparsity_weight * U
            grad_V += sparsity_weight * V
            grad_W += sparsity_weight * W
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            lr *= 0.99
    
    return None


def _snap_to_half_integer(matrix):
    """Snap matrix entries to nearest half-integer."""
    # Multiply by 2, round, divide by 2
    snapped = np.round(matrix * 2) / 2
    return snapped


def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    
    # Convert to numpy arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n * m, U_arr.shape[1]):
        return False
    if V_arr.shape != (m * p, V_arr.shape[1]):
        return False
    if W_arr.shape != (p * n, W_arr.shape[1]):
        return False
    
    rank = U_arr.shape[1]
    if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
        return False
    
    # Check that entries are half-integers
    U_int = U_arr * 2
    V_int = V_arr * 2
    W_int = W_arr * 2
    
    if not np.allclose(U_int, np.round(U_int), atol=1e-10):
        return False
    if not np.allclose(V_int, np.round(V_int), atol=1e-10):
        return False
    if not np.allclose(W_int, np.round(W_int), atol=1e-10):
        return False
    
    # Verify the tensor decomposition
    # For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(total - 1.0) > 1e-8:
                    return False
    
    return True
# EVOLVE-BLOCK-END
