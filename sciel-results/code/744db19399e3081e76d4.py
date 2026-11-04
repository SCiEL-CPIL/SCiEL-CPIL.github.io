"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 45
    
    # Try to find a lower-rank decomposition using randomized tensor decomposition
    # The multiplication tensor T has entries T[i,j,k] = 1 if we want to compute (AB)[i,k]
    # We need to find rank-1 tensors u ⊗ v ⊗ w such that sum_c U[c] ⊗ V[c] ⊗ W[c] = T
    
    # For matrix multiplication C = A*B where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 (the multiplication tensor)
    
    # Try various ranks below schoolbook
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    for rank in range(schoolbook_rank - 1, 15, -1):
        if time.time() > deadline - 30:
            break
        
        # Try to find a decomposition of this rank
        success, U, V, W = _try_decompose(n, m, p, rank, deadline)
        if success:
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
            # Continue trying to find even lower rank
            if rank <= 20:
                break
    # If we didn't find anything better, use schoolbook
    if best_U is None:
        return _schoolbook(n, m, p)
    
    # Convert to lists of lists
    U_list = [[U[i][c] for c in range(best_rank)] for i in range(n*m)]
    V_list = [[V[i][c] for c in range(best_rank)] for i in range(m*p)]
    W_list = [[W[i][c] for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_decompose(n, m, p, target_rank, deadline):
    """Try to find a rank-1 decomposition of the matrix multiplication tensor."""
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline - 20:
            break
        
        # Random initialization
        np.random.seed(attempt * 1000 + target_rank)
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        # Iterative refinement using ALS-style updates
        U, V, W, converged = _als_decompose(U, V, W, n, m, p, max_iter=200, tol=1e-10)
        
        if converged:
            # Verify exact decomposition
            if _verify(U, V, W, n, m, p):
                return True, U, V, W
    
    return False, None, None

def _als_decompose(U, V, W, n, m, p, max_iter=200, tol=1e-10):
    """ALS-style decomposition of the matrix multiplication tensor."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    prev_err = np.inf
    for it in range(max_iter):
        # Update U: for each (i,j), U[i*m+j, :] should satisfy
        # sum_j U[i*m+j,:] ⊗ V[j*p+:,:] ⊗ W[:,i] = T[i,j,:]
        # This is a linear system: for fixed i,j, 
        # sum_k W[k*n+i] * (V[j*p+k] ⊗ U[i*m+j]) = T[i,j,k] for all k
        
        # Actually, let's think about this differently.
        # The tensor T[i,j,k] = 1 for all i,j,k in the multiplication tensor.
        # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        # For fixed (i,j), varying k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # For fixed (i,k), varying j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # For fixed (j,k), varying i: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # ALS update for U: fix V and W, solve for U
        # For each (i,j): we need sum_c U[i*m+j,c] * (V[j*p+:,:] ⊙ W[:,i].T) ... 
        # This is complex. Let's use a simpler gradient-based approach.
        
        # Use simple gradient descent
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        # Compute gradient with respect to U
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # dT/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c] for each k
        
        # Simple iterative update
        lr = 0.01
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m + j
                    idx_v = j*p + k
                    idx_w = k*n + i
                    # Gradient for U[idx_u]
                    grad_u = V[idx_v] * W[idx_w]
                    # Target is 1, current value is sum_c U[idx_u,c]*V[idx_v,c]*W[idx_w,c]
                    current = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                    error = 1 - current
                    U_new[idx_u] += lr * error * V[idx_v] * W[idx_w]
                    
                    # Gradient for V[idx_v]
                    grad_v = U[idx_u] * W[idx_w]
                    V_new[idx_v] += lr * error * U[idx_u] * W[idx_w]
                    
                    # Gradient for W[idx_w]
                    grad_w = U[idx_u] * V[idx_v]
                    W_new[idx_w] += lr * error * U[idx_u] * V[idx_v]
        
        U, V, W = U_new, V_new, W_new
        
        # Compute error
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err += (val - 1)**2
        err = np.sqrt(err)
        
        if abs(prev_err - err) < tol:
            return U, V, W, True
        prev_err = err
    
    return U, V, W, False

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1) > tol:
                    return False
    return True

# EVOLVE-BLOCK-END
