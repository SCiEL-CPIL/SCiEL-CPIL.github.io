"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a low-rank exact decomposition for 4x4 * 4x8 matrix multiplication.
    Uses iterative rank reduction with randomized initialization.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 128
    
    # Deadline guard
    deadline = time.time() + 140  # 140s to be safe within 150s
    
    # Start with schoolbook decomposition
    rank = schoolbook_rank
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify schoolbook is correct
    def verify(U, V, W, n, m, p, rank):
        """Check if U,V,W exactly decompose the tensor."""
        # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # should equal 1 if i==i, j==j, k==k (diagonal) and 0 otherwise
        # Actually, for matrix multiplication: (AB)[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid (i,j,k) combinations in the multiplication
        # Wait, let me reconsider. The tensor for matrix multiplication has
        # T[i,j,k] = 1 where i is row of A, j is col of A / row of B, k is col of B
        # The decomposition requires: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # for all (i,j,k) in the range, but actually only for the "diagonal" in some sense.
        
        # Actually, the standard tensor for matrix multiplication has entries:
        # T_{i,j,k} = 1 if this corresponds to a term in the product, 0 otherwise
        # For C = A*B where A is n x m, B is m x p, C is n x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because each such triple contributes to the product.
        
        # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to reduce rank using iterative projection
    # Strategy: try to find a lower rank by solving a least squares problem
    # and checking if it's exact
    
    def try_reduce_rank(U, V, W, n, m, p, current_rank, target_rank, max_iters=50):
        """Try to reduce rank from current_rank to target_rank."""
        if target_rank >= current_rank:
            return None
        
        U_curr = U[:, :current_rank].copy()
        V_curr = V[:, :current_rank].copy()
        W_curr = W[:, :current_rank].copy()
        
        # Random initialization for the reduced factors
        np.random.seed(42)
        U_new = np.random.randn(n * m, target_rank) * 0.1
        V_new = np.random.randn(m * p, target_rank) * 0.1
        W_new = np.random.randn(p * n, target_rank) * 0.1
        
        # Build the tensor equation: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is a system of nonlinear equations. We'll use a simple iterative approach.
        
        # Alternative: use the fact that if we have a rank-r decomposition, we can try to
        # find a rank-r' < r decomposition by projecting.
        
        # Let's try a different approach: use alternating minimization
        # Fix two factors, solve for the third.
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Fix V and W, solve for U
            # For each row i*m+j of U, we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is a linear system in U[i*m+j,:]
            
            U_solved = np.zeros((n * m, target_rank))
            for i in range(n):
                for j in range(m):
                    # Build the linear system: A * u = b where u = U[i*m+j,:]
                    # A[k, c] = V[j*p+k, c] * W[k*n+i, c] for k in range(p)
                    # b[k] = 1 for all k
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V_new[j * p + k, :] * W_new[k * n + i, :]
                    b = np.ones(p)
                    # Solve A * u = b
                    try:
                        u, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                        U_solved[i * m + j, :] = u
                    except:
                        U_solved[i * m + j, :] = U_curr[i * m + j, :target_rank]
            
            # Fix U and W, solve for V
            V_solved = np.zeros((m * p, target_rank))
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, target_rank))
                    for i in range(n):
                        A[i, :] = U_new[i * m + j, :] * W_new[k * n + i, :]
                    b = np.ones(n)
                    try:
                        v, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                        V_solved[j * p + k, :] = v
                    except:
                        V_solved[j * p + k, :] = V_curr[j * p + k, :target_rank]
            
            # Fix U and V, solve for W
            W_solved = np.zeros((p * n, target_rank))
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        A[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                    b = np.ones(m)
                    try:
                        w, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_solved[k * n + i, :] = w
                    except:
                        W_solved[k * n + i, :] = W_curr[k * n + i, :target_rank]
            
            # Update
            U_new = U_solved
            V_new = V_solved
            W_new = W_solved
        
        # Check if the solution is exact
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_new[i * m + j, :] * V_new[j * p + k, :] * W_new[k * n + i, :])
                    error += (s - 1.0) ** 2
        
        if error < 1e-6:
            return (U_new, V_new, W_new)
        return None
    
    # Try to reduce rank in steps
    for target_rank in range(schoolbook_rank - 1, max(10, schoolbook_rank // 2 - 1), -1):
        if time.time() > deadline:
            break
        
        result = try_reduce_rank(U, V, W, n, m, p, best_rank, target_rank, max_iters=30)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Convert to lists for return
    U_list = [[float(U[i, c]) for c in range(best_rank)] for i in range(n * m)]
    V_list = [[float(V[i, c]) for c in range(best_rank)] for i in range(m * p)]
    W_list = [[float(W[i, c]) for c in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
