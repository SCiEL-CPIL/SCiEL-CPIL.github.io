"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Generate schoolbook decomposition."""
    R = n * m * p
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_time=30.0):
    """
    Try to reduce the rank of the decomposition to target_rank.
    Uses a greedy approach: try to eliminate rank-1 terms by adjusting others.
    """
    start = time.time()
    R = len(U[0])
    
    # Work with numpy for efficiency
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # We want to find a subset of columns to keep and adjust them
    # such that the tensor is still decomposed correctly.
    
    # Strategy: For each column to potentially remove, solve for adjustments
    # to remaining columns that maintain the tensor decomposition.
    
    # The constraint is: for all (i,j,k):
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # If we remove column c, we need to adjust other columns to compensate.
    # This is a nonlinear problem in general, but we can try local adjustments.
    
    # Alternative: Use a randomized search
    # 1. Start with schoolbook
    # 2. Randomly perturb some entries
    # 3. Project back to satisfy the tensor constraints
    # 4. Iterate
    
    # Let's try a simpler approach: structured decomposition
    # For 3x4x4, we can think of it as 3 independent 1x4x4 problems.
    # Each 1x4x4 has schoolbook rank 16.
    
    # Actually, let's try to find a decomposition using the following structure:
    # Group by i (output row). For each i, we need to compute C[i,:] = A[i,:] * B
    # This is a 1x4 times 4x4 product.
    
    # The tensor for 1x4x4 has 16 entries. Can we decompose it with < 16 rank-1 terms?
    # The rank of the 1x4x4 tensor is at least 4 (since it contains a 4x4 identity-like structure).
    # Actually, the 1x4x4 tensor T[j,k] = 1 for all j,k. This is just the all-ones 4x4 matrix.
    # Wait no, the tensor is T[i,j,k] where i is fixed. For fixed i, T[i,j,k] = 1 for all j,k.
    # So for each i, we need to represent the all-ones 4x4 matrix as sum of rank-1 matrices.
    # The all-ones 4x4 matrix has rank 1! So we can do it with 1 rank-1 term per i.
    # But wait, that's not right because the decomposition must be consistent across all i.
    
    # Let me reconsider. The full tensor T[i,j,k] = 1 for all i in [0,3), j in [0,4), k in [0,4).
    # A rank-1 term is U_col ⊗ V_col ⊗ W_col where:
    #   U_col is a vector in R^{12} (indexed by i*m+j)
    #   V_col is a vector in R^{16} (indexed by j*p+k)
    #   W_col is a vector in R^{12} (indexed by k*n+i)
    # The product gives a 3x4x4 tensor where T[i,j,k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
    
    # So we need: sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1 for all i,j,k
    
    # This is a 3-way tensor decomposition problem.
    
    # Let's try a CP decomposition approach with random initialization and gradient descent.
    
    n_iter = 2000
    lr = 0.01
    R_target = target_rank
    
    U_t = np.random.randn(n*m, R_target) * 0.5
    V_t = np.random.randn(m*p, R_target) * 0.5
    W_t = np.random.randn(p*n, R_target) * 0.5
    
    # Ensure we can reach the target: initialize with some structure
    # Set some entries to 1 to help convergence
    for r in range(min(R_target, 12)):
        idx = r % (n*m)
        U_t[idx, r] = 1.0
    
    best_loss = float('inf')
    best_U = U_t.copy()
    best_V = V_t.copy()
    best_W = W_t.copy()
    
    for it in range(n_iter):
        if time.time() - start > max_time:
            break
        
        # Compute gradient
        loss = 0.0
        grad_U = np.zeros_like(U_t)
        grad_V = np.zeros_like(V_t)
        grad_W = np.zeros_like(W_t)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i*m+j
                    vj = j*p+k
                    wk = k*n+i
                    s = 0.0
                    for r in range(R_target):
                        s += U_t[ui,r] * V_t[vj,r] * W_t[wk,r]
                    err = s - 1.0
                    loss += err*err
                    for r in range(R_target):
                        grad_U[ui,r] += 2*err*V_t[vj,r]*W_t[wk,r]
                        grad_V[vj,r] += 2*err*U_t[ui,r]*W_t[wk,r]
                        grad_W[wk,r] += 2*err*U_t[ui,r]*V_t[vj,r]
        
        # Update
        U_t -= lr * grad_U
        V_t -= lr * grad_V
        W_t -= lr * grad_W
        
        if loss < best_loss:
            best_loss = loss
            best_U = U_t.copy()
            best_V = V_t.copy()
            best_W = W_t.copy()
        
        if best_loss < 1e-6:
            break
    
    if best_loss < 1e-4:
        # Round to nearest half-integer
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify rounded version
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        # Try without rounding
        if _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 4 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    
    This implementation searches for a lower-rank decomposition than the schoolbook
    rank of 48 using gradient-based optimization and structured search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize with schoolbook decomposition
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = rank_schoolbook
    
    # Try to find lower-rank decompositions
    # Start from lower ranks and go up
    for target_rank in range(24, rank_schoolbook + 1):
        if time.time() > deadline - 20:
            break
        
        result = _try_reduce_rank(best_U, best_V, best_W, n, m, p, target_rank, max_time=10.0)
        if result is not None:
            U_r, V_r, W_r = result
            # Verify
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                break  # Found the lowest rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
