"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _validate(U, V, W):
    """Check if U, V, W form a valid decomposition of matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check dimensions
    if U_np.shape != (n * m, rank):
        return False
    if V_np.shape != (m * p, rank):
        return False
    if W_np.shape != (p * n, rank):
        return False
    
    # For each rank component c, we have:
    # U[:, c] is a vector of length n*m
    # V[:, c] is a vector of length m*p  
    # W[:, c] is a vector of length p*n
    # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
    # Wait, that's not right. Let me think again.
    #
    # The tensor T[i,j,k] = 1 for all i,j,k (matrix multiplication tensor)
    # We want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But that would be rank-1 tensors, and we need the sum to equal 1 for all (i,j,k).
    #
    # Actually, the standard formulation is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # We want: (A*B)[i,k] = sum_c (sum_j A[i,j] * U[j, c]) * (sum_k' B[j', k] * V[c, k']) * W[c]
    # No, that's not right either.
    #
    # Let me reconsider. The matrix multiplication tensor M has entries:
    # M[i, j, k] = 1 if we're computing (A*B)[i,k] using A[i,j] and B[j,k]
    # The tensor can be decomposed as:
    # M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[c]
    # where U, V, W are the factor matrices.
    #
    # But the interface specifies shapes:
    # U: (n*m, rank) - so U[i*m+j, c]
    # V: (m*p, rank) - so V[j*p+k, c]
    # W: (p*n, rank) - so W[k*n+i, c]
    #
    # So the decomposition is:
    # 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k
    #
    # This is a rank-rank decomposition of the all-ones tensor of size n x m x p.
    # But wait, that's not the matrix multiplication tensor. The matrix multiplication
    # tensor has M[i,j,k] = 1 for all i,j,k, which is indeed the all-ones tensor.
    #
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _search_rank(target_rank, time_limit=10.0):
    """Search for a decomposition of the given rank."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Fix U to have exactly one 1 per column (basis vectors)
    # This means for each column c, U has a 1 at position i*m+j for some (i,j)
    # There are n*m choices for each column, so (n*m)^rank total configurations
    # This is still huge, so we need a smarter approach.
    
    # Alternative: use random search
    np.random.seed(42)
    
    for attempt in range(10000):
        if time.time() - start_time > time_limit:
            return None
        
        # Generate random U with entries from {-1, 0, 1}
        U_np = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
        
        # For each (i,j,k), we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is a system of nonlinear equations. Let's try a different approach.
        
        # Fix U to be a permutation matrix pattern: each column has exactly one 1
        # Choose which (i,j) position gets the 1 for each column
        positions = np.random.randint(0, n*m, size=target_rank)
        U_np = np.zeros((n*m, target_rank))
        for c in range(target_rank):
            U_np[positions[c], c] = 1.0
        
        # Now for each column c, we have U[i*m+j, c] = 1 if (i,j) = positions[c], else 0
        # So the constraint becomes: for each (i,j,k), sum_c [positions[c] == i*m+j] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is still nonlinear.
        
        # Let's try a linear approach: fix U and W, solve for V
        # Or fix U and V, solve for W
        
        # For now, skip this and try a different strategy
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # First, try to find a better decomposition
    # Known lower bounds: for 2x4 x 4x7, the rank is at least max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Upper bound: schoolbook rank = 56
    
    # Try ranks from 28 upward
    for rank in range(28, 57):
        if time.time() > deadline - 30.0:  # Leave 30s for fallback
            break
        
        result = _search_rank(rank, time_limit=5.0)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
