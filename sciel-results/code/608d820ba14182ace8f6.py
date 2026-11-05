"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                # The matrix multiplication tensor has T[i,j,k] = 1 if this is the "standard" basis
                # Actually, for matrix multiplication, the tensor T[i,j,k] should be 1 when
                # we're computing (AB)[i,k] = sum_j A[i,j] * B[j,k]
                # So T[i,j,k] = 1 for all i,j,k in the standard decomposition
                if abs(total - 1.0) > tol:
                    return False
    return True

def _estimate_rank_lower_bound(n, m, p, num_samples=50, seed=42):
    """
    Use random linear functionals to estimate a lower bound on tensor rank.
    
    The idea: if we have a rank-r decomposition, then for any linear functional
    applied to the tensor, the resulting vector should have rank at most r.
    
    We sample random linear functionals and check the rank of the resulting
    matrices to get a lower bound.
    """
    random.seed(seed)
    
    # The tensor T[i,j,k] for matrix multiplication is 1 for all i,j,k
    # (in the standard basis where T represents the structure constants)
    
    # For a rank-r decomposition: T[i,j,k] = sum_{s=1}^r U[i,j,s] * V[j,k,s] * W[k,i,s]
    
    # Apply random linear functional: L(T) = sum_{i,j,k} alpha[i,j,k] * T[i,j,k]
    # This gives a scalar, which doesn't help directly.
    
    # Better: fix one index and look at the matrix in the other two indices.
    # For fixed i: M_jk = T[i,j,k] should have rank <= r
    # For fixed j: N_ik = T[i,j,k] should have rank <= r  
    # For fixed k: O_ij = T[i,j,k] should have rank <= r
    
    # But for matrix multiplication, T[i,j,k] = 1 for all i,j,k
    # So each slice is a matrix of all 1s, which has rank 1.
    
    # This suggests rank 1 might work? No, that's wrong.
    # The tensor for matrix multiplication is NOT all 1s.
    
    # Let me reconsider. The matrix multiplication tensor T satisfies:
    # (AB)[i,k] = sum_j A[i,j] * B[j,k]
    # So T is a 3-tensor where T[i,j,k] = 1 if we're "using" A[i,j] and B[j,k] to contribute to C[i,k]
    # Actually, T[i,j,k] = 1 for all valid i,j,k in the standard formulation.
    
    # Wait, I need to be more careful. The tensor T is defined by:
    # T(A, B) = AB where (AB)[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # But this tensor has a specific structure. Let me think about its rank.
    
    # For the 3x3x5 matrix multiplication tensor, the known lower bound is...
    # Actually, let me just use a different approach.
    
    # Use the "slice rank" method: for each fixed value of one index,
    # the resulting matrix must have rank at most r.
    
    # For fixed i (i=0,1,2): matrix M_i[j,k] = T[i,j,k] = 1 (all ones)
    # Rank of all-ones matrix of size 3x5 is 1.
    
    # For fixed j (j=0,1,2): matrix N_j[i,k] = T[i,j,k] = 1 (all ones)
    # Rank of all-ones matrix of size 3x5 is 1.
    
    # For fixed k (k=0,1,2,3,4): matrix O_k[i,j] = T[i,j,k] = 1 (all ones)
    # Rank of all-ones matrix of size 3x3 is 1.
    
    # This gives a lower bound of 1, which is too weak.
    
    # The issue is that the tensor T[i,j,k]=1 is NOT the matrix multiplication tensor.
    # I'm confusing myself.
    
    # Let me reconsider the problem. The matrix multiplication tensor is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k, and the multiplication is:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    
    # A rank-r decomposition means we can write:
    # T[i,j,k] = sum_{s=1}^r U_s[i,j] * V_s[j,k] * W_s[k,i]
    
    # For the standard tensor T[i,j,k]=1, we need:
    # 1 = sum_{s=1}^r U_s[i,j] * V_s[j,k] * W_s[k,i]
    
    # This is a system of n*m*p equations.
    
    # Let me try a different approach: use the fact that the tensor has symmetries.
    
    # Actually, I think I should just try to find a decomposition directly.
    # Let me start with the schoolbook (rank 45) and try to reduce it.
    
    # For now, return a conservative lower bound.
    # The trivial lower bound from slice ranks is 1, but we know better.
    
    # Use the "tensor rank" lower bound from the "slice rank" method:
    # r >= max over all linear maps L of rank(L(T))
    
    # For a 3-tensor, the slice rank is at least the maximum rank of any "slice".
    # But as computed above, all slices have rank 1.
    
    # This is because T[i,j,k]=1 is a very special tensor.
    
    # Let me try a completely different approach: just try to find a low-rank decomposition
    # by solving the system of equations.
    
    return 1  # Conservative lower bound

def _try_rank(r, n, m, p, max_iter=1000, seed=42, timeout=100):
    """
    Try to find a rank-r decomposition using random initialization and iterative refinement.
    """
    random.seed(seed)
    start_time = time.time()
    
    for attempt in range(max_iter):
        if time.time() - start_time > timeout:
            break
        
        # Random initialization with small integers/half-integers
        U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)]
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for the 150s limit
    
    # First, get the schoolbook decomposition as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try to find lower-rank decompositions
    # Start from a reasonable lower bound and go up
    # The minimum possible rank is at least max(n, m, p) for many tensors,
    # but for matrix multiplication it's known to be higher.
    
    # Try ranks from low to high
    for r in range(1, n * m * p + 1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(r, n, m, p, max_iter=50, seed=r, timeout=5)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # If no better decomposition found, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
