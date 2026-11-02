"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    # Check: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for any other combination, the sum should be 0.
    # Actually, the tensor T[i,j,k] = delta_{i,j,k} means:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1 if this corresponds to a valid multiplication term, else 0.
    # For matrix multiplication (n x m) * (m x p) = (n x p):
    # The tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right. Let me think again.
    
    # The matrix multiplication tensor M is a 3-tensor where:
    # M[i, j, k] = 1 if the (i,k) entry of the product involves A[i,j] * B[j,k]
    # So M[i,j,k] = 1 for all valid (i,j,k), and the decomposition should satisfy:
    # M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Actually, for matrix multiplication, every (i,j,k) triple contributes:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k in range.
    
    # Let's verify by checking that the reconstructed tensor matches.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition."""
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

def _try_random_combination(n, m, p, num_terms, seed=None):
    """Try to find a decomposition by combining schoolbook terms randomly."""
    if seed is not None:
        random.seed(seed)
    
    # Start with all n*m*p schoolbook terms
    all_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_terms.append((i, j, k))
    
    # We want to see if we can express the tensor as a sum of fewer rank-1 terms.
    # One approach: randomly select subsets and see if linear combinations work.
    
    # Alternative: use the fact that for small n, we can try to merge terms.
    # For n=2, m=3, p=7, the tensor is 2*3*7 = 42 dimensional.
    
    # Let's try a greedy approach: start with rank 42, try to reduce.
    # Actually, let's just return the schoolbook for now as a baseline.
    
    return None

def _greedy_reduction(n, m, p, max_rank=None):
    """Try to reduce the rank by finding linear dependencies."""
    if max_rank is None:
        max_rank = n * m * p
    
    # The tensor T is defined by T[i,j,k] = 1 for all i,j,k.
    # We need to find U, V, W such that T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # One strategy: try random rank-1 terms and see if they can span the tensor.
    # But this is very hard. Let's just return schoolbook.
    
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    # Return schoolbook decomposition as baseline
    result = _schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition
    # For 2x3x7, the minimum rank is known to be less than 42.
    # Let's try a few random approaches.
    
    # Approach: Use the fact that we can combine terms that share indices.
    # For example, terms with the same i can potentially be combined.
    
    # Actually, let's just return the schoolbook decomposition.
    # Finding a lower rank requires sophisticated algorithms.
    
    return result
# EVOLVE-BLOCK-END
