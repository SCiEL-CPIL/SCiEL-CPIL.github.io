"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 by 3x5 matrix.
    
    The tensor T has entries T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,5).
    We want to find rank r and matrices U (2*3 x r), V (3*5 x r), W (5*2 x r) such that
    for all i,j,k: T[i,j,k] = sum_c U[i*3+j, c] * V[j*5+k, c] * W[k*2+i, c]
    
    The output factor follows the official trace convention W[k*n+i,r].
    
    We'll try to find a decomposition with rank less than 2*3*5 = 30.
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    n, m, p = 2, 3, 5
    
    # The trivial schoolbook decomposition has rank n*m*p = 30
    # Let's try to find a better one.
    
    # For matrix multiplication, the tensor is the "multiplication tensor" 
    # which for A (n x m) times B (m x p) gives C (n x p).
    # T[i,j,k] = delta_{j, j} (always 1 in our case since we're just doing element-wise)
    # Wait, actually for standard matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k.
    
    # This is actually a very simple tensor! It's just the all-ones tensor of shape (2,3,5).
    # The rank of the all-ones tensor of shape (a,b,c) is min(a,b,c) if we're talking about 
    # standard tensor rank... but wait, let me think more carefully.
    
    # Actually, for the matrix multiplication tensor where T[i,j,k] = 1 for all i,j,k,
    # this can be decomposed as:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # One simple decomposition: use rank = min(n*m, m*p, p*n) = min(6, 15, 10) = 6
    # Let's try rank 6 first.
    
    # Actually, let me think about this differently. The tensor T[i,j,k] = 1 for all i,j,k
    # can be written as an outer product of three vectors:
    # T[i,j,k] = (1 for all i) * (1 for all j) * (1 for all k)
    # But that's only if U, V, W are vectors, not matrices.
    
    # For matrix factorization U (n*m x r), V (m*p x r), W (p*n x r):
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # If we set r = 1:
    # U[i*m+j, 0] = a_{i,j}, V[j*p+k, 0] = b_{j,k}, W[k*n+i, 0] = c_{k,i}
    # T[i,j,k] = a_{i,j} * b_{j,k} * c_{k,i}
    # We need a_{i,j} * b_{j,k} * c_{k,i} = 1 for all i,j,k.
    # This is possible! Set a_{i,j} = 1, b_{j,k} = 1, c_{k,i} = 1 for all i,j,k.
    # Then T[i,j,k] = 1 * 1 * 1 = 1. 
    
    # So rank 1 works! Let me verify this is correct.
    
    # Wait, but that seems too good. Let me re-read the problem.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix."
    
    # The standard matrix multiplication tensor for A (n x m) * B (m x p) = C (n x p) is:
    # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # So yes, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # And the decomposition is:
    # C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    # Wait, that's not quite right. Let me think again.
    
    # The standard formulation: given A (n x m) and B (m x p), compute C = A * B.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor decomposition says:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # Hmm, that's getting complicated. Let me use the standard formulation.
    
    # Actually, the standard way to write matrix multiplication as a tensor contraction:
    # The multiplication tensor M has entries M[i,j,k] = 1 for all i,j,k (in the appropriate ranges).
    # Then C[i,k] = sum_{j,c} A[i,j] * M[i,j,k] * B[j,k] * ... 
    
    # I think I'm overcomplicating this. Let me just use the formulation from the seed program.
    
    # From the seed program:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # And the decomposition is: for all i,j,k, T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # In the seed program, T[i,j,k] = 1 for all i,j,k, and they use the schoolbook decomposition
    # where each (i,j,k) triple gets its own column c, with U[i*m+j, c] = V[j*p+k, c] = W[k*n+i, c] = 1.
    
    # So the tensor to decompose is indeed the all-ones tensor of shape (n, m, p) = (2, 3, 5).
    
    # And as I showed above, rank 1 works:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    
    # Let me verify: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
    # = 1 * 1 * 1 = 1 ✓
    
    # So the answer is rank 1!
    
    rank = 1
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][0] = 1
    
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1
    
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
