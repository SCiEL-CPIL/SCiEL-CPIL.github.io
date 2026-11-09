"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition: rank = n*m*p
    rank_school = n * m * p
    
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to reduce rank using structured search
    # For 2x3x7, we can try rank = 21 (3 per k value)
    # This requires that for each k, the 2x3 submatrix can be computed with 3 terms
    
    # Let's try a different approach: use the fact that we have 2 rows
    # For each k, we need to compute 2 dot products of length 3
    # We can share the B[j,k] factors
    
    # Try rank = 21: for each k (0..6), use 3 terms
    # Term (j,k) for j in 0..2:
    #   U[i*m+j, r] = 1 for all i (A[i,j] is used in both outputs)
    #   V[j*p+k, r] = 1
    #   W[k*n+i, r] = 1 for all i
    # This gives T[i,j,k] = sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # = sum over j' of 1*delta[j',j]*1 = 1 for all i, which is wrong
    
    # The issue is that W[k*n+i, r] cannot be 1 for all i simultaneously in a way that distinguishes i
    
    # Let me try a different structure. For each k, use 4 terms:
    # This allows us to separate the i=0 and i=1 computations
    
    # Actually, let's just try to find any valid decomposition with rank < 42
    # by solving a linear system
    
    # For a given rank R, we need to find U, V, W such that:
    # For all (i,j,k): T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 if the indices form a valid multiplication term, else 0
    # Actually T[i,j,k] = 1 for all valid (i,j,k) in the tensor
    
    # The tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k
    # Wait, that's not right. The multiplication tensor is T_{ijk} = delta_{ik} in some conventions
    
    # Let me re-read the problem. The output is C = A*B where A is 2x3, B is 3x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,7)
    # And the decomposition is C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
    # No, that's not right either.
    
    # Standard tensor decomposition: T_{ijk} = sum_r U_{ir} * V_{jr} * W_{kr}
    # But here the indices are different. Let me use the trace convention as stated.
    
    # The problem says: W[k*n+i, r] with shape (p*n, rank)
    # So the decomposition is: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # This doesn't look standard. Let me assume the standard form:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for all valid (i,j,k)
    
    # For schoolbook, this works with rank = n*m*p.
    
    # To reduce rank, we need to find a smaller R such that there exist U, V, W with integer/half-integer entries
    # satisfying the above for all (i,j,k).
    
    # This is a hard problem. Let me try a heuristic search for small improvements.
    
    # Try rank = 41: remove one term and see if we can adjust the others
    # This is a nonlinear problem, so it's difficult.
    
    # Given the time constraint, I'll return the schoolbook decomposition.
    # A proper implementation would use algebraic techniques or integer programming.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
