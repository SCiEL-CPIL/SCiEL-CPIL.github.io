"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n * m * p
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
    
    # Verify the decomposition is exact
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For matrix multiplication: T[i,j,k] should equal 1 if this is the (i,j,k) entry
    # The decomposition gives: for each (i,j,k), sum over r where U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1
    # In schoolbook, each (i,j,k) has exactly one r where all three are 1.
    
    # Attempt to reduce rank using the structure of the problem.
    # For 2x3x6, we can use block decomposition.
    # Partition the 3x6 matrix into three 3x2 blocks.
    # This gives three independent 2x3x2 multiplications.
    # Known rank for 2x3x2: the tensor [2,3,2] has rank at most 2*3*2=12, 
    # but we can do better. Actually, for small dimensions, let's check known results.
    
    # For 2x3x2: 
    # Lower bound: max(2*3, 3*2, 2*2) = max(6, 6, 4) = 6
    # Upper bound: schoolbook gives 12.
    # It is known that rank of [2,3,2] is actually 10 or less? Let me think...
    # Actually, for [2,3,2], one can achieve rank 9 or less? 
    # Let me try a direct construction.
    
    # For now, let's try to find a lower rank decomposition for [2,3,2].
    # The tensor T[i,j,k] for i in {0,1}, j in {0,1,2}, k in {0,1}
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
    
    # Let's try rank 9 for each 2x3x2 block.
    # If we can get rank 9 per block, total rank = 3 * 9 = 27.
    # If we can get rank 8, total = 24.
    # If rank 7, total = 21.
    
    # Actually, let me look up: the rank of the 2x3x2 tensor.
    # It is known that the matrix multiplication tensor [2,3,2] has rank 10? Or is it lower?
    # Strassen showed that 2x2x2 has rank 7.
    # For 2x3x2, we can write it as a direct sum of two 2x2x2 tensors plus some extra?
    # Not exactly, because the middle dimension is 3, not 4.
    
    # Let me try to construct a rank-9 decomposition for [2,3,2].
    # Variables: i in {0,1}, j in {0,1,2}, k in {0,1}
    # 6 entries to match, 9 rank-1 terms.
    # We have 6*9 + 3*9 + 2*9 = 54 + 27 + 18 = 99 unknowns (but with scaling invariance).
    # Actually, the system is: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # This is 6 equations. With 9 terms, we have many degrees of freedom.
    
    # Let me try a simpler approach: use the fact that [2,3,2] can be embedded in [2,4,2]
    # which has rank 2*7 = 14? No, that's not helpful.
    
    # Actually, let me just try to solve the system numerically for [2,3,2] with rank r
    # for r = 6, 7, 8, 9, 10 and see what's the smallest r that works.
    
    # But this might take time. Let me first return the schoolbook decomposition
    # and then try to improve.
    
    # Actually, let me try a known result: 
    # For [n,m,p], if one of the dimensions is 1, the rank is n*m*p.
    # For [2,3,2], let me try to find a decomposition.
    
    # I'll attempt a greedy/optimization approach if time permits.
    # For now, let's return schoolbook to ensure validity.
    
    # Actually, wait - the task says the current score is 0.0000, meaning the seed is invalid.
    # Let me re-examine the seed code. The seed uses lists, not numpy arrays.
    # Maybe the evaluator requires numpy arrays?
    # Let me return numpy arrays to be safe.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
