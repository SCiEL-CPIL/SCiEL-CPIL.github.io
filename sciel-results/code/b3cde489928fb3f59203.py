"""Program for matrix multiplication tensor decomposition 4x4x7."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def construct():
    """
    Construct a rank-28 decomposition for 4x4 * 4x7 matrix multiplication.
    
    Uses the standard schoolbook decomposition:
    For each output position (i, k), we sum over j:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T_{i,j,k} = 1 for the matrix multiplication tensor.
    
    Schoolbook rank = n*m*p = 4*4*7 = 112.
    
    However, we can do better. Let's think about this differently.
    
    The matrix multiplication tensor has a natural structure. For M4,4,7:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The trivial decomposition gives rank 112. But we can use blocking.
    
    Actually, let me reconsider. The standard schoolbook for 4x4 * 4x7 is rank 4*4*7 = 112.
    
    Can we do better? Yes, by using Strassen-like techniques on sub-blocks.
    
    Let's partition the 4x4 matrix A into 2x2 blocks:
    A = [[A00, A01], [A10, A11]]
    
    And B (4x7) into 2x2 blocks where the second dimension is 7:
    B = [[B00, B01], [B10, B11]] where B00, B10 are 2x7 and B01, B11 are 2x7
    
    Wait, B is 4x7, so splitting rows into two groups of 2:
    B = [[B_top], [B_bottom]] where B_top is 2x7 and B_bottom is 2x7
    
    Then C = A*B is 4x7:
    C = [[A00*B_top + A01*B_bottom], [A10*B_top + A11*B_bottom]]
    
    Each entry in C is a dot product. This doesn't directly help with rank.
    
    Let me just use the schoolbook decomposition with rank 112 as a valid baseline.
    The task says score is the rank (lower is better), and we need a valid decomposition.
    
    Actually, I realize the current program has rank 28 which is wrong. Let me verify:
    For 4x4 * 4x7, the output is 4x7, and each output entry C[i,k] = sum_j A[i,j]*B[j,k].
    The schoolbook rank is n*m*p = 4*4*7 = 112.
    
    The current program's "rank 28" decomposition is actually incorrect because it doesn't 
    properly account for all the terms in the matrix multiplication.
    
    Let me construct a correct schoolbook decomposition with rank 112.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Schoolbook decomposition: rank = n * m * p = 112
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1 (factor for A[i,j])
                # V[j*p + k, col] = 1 (factor for B[j,k])
                # W[k*n + i, col] = 1 (factor for C[i,k])
                U[i * m + j, col] = Fraction(1)
                V[j * p + k, col] = Fraction(1)
                W[k * n + i, col] = Fraction(1)
                col += 1
    
    U_list = [[U[i, r] for r in range(rank)] for i in range(n * m)]
    V_list = [[V[i, r] for r in range(rank)] for i in range(m * p)]
    W_list = [[W[i, r] for r in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
