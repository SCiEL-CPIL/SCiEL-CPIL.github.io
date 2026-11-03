import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for 3x3 by 3x6 matrix multiplication.
    
    Strategy:
    The tensor T_{i,j,k} = A_{i,j} * B_{j,k} for i in [0,2], j in [0,2], k in [0,5].
    
    We exploit the structure that the 3x3x6 multiplication can be seen as
    two 3x3x3 multiplications (columns 0-2 and 3-5 of B).
    
    The best known rank for 3x3x3 matrix multiplication is 23 (Bini et al., 1979).
    If we use two independent copies, we get rank 46.
    
    However, we can potentially do better by sharing some structure.
    But for simplicity and correctness, let's use the known Bini et al. rank-23
    decomposition for 3x3x3, applied twice.
    
    Actually, let me recall the Bini et al. (1979) rank-23 algorithm coefficients.
    
    The Bini et al. algorithm for 3x3x3 uses 23 terms. The coefficients are:
    
    Let me define the three factor matrices U, V, W each of size 9x23 (for 3x3x3).
    
    The standard Bini et al. decomposition:
    
    U (rows indexed by (i,j) for A_{i,j}):
    V (rows indexed by (j,k) for B_{j,k}):
    W (rows indexed by (k,i) for C_{k,i}):
    
    Here's the Bini et al. (1979) rank-23 decomposition. I'll write it out.
    
    The 23 terms are defined by triples (u_t, v_t, w_t) where each is a 9-element vector.
    
    Let me use the known coefficients. The Bini et al. algorithm:
    
    Term 0: U=[1,0,0,0,0,0,0,0,0], V=[1,0,0,0,0,0,0,0,0], W=[1,0,0,0,0,0,0,0,0]
    Term 1: U=[0,1,0,0,0,0,0,0,0], V=[0,0,1,0,0,0,0,0,0], W=[0,0,0,1,0,0,0,0,0]
    Term 2: U=[0,0,1,0,0,0,0,0,0], V=[0,0,0,0,1,0,0,0,0], W=[0,0,0,0,0,1,0,0,0]
    Term 3: U=[0,0,0,1,0,0,0,0,0], V=[0,0,0,0,0,1,0,0,0], W=[0,0,0,0,0,0,1,0,0]
    Term 4: U=[0,0,0,0,1,0,0,0,0], V=[0,0,0,0,0,0,1,0,0], W=[0,0,0,0,0,0,0,1,0]
    Term 5: U=[0,0,0,0,0,1,0,0,0], V=[0,0,0,0,0,0,0,1,0], W=[0,0,0,0,0,0,0,0,1]
    Term 6: U=[0,0,0,0,0,0,1,0,0], V=[0,0,0,0,0,0,0,0,1], W=[1,0,0,0,0,0,0,0,0]
    Term 7: U=[0,0,0,0,0,0,0,1,0], V=[1,0,0,0,0,0,0,0,0], W=[0,1,0,0,0,0,0,0,0]
    Term 8: U=[0,0,0,0,0,0,0,0,1], V=[0,1,0,0,0,0,0,0,0], W=[0,0,1,0,0,0,0,0,0]
    Term 9: U=[1,0,0,0,0,0,0,0,0], V=[0,1,0,0,0,0,0,0,0], W=[0,0,0,0,1,0,0,0,0]
    Term 10: U=[0,1,0,0,0,0,0,0,0], V=[0,0,1,0,0,0,0,0,0], W=[0,0,0,0,0,1,0,0,0]
    Term 11: U=[0,0,1,0,0,0,0,0,0], V=[0,0,0,0,1,0,0,0,0], W=[0,0,0,0,0,0,1,0,0]
    Term 12: U=[1,0,0,0,0,0,0,0,0], V=[0,0,0,1,0,0,0,0,0], W=[0,0,0,0,0,0,0,1,0]
    Term 13: U=[0,1,0,0,0,0,0,0,0], V=[0,0,0,0,1,0,0,0,0], W=[0,0,0,0,0,0,0,0,1]
    Term 14: U=[0,0,1,0,0,0,0,0,0], V=[0,0,0,0,0,1,0,0,0], W=[1,0,0,0,0,0,0,0,0]
    Term 15: U=[0,0,0,1,0,0,0,0,0], V=[0,0,0,0,0,0,1,0,0], W=[0,0,1,0,0,0,0,0,0]
    Term 16: U=[0,0,0,0,1,0,0,0,0], V=[0,0,0,0,0,0,0,1,0], W=[0,0,0,1,0,0,0,0,0]
    Term 17: U=[0,0,0,0,0,1,0,0,0], V=[0,0,0,0,0,0,0,0,1], W=[0,0,0,0,1,0,0,0,0]
    Term 18: U=[0,0,0,0,0,0,1,0,0], V=[1,0,0,0,0,0,0,0,0], W=[0,0,0,0,0,1,0,0,0]
    Term 19: U=[0,0,0,0,0,0,0,1,0], V=[0,1,0,0,0,0,0,0,0], W=[0,0,0,0,0,0,1,0,0]
    Term 20: U=[0,0,0,0,0,0,0,0,1], V=[0,0,1,0,0,0,0,0,0], W=[0,0,0,0,0,0,0,1,0]
    Term 21: U=[1,1,0,0,0,0,0,0,0], V=[0,0,1,0,0,0,0,0,0], W=[0,0,0,1,1,0,0,0,0]
    Term 22: U=[0,0,1,1,0,0,0,0,0], V=[0,0,0,0,1,0,0,0,0], W=[1,0,0,0,0,1,0,0,0]
    
    Wait, this doesn't look right. Let me use the actual Bini et al. coefficients.
    
    Actually, I don't have the exact coefficients memorized with confidence. Let me use a different approach.
    
    I'll use the schoolbook decomposition which is guaranteed correct. Rank = 3*3*6 = 54.
    """
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
# EVOLVE-BLOCK-END
