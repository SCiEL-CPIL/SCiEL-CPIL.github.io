"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication tensor decomposition for n x m by m x p.
    
    For n=3, m=4, p=8:
    - Schoolbook rank = n*m*p = 96
    - We attempt to find a lower-rank exact decomposition.
    
    Known results for rectangular matrix multiplication:
    - For 3x4x8, the schoolbook decomposition is rank 96.
    - Strassen-based decompositions typically work best for square or 2x2 blocks.
    - For 3x4x8, we can view this as 3 rows of (4x8) multiplications.
    
    Key insight: A 4x8 matrix multiplication can be decomposed.
    Actually, let's think about this differently.
    
    The tensor T_{n,m,p} has entries T[i,j,k] where i in [n], j in [m], k in [p].
    The multiplication is: C[i,k] = sum_j A[i,j] B[j,k]
    
    For n=3, m=4, p=8:
    We have 3 output rows, each is a 1x4 times 4x8 multiplication.
    
    Actually, let me reconsider. The standard approach for non-square matrices:
    
    One known result: For multiplying an m x m matrix by an m x p matrix (p >= m),
    the rank is at most m * rank(m x m -> m x m) + m * (p-m) ... no, that's not right.
    
    Let's use the fact that T_{3,4,8} can be embedded or related to known tensors.
    
    Actually, for 3x4x8, let's consider:
    - Split the 4 middle dimension into 2+2
    - Split the 8 output columns into 4+4
    
    So we have 3x(2+2) by (2+2)x(4+4)
    = 3x2 by 2x4 (four blocks)
    
    Each 3x2 by 2x4 block: rank? 
    For 3x2 by 2x4: schoolbook is 3*2*4 = 24.
    
    But we can use Strassen-like tricks on the 2x2 subblocks.
    
    Let me think about a 2x2 by 2x4 multiplication.
    A (2x2) * B (2x4) = C (2x4)
    C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    
    This is rank 8 by schoolbook. Can we do better?
    For 2x2x4, the minimum rank is known to be 14? No, that's for 2x2x2 -> 7.
    For 2x2x4, I believe the rank is 14 (two 2x2x2 multiplications sharing structure).
    Actually, 2x2x4: we can do it in 2*7 = 14? Or less?
    
    Hmm, this is getting complex. Let me just implement the schoolbook decomposition
    which is guaranteed correct with rank 96, and see if there's a simple improvement.
    
    Actually, wait. Let me re-read the task. The score is rank, lower is better.
    The seed has rank 96. If I can find rank < 96, I improve.
    
    For 3x4x8, one approach:
    View as 3 copies of 1x4 by 4x8.
    1x4 by 4x8: rank is 32 by schoolbook (4*8).
    Can 1x4 by 4x8 be done in less than 32?
    C[k] = sum_j A[j] B[j,k] for k=0..7
    This is essentially computing 8 dot products of 4-dim vectors with a fixed vector.
    The rank of this tensor (1x4x8) is at most 32.
    Actually, for 1xm by mxp, the rank is m*p because each output element is a product
    of one A entry and one B entry, and they're all independent.
    Wait, no. C[k] = A[0]B[0,k] + A[1]B[1,k] + A[2]B[2,k] + A[3]B[3,k]
    The terms are A[j]*B[j,k]. These are 4*8 = 32 independent products.
    So rank of 1x4x8 is 32.
    
    So 3 copies would be 3*32 = 96. Same as schoolbook.
    
    The question is whether we can share computations across the 3 rows.
    
    Let me just output the schoolbook decomposition. It's correct and rank 96.
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
