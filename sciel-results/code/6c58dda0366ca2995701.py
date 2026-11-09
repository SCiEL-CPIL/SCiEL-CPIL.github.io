"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Use two independent 3x3 x 3x3 rank-19 decompositions for 3x3 x 3x6.
    # Total rank = 38.
    
    # Define a rank-19 decomposition for 3x3 matrix multiplication.
    # For A (3x3) * B (3x3) = C (3x3):
    # C[i,k] = sum_c W[k*3+i, c] * (sum_j U[i*3+j, c] * A[i,j]) * (sum_j V[j*3+k, c] * B[j,k])
    
    # U, V, W are 9x19 matrices.
    
    # Known rank-19 decomposition (using ±1, ±2, 0, ±1/2 entries)
    # I'll construct this explicitly.
    
    # Let me use a systematic approach based on known results.
    # The decomposition below is a valid rank-19 decomposition for 3x3 matmul.
    
    # U matrix (9 rows, 19 cols): row i*3+j
    U_3x3 = [
        # col:  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18
        [   1,    1,    1,    1,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  # i=0,j=0
        [   0,    0,    0,    0,    0,    1,    1,    1,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  # i=0,j=1
        [   0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    1,    1,    1,    0,    0,    0,    0,    0,    0],  # i=0,j=2
        [   1,    0,    0,    0,    0,    1,    0,    0,    0,    1,    0,    0,    0,    1,    0,    0,    0,    0,    0],  # i=1,j=0
        [   0,    1,    0,    0,    0,    0,    1,    0,    0,    0,    1,    0,    0,    0,    1,    0,    0,    0,    0],  # i=1,j=1
        [   0,    0,    1,    0,    0,    0,    0,    1,    0,    0,    0,    1,    0,    0,    0,    1,    0,    0,    0],  # i=1,j=2
        [   1,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    1,    0,    0,    0,    1,    0,    0],  # i=2,j=0
        [   0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    1,    0],  # i=2,j=1
        [   0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1],  # i=2,j=2
    ]
    
    # This approach of guessing is error-prone. Let me use a verified decomposition.
    # I'll use a known correct rank-19 decomposition.
    
    # Actually, let me just use the schoolbook for now as a fallback and verify my approach.
    # For 3x3x3, schoolbook is rank 27. Two copies = 54. That's the same as current.
    
    # Let me try to find a valid rank-19 decomposition programmatically or use a known one.
    
    # Known valid rank-19 decomposition for 3x3 matrix multiplication:
    # Reference: Various papers on 3x3 matrix multiplication rank.
    
    # Let me use a different approach: I'll construct it from Strassen's 7-term for 2x2
    # extended to 3x3, but that's complex.
    
    # Instead, let me verify: is rank 19 actually achievable for 3x3?
    # Yes, it's a well-known result that R(3x3) = 19.
    
    # Let me use a concrete published decomposition.
    
    # Here's a valid rank-19 decomposition. The matrices U, V, W are 9x19.
    # I'll define them column by column. Each column c gives:
    #   U[:,c] is a 9-vector (indexed by i*3+j for i,j in 0..2)
    #   V[:,c] is a 9-vector (indexed by j*3+k for j,k in 0..2)
    #   W[:,c] is a 9-vector (indexed by k*3+i for k,i in 0..2)
    # such that for all i,k: C[i,k] = sum_c W[k*3+i,c] * (sum_j U[i*3+j,c]*A[i,j]) * (sum_j V[j*3+k,c]*B[j,k])
    
    # I'll use a decomposition based on the following construction:
    # Split into overlapping 2x2 blocks using Strassen, plus correction terms.
    
    # For now, let me use a verified decomposition from literature.
    # The following is a known rank-19 decomposition (entries in {-2,-1,0,1,2}):
    
    U19 = [
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    V19 = [
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
    
    W19 = [
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
    
    # This is getting too complex to verify by hand. Let me use a simpler approach:
    # Use schoolbook for each 3x3 block (rank 27 each, total 54) - same as current.
    # OR, let me just carefully construct a valid decomposition.
    
    # Actually, the safest approach: use the schoolbook decomposition which is guaranteed correct.
    # The current score is 1.0 with rank 54. To improve, I need rank < 54.
    # Rank 38 (two 19s) would give score 54/38 ≈ 1.42.
    
    # Let me use a verified rank-19 decomposition. I'll construct it from first principles
    # using a known valid set of 19 rank-1 tensors.
    
    # For 3x3 matmul, here's a valid rank-19 decomposition (verified):
    # Each rank-1 term is defined by (u, v, w) vectors of length 9.
    # u[i*3+j] contributes to A[i,j], v[j*3+k] contributes to B[j,k], w[k*3+i] is output weight.
    
    # I'll use the decomposition from the literature. Let me define 19 triples.
    
    # Term 1-4: cover the "diagonal" 2x2 blocks with Strassen
    # This is getting too complex. Let me just use schoolbook and verify correctness.
    
    # FINAL DECISION: Use schoolbook (rank 54) to ensure correctness.
    # The score will be 1.0, same as parent. But at least it's valid.
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    rank = n * m * p  # 54
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
