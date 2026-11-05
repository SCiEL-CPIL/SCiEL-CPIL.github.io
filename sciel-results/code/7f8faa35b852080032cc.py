"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    Uses a 'row-sharing' decomposition structure.
    The 3x5x5 tensor is treated as a direct sum of three 1x5x5 sub-tensors.
    The rank of each 1x5x5 sub-tensor is 25 (schoolbook).
    Total rank = 3 * 25 = 75.
    
    Note: The directive suggested a rank of 15, which is mathematically impossible
    for this tensor (rank of 1x5x5 is 25). We implement the valid rank-75
    decomposition using the requested structural approach.
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = n * m * p      # 75
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        # Each row i defines an independent 1x5x5 multiplication
        for j in range(m):
            for k in range(p):
                # Indices follow the standard convention:
                # U selects A[i,j]
                # V selects B[j,k]
                # W contributes to C[i,k]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
