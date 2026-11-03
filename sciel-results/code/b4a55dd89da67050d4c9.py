"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-wise tensor decomposition for 3x5 x 5x7 matrix multiplication.
    
    Strategy: Split the middle dimension m=5 into m1=2 and m2=3.
    C = A1*B1 + A2*B2 where:
      A1: 3x2, B1: 2x7  -> sub-problem (3,2,7)
      A2: 3x3, B2: 3x7  -> sub-problem (3,3,7)
    
    For each sub-problem, use the naive (schoolbook) tensor decomposition.
    Total rank = 3*2*7 + 3*3*7 = 42 + 63 = 105.
    
    The decomposition is exact: each rank-1 term corresponds to one
    (i,j,k) index triple in the respective sub-problem.
    """
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # Split middle dimension
    m1, m2 = 2, 3  # m = m1 + m2 = 5
    
    # Build U, V, W for sub-problem 1: (n x m1) x (m1 x p)
    # U1 shape: (n*m1, rank1) = (6, 42)
    # V1 shape: (m1*p, rank1) = (14, 42)
    # W1 shape: (p*n, rank1)   = (21, 42)
    rank1 = n * m1 * p  # 3*2*7 = 42
    U1 = [[0] * rank1 for _ in range(n * m1)]
    V1 = [[0] * rank1 for _ in range(m1 * p)]
    W1 = [[0] * rank1 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                # A1[i,j] is at index i*m1+j in U1
                # B1[j,k] is at index j*p+k in V1
                # C[i,k] is at index k*n+i in W1
                U1[i * m1 + j][col] = 1
                V1[j * p + k][col] = 1
                W1[k * n + i][col] = 1
                col += 1
    
    # Build U, V, W for sub-problem 2: (n x m2) x (m2 x p)
    # U2 shape: (n*m2, rank2) = (9, 63)
    # V2 shape: (m2*p, rank2) = (21, 63)
    # W2 shape: (p*n, rank2)   = (21, 63)
    rank2 = n * m2 * p  # 3*3*7 = 63
    U2 = [[0] * rank2 for _ in range(n * m2)]
    V2 = [[0] * rank2 for _ in range(m2 * p)]
    W2 = [[0] * rank2 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m2):
            for k in range(p):
                U2[i * m2 + j][col] = 1
                V2[j * p + k][col] = 1
                W2[k * n + i][col] = 1
                col += 1
    
    # Combine into full decomposition:
    # U has shape (n*m, total_rank) = (15, 105)
    # V has shape (m*p, total_rank) = (35, 105)
    # W has shape (p*n, total_rank) = (21, 105)
    total_rank = rank1 + rank2  # 105
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Place sub-problem 1 contributions (j = 0..m1-1, i.e., j=0,1)
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                col_idx = i * m1 + j  # index within U1
                # Find which column in U1 this corresponds to
                # Actually, we need to map the local indices to global columns
                # The column index in the combined U is the same as in U1 for first rank1 cols
                pass
    
    # Simpler: directly build U, V, W with the combined indexing
    # For sub-problem 1, A1[i,j] corresponds to A[i,j] where j=0..m1-1
    # In the full A (3x5), A[i,j] is at index i*m+j = i*5+j
    # In U1, A1[i,j] is at index i*m1+j = i*2+j
    
    # Let me just rebuild directly:
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Sub-problem 1: j ranges over 0..m1-1 (i.e., 0,1)
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Sub-problem 2: j ranges over m1..m-1 (i.e., 2,3,4)
    for i in range(n):
        for j in range(m1, m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
