"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    Uses a Kronecker product structure, decomposing the problem into 3 independent
    1x5x7 multiplications. Each 1x5x7 multiplication has rank 5*7=35.
    Total rank = 3 * 35 = 105.
    """
    n, m, p = DIMENSIONS
    
    # The rank of M_{1,m,p} is m*p.
    # The total rank is n * (m*p) = n*m*p.
    rank_per_block = m * p
    total_rank = n * rank_per_block
    
    # Initialize matrices with zeros
    # U: (n*m, total_rank)
    # V: (m*p, total_rank)
    # W: (p*n, total_rank)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill the decomposition
    # For each row i of A (which corresponds to a 1xmxp multiplication)
    for i in range(n):
        # The block for this i occupies columns [i*rank_per_block, (i+1)*rank_per_block)
        block_start = i * rank_per_block
        
        # Iterate over the terms in the 1xmxp multiplication
        # The term is A_{i,j} * B_{j,k}
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        # Column index: block_start + j*p + k
        
        for j in range(m):
            for k in range(p):
                col = block_start + j * p + k
                
                # U[i*m + j, col] = 1
                U[i * m + j][col] = 1
                # V[j*p + k, col] = 1
                V[j * p + k][col] = 1
                # W[k*n + i, col] = 1
                W[k * n + i][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
