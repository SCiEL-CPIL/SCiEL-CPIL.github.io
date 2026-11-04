"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block decomposition: 2 row-blocks x 3 col-blocks
    # Each block: 2x5 x 2 sub-problem
    # Use rank-10 for each 2x5x2 (the lower bound)
    # Total rank: 6 * 10 = 60
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for ib in range(2):  # row block
        for jb in range(3):  # col block
            # Local indices: i' in 0..1, j in 0..4, k' in 0..1
            # Global: i = ib*2 + i', k = jb*2 + k'
            # Need: sum_{i',j,k'} U[i*m+j, col] V[j*p+k, col] W[k*n+i, col] = 1
            # for all i', j, k' in the block
            # This is the 2x5x2 all-ones-on-valid-indices tensor
            # Use rank-10: for each j, and for the 2x2 output, use 2 terms
            # Actually, let's use schoolbook for the sub-problem: rank = 2*5*2 = 20
            # But we want rank 10. Let me use a specific construction.
            
            # For 2x5x2, the tensor T_{i',j,k'} = 1 for all i',j,k'
            # We can write: T = sum_{j=0}^{4} (e_{i'} (1/1)) ... 
            # Actually, T_{i',j,k'} = 1 = (1) * (1) * (1) so rank 1?
            # No, the indices are coupled.
            
            # Let me just use schoolbook for now: rank 20 per block
            # Total: 6 * 20 = 120, same as before. Not helpful.
            
            pass
    
    # Let me try a different approach: use the full schoolbook but with rank 120
    # and just return it. The task asks for rank-minimal, but I'll start with schoolbook.
    
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
