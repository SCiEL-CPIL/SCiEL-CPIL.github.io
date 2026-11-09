"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    # Use block decomposition: B is 3x6 = [B1 | B2], each 3x3
    # C = [A*B1 | A*B2], each 2x3
    # Use rank-9 decomposition for 2x3 x 3x3 -> 2x3
    # Total rank = 9 + 9 = 18
    
    # Rank-9 decomposition for 2x3 x 3x3:
    # This uses the structure that the 2x3x3x3 tensor has rank 9
    # We'll construct it explicitly
    
    def rank9_2x3x3x3():
        """Return (U, V, W) for 2x3 x 3x3 -> 2x3 with rank 9."""
        # For 2x3 x 3x3 -> 2x3:
        # U: 6x9, V: 9x9, W: 6x9
        # Indices: U[i*3+j, c], V[j*3+k, c], W[k*2+i, c]
        U = [[0]*9 for _ in range(6)]
        V = [[0]*9 for _ in range(9)]
        W = [[0]*9 for _ in range(6)]
        
        # Known rank-9 decomposition:
        # We use the fact that 2x3x3x3 can be seen as combining
        # specific rank-1 terms. Here's a constructive approach:
        
        # For each output position (i, k) where i in {0,1}, k in {0,1,2},
        # we need C[i,k] = sum_j A[i,j]*B[j,k]
        # 
        # Rank-9 decomposition exists. Let me use a known one:
        # The decomposition uses 9 rank-1 tensors.
        
        # Actually, let me use a simpler approach: just use schoolbook for now
        # and verify it works, then optimize.
        # Schoolbook for 2x3x3x3 has rank 18, not 9.
        
        # Let me use a different strategy: the rank of 2x3x3x3 is actually 18 (schoolbook)
        # unless there's a known better decomposition.
        
        # For safety, let me just use schoolbook for the full 2x3x3x6 which gives rank 36.
        # But that's the baseline. Let me try to do better.
        
        # Alternative: use the fact that we can share some computations.
        # Let me just implement schoolbook for now and return rank 36.
        pass
    
    # Let me just use the schoolbook decomposition with rank 36 for correctness,
    # but structure it in a way that's easy to optimize.
    
    rank = n * m * p  # 36
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
