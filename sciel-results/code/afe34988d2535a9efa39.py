"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank-71 decomposition: 4 disjoint 2x2 Strassen blocks + schoolbook remainder.
    # Blocks: rows {0,1}xcols {0,1}, rows {0,1}xcols {2,3}, rows {2,3}xcols {0,1}, rows {2,3}xcols {2,3}
    # Each 2x2 block uses rank-7 Strassen instead of rank-4 schoolbook, saving 1 rank per block.
    # Total: 75 - 4 = 71.
    n, m, p = DIMENSIONS
    
    rank = 71
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    # For blocks with local indices (a,b) in A, (b,c) in B, (c,a) in C:
    # S1 = A00*B00          -> C00, C11
    # S2 = (A00+A11)*B00    -> C11
    # S3 = A11*(B10-B00)    -> C10
    # S4 = A00*(B11-B01)    -> C01
    # S5 = (A10+A00)*B01    -> C01
    # S6 = (A11-A10)*(B00+B10) -> C10
    # S7 = (A01-A11)*(B11+B01) -> C01
    
    # Define the 4 blocks: (row_indices, col_indices)
    blocks = [
        ([0, 1], [0, 1]),
        ([0, 1], [2, 3]),
        ([2, 3], [0, 1]),
        ([2, 3], [2, 3]),
    ]
    
    for (r0, r1), (c0, c1) in blocks:
        # S1: A[r0,c0] * B[c0,c0]
        U[r0 * m + c0][col] = 1.0
        V[c0 * p + c0][col] = 1.0
        W[c0 * n + r0][col] = 1.0  # C00
        W[c1 * n + r1][col] = 1.0  # C11
        col += 1
        
        # S2: (A[r0,c0]+A[r1,c1]) * B[c0,c0]
        U[r0 * m + c0][col] = 1.0
        U[r1 * m + c1][col] = 1.0
        V[c0 * p + c0][col] = 1.0
        W[c1 * n + r1][col] = 1.0  # C11
        col += 1
        
        # S3: A[r1,c1] * (B[c1,c0] - B[c0,c0])
        U[r1 * m + c1][col] = 1.0
        V[c1 * p + c0][col] = 1.0
        V[c0 * p + c0][col] = -1.0
        W[c0 * n + r1][col] = 1.0  # C10
        col += 1
        
        # S4: A[r0,c0] * (B[c0,c1] - B[c0,c0])
        U[r0 * m + c0][col] = 1.0
        V[c0 * p + c1][col] = 1.0
        V[c0 * p + c0][col] = -1.0
        W[c1 * n + r0][col] = 1.0  # C01
        col += 1
        
        # S5: (A[r1,c0]+A[r0,c0]) * B[c0,c1]
        U[r1 * m + c0][col] = 1.0
        U[r0 * m + c0][col] = 1.0
        V[c0 * p + c1][col] = 1.0
        W[c1 * n + r0][col] = 1.0  # C01
        col += 1
        
        # S6: (A[r1,c1]-A[r1,c0]) * (B[c0,c0]+B[c1,c0])
        U[r1 * m + c1][col] = 1.0
        U[r1 * m + c0][col] = -1.0
        V[c0 * p + c0][col] = 1.0
        V[c1 * p + c0][col] = 1.0
        W[c0 * n + r1][col] = 1.0  # C10
        col += 1
        
        # S7: (A[r0,c1]-A[r1,c1]) * (B[c0,c1]+B[c1,c1])
        U[r0 * m + c1][col] = 1.0
        U[r1 * m + c1][col] = -1.0
        V[c0 * p + c1][col] = 1.0
        V[c1 * p + c1][col] = 1.0
        W[c1 * n + r0][col] = 1.0  # C01
        col += 1
    
    # Schoolbook terms for remaining indices:
    # Row indices involved in blocks: {0,1,2,3} for rows, cols {0,1,2,3}
    # Index 4 in m and p is not covered by any block.
    # Also, cross terms between block and non-block indices.
    # 
    # The full schoolbook is sum over all (i,j,k). We've handled:
    # For each block (r0,r1) x (c0,c1): all (i in {r0,r1}, j in {c0,c1}, k in {c0,c1})
    # 
    # Remaining terms are those where NOT (i in {r0,r1} AND j in {c0,c1} AND k in {c0,c1})
    # for all four blocks combined. Let's just enumerate all schoolbook terms and skip
    # those already covered.
    
    # Track which (i,j,k) tuples are covered by the Strassen blocks
    covered = set()
    for (r0, r1), (c0, c1) in blocks:
        for i in (r0, r1):
            for j in (c0, c1):
                for k in (c0, c1):
                    covered.add((i, j, k))
    
    # Add schoolbook for uncovered terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
