"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    # Use a decomposition with rank less than n*m*p = 112
    # We'll use a rank of 98 by exploiting linear dependencies
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 28 x 98
    W = [[0] * rank for _ in range(p * n)]  # 28 x 98
    
    # Standard schoolbook would use 112 terms
    # We'll try to reduce by sharing some computations
    # For now, use a simple reduction: for the last j=3, share with j=2
    # This is a heuristic; we need to verify correctness
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= rank:
                    break
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if col >= rank:
                    break
            if col >= rank:
                break
        if col >= rank:
            break
    
    # This only fills 98 of the 112 terms, which is NOT a valid decomposition
    # We need a proper decomposition
    
    # Let me try a different approach: use a known decomposition
    # For 4x4 x 4x7, I'll use a decomposition based on grouping
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use a decomposition where we process in blocks
    # For j=0,1: use standard (4*7*2 = 56 terms)
    # For j=2,3: use a shared decomposition (42 terms)
    # Total: 56 + 42 = 98
    
    col = 0
    # j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # j=1
    for i in range(n):
        for k in range(p):
            U[i * m + 1][col] = 1
            V[1 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # For j=2 and j=3, use a combined approach
    # We need to compute sum_{j=2,3} A[i,j]*B[j,k]
    # This is a 4x4 x 2x7 multiplication, which has rank at most 4*2*7=56
    # But we only have 42 slots left, so this won't work directly
    
    # I think I need to accept that a simple improvement is hard
    # Let me just return the schoolbook for now, but with a slightly better structure
    
    # Actually, let me try rank = 105
    rank = 105
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < rank:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                else:
                    break
            if col >= rank:
                break
        if col >= rank:
            break
    
    # This is still not a valid decomposition since we're missing terms
    # The only valid decomposition I can guarantee is the schoolbook
    
    # Let me just return the schoolbook but with rank = 112
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
