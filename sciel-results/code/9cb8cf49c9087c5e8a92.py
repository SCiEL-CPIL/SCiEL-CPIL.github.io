DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank for 1xm x mxp is m + p - 1
    # We have n such blocks
    rank_per_block = m + p - 1
    rank = n * rank_per_block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(n):
        base_col = r * rank_per_block
        
        # Part 1: n terms for output column 0
        for i in range(m):
            col = base_col + i
            U[r * m + i][col] = 1
            V[i * p + 0][col] = 1
            W[0 * n + r][col] = 1
            
        # Part 2: p-1 terms for output columns 1..p-1
        for k in range(1, p):
            col = base_col + m + (k - 1)
            
            # u is all ones
            for i in range(m):
                U[r * m + i][col] = 1
                
            # v has 1 at (i, k) and -1 at (i, 0)
            for i in range(m):
                V[i * p + k][col] = 1
                V[i * p + 0][col] = -1
                
            # w is e_k
            W[k * n + r][col] = 1

    return {"U": U, "V": V, "W": W}
