    #   
    #   This is exactly what I analyzed.
    #   
    #   So rank 1 should work.
    #   
    #   Let's test it.
    #   
    #   If rank 1 works, the score would be 96/1 = 96.
    #   
    #   But the current score is 1.0, which means the value is 96.
    #   
    #   So either:
    #   1. Rank 1 does not work.
    #   2. The score calculation is different.
    #   
    #   The task says: "score is the rank".
    #   "The evaluator measures the raw value (lower is better)."
    #   "The search score is seed/value, so 1.0 equals the seed program".
    #   
    #   Seed program has rank 96.
    #   So score = 96 / rank.
    #   
    #   If rank 1 works, score = 96.
    #   
    #   But the current score is 1.0.
    #   
    #   This implies the current program has rank 96.
    #   
    #   If I return rank 1, the score should be 96.
    #   
    #   Let's try it.

    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    if _validate(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If rank 1 doesn't work, fall back to schoolbook
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
