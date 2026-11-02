def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # Leave 10s safety margin before 150s kill
    
    # Schoolbook: rank = n*m*p = 48
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Verify the schoolbook solution just in case
    if _verify(n, m, p, U_sb, V_sb, W_sb):
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Fallback (should not happen)
    return {"U": U_sb, "V": V_sb, "W": W_sb}
