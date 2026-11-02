<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Schoolbook baseline: rank = n*m*p = 24
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    T_target = _build_tensor(n, m, p)
=======
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Check for Rank 1 solution for the All-Ones tensor
    # T[i,j,k] = 1 can be decomposed as U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1
    rank_1 = 1
    U_1 = [[1.0] * rank_1 for _ in range(n * m)]
    V_1 = [[1.0] * rank_1 for _ in range(m * p)]
    W_1 = [[1.0] * rank_1 for _ in range(p * n)]
    
    if _validate(U_1, V_1, W_1, n, m, p):
        return {"U": U_1, "V": V_1, "W": W_1}

    # Schoolbook baseline: rank = n*m*p = 24
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    T_target = _build_tensor(n, m, p)
>>>>>>> REPLACE
