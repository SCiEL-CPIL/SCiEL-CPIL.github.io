def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # Kronecker: M_{2,1,1} ⊗ M_{1,3,10}
    rank1 = 2  # rank of M_{2,1,1}
    rank2 = 30  # rank of M_{1,3,10}
    total_rank = rank1 * rank2  # 60
    
    U = [[0] * total_rank for _ in range(n * m)]  # (6, 60)
    V = [[0] * total_rank for _ in range(m * p)]  # (30, 60)
    W = [[0] * total_rank for _ in range(p * n)]  # (20, 60)
    
    for a in range(rank1):  # a = 0, 1 (rank index of M_{2,1,1})
        for b in range(rank2):  # b = 0..29 (rank index of M_{1,3,10})
            col = a * rank2 + b
            # b corresponds to j2 = b // 10, k2 = b % 10 in M_{1,3,10}
            j2 = b // 10
            k2 = b % 10
            
            # U: U[i*3+j, col] = 1 if i==a and j==j2
            # i = a, j = j2
            U[a * m + j2][col] = 1
            
            # V: V[j*10+k, col] = 1 if (j,k) == (j2, k2)
            # j = j2, k = k2
            V[j2 * p + k2][col] = 1
            
            # W: W[k*2+i, col] = 1 if i==a and k==k2
            # k = k2, i = a
            W[k2 * n + a][col] = 1
    
    return {"U": U, "V": V, "W": W}
