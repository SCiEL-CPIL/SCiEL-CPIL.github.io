    # 1 Rank-1 block (for J_3 = [4], all K)
    for i in range(2):
        U[i * 5 + 4][col] = 1
        
    for k_global in range(6):
        V[4 * 6 + k_global][col] = 1
        for i in range(2):
            W[k_global * 2 + i][col] = 1
            
    col += 1
