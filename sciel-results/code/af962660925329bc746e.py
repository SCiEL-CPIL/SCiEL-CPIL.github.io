for b in range(5):  # 5 blocks
    for r in range(7):  # 7 Strassen terms per block
        # U[i][col] = strassen_A[r][i] for i=0..3
        # V[2*b][col] = strassen_B[r][0]
        # V[2*b+1][col] = strassen_B[r][1]
        # V[10+2*b][col] = strassen_B[r][2]
        # V[10+2*b+1][col] = strassen_B[r][3]
        # W[4*b][col] = w_contrib[r][0]
        # W[4*b+2][col] = w_contrib[r][1]
        # W[4*b+1][col] = w_contrib[r][2]
        # W[4*b+3][col] = w_contrib[r][3]
        col += 1
