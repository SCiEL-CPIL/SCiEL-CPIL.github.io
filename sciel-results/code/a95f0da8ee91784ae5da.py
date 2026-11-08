def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    R = 18  # 2 blocks * 9 terms each
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    col = 0
    for blk in range(2):
        for j in range(m):
            for i in range(n):
                for k in range(3):
                    U[i*m + j, col] = 1
                    V[j*p + blk*3 + k, col] = 1
                    W[(blk*3 + k)*n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
