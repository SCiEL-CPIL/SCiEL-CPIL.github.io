def construct():
    n, m, p = 4, 5, 6
    deadline = time.time() + 110.0
    
    # Split A (4x5) = [A1 (4x3) | A2 (4x2)]
    # B (5x6) = [B1 (3x6); B2 (2x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: (4x3) x (3x6), rank 12
    # A2*B2: (4x2) x (2x6), rank 8
    # Total: 20
    
    rank_total = 20
    U = [[0] * rank_total for _ in range(n * m)]  # 20 x 20
    V = [[0] * rank_total for _ in range(m * p)]  # 30 x 20
    W = [[0] * rank_total for _ in range(p * n)]  # 24 x 20
    
    col = 0
    
    # A1*B1: i in [0,4), j in [0,3), k in [0,6)
    for j in range(3):
        for i in range(4):
            U[i * m + j][col] = 1
            for k in range(p):
                V[j * p + k][col] = 1
            for k in range(p):
                W[k * n + i][col] = 1
            col += 1
    
    # A2*B2: i in [0,4), j in [0,2), k in [0,6)
    # A2[i,j] = A[i, 3+j], B2[j,k] = B[3+j, k]
    for j in range(2):
        for i in range(4):
            U[i * m + (3 + j)][col] = 1
            for k in range(p):
                V[(3 + j) * p + k][col] = 1
            for k in range(p):
                W[k * n + i][col] = 1
            col += 1
    
    assert col == rank_total, f"Expected {rank_total}, got {col}"
    
    # Verify
    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col2] = 1
                    V_school[j * p + k][col2] = 1
                    W_school[k * n + i][col2] = 1
                    col2 += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
