"""Matrix multiplication 2x3x8 decomposition."""
DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = 2, 3, 8
    
    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Strassen 2x2x2 coefficients
    strassen_U = [
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ]
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x4(j0, k0, col_offset):
        for half in range(2):
            for t in range(7):
                for i in range(2):
                    for dj in range(2):
                        j = j0 + dj
                        U[i*3 + j][col_offset + t] += strassen_U[t][i*2 + dj]
                for dj in range(2):
                    j = j0 + dj
                    for dk in range(2):
                        k = k0 + half*2 + dk
                        V[j*8 + k][col_offset + t] += strassen_V[t][dj*2 + dk]
                for dk in range(2):
                    k = k0 + half*2 + dk
                    for i in range(2):
                        W[k*2 + i][col_offset + t] += strassen_W[t][dk*2 + i]
        return col_offset + 14
    
    def add_schoolbook_2x1x4(j, k0, col_offset):
        for i in range(2):
            for dk in range(4):
                k = k0 + dk
                U[i*3 + j][col_offset] = 1
                V[j*8 + k][col_offset] = 1
                W[k*2 + i][col_offset] = 1
                col_offset += 1
        return col_offset
    
    col = add_strassen_2x2x4(0, 0, col)
    col = add_strassen_2x2x4(0, 4, col)
    col = add_schoolbook_2x1x4(2, 0, col)
    col = add_schoolbook_2x1x4(2, 4, col)
    
    assert col == rank
    
    if not verify_decomposition(U, V, W, rank):
        # Fall back to schoolbook 48-rank
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
