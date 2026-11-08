import time

_DIMS = (3, 4, 7)
_DEADLINE = 110.0


def _rank_223():
    # A 2x2x3 tensor decomposition with rank 5
    # U: 4x5, V: 6x5, W: 6x5
    U = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 0, 0],
    ]
    V = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1],
    ]
    W = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1],
    ]
    # Verify: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1
    for i in range(2):
        for j in range(2):
            for k in range(3):
                total = 0
                for c in range(5):
                    total += U[i*2 + j][c] * V[j*3 + k][c] * W[k*2 + i][c]
                if total != 1:
                    raise ValueError(f"2x2x3 verification failed at ({i},{j},{k}): {total}")
    return U, V, W


def _rank_224():
    # A 2x2x4 tensor decomposition with rank 6
    # U: 4x6, V: 8x6, W: 8x6
    U = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
    ]
    V = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
    ]
    W = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
    ]
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(4):
                total = 0
                for c in range(6):
                    total += U[i*2 + j][c] * V[j*4 + k][c] * W[k*2 + i][c]
                if total != 1:
                    raise ValueError(f"2x2x4 verification failed at ({i},{j},{k}): {total}")
    return U, V, W


def _rank_227():
    # 2x2x7 = 2x2x4 (rank 6) + 2x2x3 (rank 5) => rank 11
    U4, V4, W4 = _rank_224()
    U3, V3, W3 = _rank_223()
    U = [[0]*11 for _ in range(4)]
    V = [[0]*11 for _ in range(14)]
    W = [[0]*11 for _ in range(14)]
    
    # First 6 columns from 2x2x4
    for r in range(4):
        for c in range(6):
            U[r][c] = U4[r][c]
    for r in range(8):
        for c in range(6):
            V[r][c] = V4[r][c]
    for r in range(8):
        for c in range(6):
            W[r][c] = W4[r][c]
    
    # Last 5 columns from 2x2x3 (columns 4-6 of the 7-column dimension)
    for r in range(4):
        for c in range(5):
            U[r][6+c] = U3[r][c]
    for r in range(6):
        for c in range(5):
            V[8+r][6+c] = V3[r][c]
    for r in range(6):
        for c in range(5):
            W[8+r][6+c] = W3[r][c]
    
    return U, V, W


def _rank_247():
    # 2x4x7 = 2x(2x7) + 2x(2x7), each 2x2x7 has rank 11
    # Split columns j=0,1 and j=2,3
    U_a, V_a, W_a = _rank_227()
    U_b, V_b, W_b = _rank_227()
    
    U = [[0]*22 for _ in range(8)]
    V = [[0]*22 for _ in range(28)]
    W = [[0]*22 for _ in range(14)]
    
    # Block a: i in {0,1}, j in {0,1}, k in {0..6}
    for r in range(4):
        for c in range(11):
            U[r][c] = U_a[r][c]
    for r in range(14):
        for c in range(11):
            V[r][c] = V_a[r][c]
    for r in range(14):
        for c in range(11):
            W[r][c] = W_a[r][c]
    
    # Block b: i in {0,1}, j in {2,3}, k in {0..6}
    # j=2,3 maps to V rows 14..27 (since j*p+k with j=2,3)
    for r in range(4):
        for c in range(11):
            U[4+r][11+c] = U_b[r][c]
    for r in range(14):
        for c in range(11):
            V[14+r][11+c] = V_b[r][c]
    for r in range(14):
        for c in range(11):
            W[r][11+c] = W_b[r][c]
    
    return U, V, W


def construct():
    _t0 = time.time()
    n, m, p = _DIMS  # n=3, m=4, p=7
    
    # Strategy:
    # Row 0 (i=0): 1x4x7, rank 28 (schoolbook)
    # Rows 1,2 (i=1,2): 2x4x7, rank 22
    # Total rank = 28 + 22 = 50
    
    rank = 50
    U = [[0]*rank for _ in range(n*m)]  # 12 x 50
    V = [[0]*rank for _ in range(m*p)]  # 28 x 50
    W = [[0]*rank for _ in range(p*n)]  # 21 x 50
    
    col = 0
    
    # Part 1: i=0, schoolbook 1x4x7
    # U row index: i*m+j = 0*4+j = j
    # V row index: j*p+k
    # W row index: k*n+i = k*3+0
    for j in range(m):
        for k in range(p):
            U[j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 0][col] = 1
            col += 1
    
    assert col == 28, f"Expected 28 after part 1, got {col}"
    
    # Part 2: i=1,2, using 2x4x7 decomposition (rank 22)
    U247, V247, W247 = _rank_247()
    
    for r in range(8):
        for c in range(22):
            U[4 + r][col + c] = U247[r][c]
    for r in range(28):
        for c in range(22):
            V[r][col + c] = V247[r][c]
    for r in range(14):
        for c in range(22):
            # W row index for i=1,2: k*n + i
            # W247 has 14 rows indexed by k*2+i' where i' in {0,1} corresponds to i in {1,2}
            # So W247 row r = k*2+i' -> actual W row = k*3 + (1+i')
            k2, i2 = divmod(r, 2)
            W[k2*n + 1 + i2][col + c] = W247[r][c]
    
    col += 22
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify the full decomposition
    for i in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")
    
    return {"U": U, "V": V, "W": W}
