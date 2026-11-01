import time

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s margin
    
    n, m, p = DIMENSIONS  # 2, 2, 13
    # Rank: 6 blocks of 2 cols (Strassen, rank 7 each) + 1 block of 1 col (Schoolbook, rank 4)
    rank = 6 * 7 + 4  # 46
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46
    
    # Index Mapping:
    # U (A, n x m): Column-Major. Index = j * n + i.
    #   A[0,0] -> 0
    #   A[1,0] -> 1
    #   A[0,1] -> 2
    #   A[1,1] -> 3
    # V (B, m x p): Column-Major. Index = k * m + j.
    #   B[0,k] -> 2*k
    #   B[1,k] -> 2*k + 1
    # W (C, p x n): Column-Major (as per task). Index = k * n + i.
    #   C[0,k] -> 2*k
    #   C[1,k] -> 2*k + 1
    
    col = 0
    # Process 6 blocks of 2 columns each
    for block in range(6):
        c0 = 2 * block      # first column of block
        c1 = 2 * block + 1  # second column of block
        
        # Indices for V and W
        # V: B[0,c0]=2*c0, B[1,c0]=2*c0+1, B[0,c1]=2*c1, B[1,c1]=2*c1+1
        # W: C[0,c0]=2*c0, C[1,c0]=2*c0+1, C[0,c1]=2*c1, C[1,c1]=2*c1+1
        
        v_00 = 2 * c0
        v_10 = 2 * c0 + 1
        v_01 = 2 * c1
        v_11 = 2 * c1 + 1
        
        w_00 = 2 * c0
        w_10 = 2 * c0 + 1
        w_01 = 2 * c1
        w_11 = 2 * c1 + 1
        
        # U Indices
        u_a = 0  # A[0,0]
        u_c = 1  # A[1,0]
        u_b = 2  # A[0,1]
        u_d = 3  # A[1,1]
        
        # M1: (a+d) * (e+h)
        # e = B[0,c0], h = B[1,c1]
        U[u_a][col] = 1
        U[u_d][col] = 1
        V[v_00][col] = 1
        V[v_11][col] = 1
        W[w_00][col] = 1   # C[0,c0]
        W[w_11][col] = 1   # C[1,c1]
        col += 1
        
        # M2: (c+d) * e
        # e = B[0,c0]
        U[u_c][col] = 1
        U[u_d][col] = 1
        V[v_00][col] = 1
        W[w_10][col] = 1   # C[1,c0]
        col += 1
        
        # M3: a * (f-h)
        # f = B[0,c1], h = B[1,c1]
        U[u_a][col] = 1
        V[v_01][col] = 1
        V[v_11][col] = -1
        W[w_01][col] = 1   # C[0,c1]
        W[w_11][col] = 1   # C[1,c1]
        col += 1
        
        # M4: d * (g-e)
        # g = B[1,c0], e = B[0,c0]
        U[u_d][col] = 1
        V[v_10][col] = 1
        V[v_00][col] = -1
        W[w_00][col] = 1   # C[0,c0]
        W[w_10][col] = 1   # C[1,c0]
        col += 1
        
        # M5: (a+b) * h
        # h = B[1,c1]
        U[u_a][col] = 1
        U[u_b][col] = 1
        V[v_11][col] = 1
        W[w_00][col] = -1  # C[0,c0]
        W[w_01][col] = 1   # C[0,c1]
        col += 1
        
        # M6: (c-a) * (e+f)
        # e = B[0,c0], f = B[0,c1]
        U[u_c][col] = 1
        U[u_a][col] = -1
        V[v_00][col] = 1
        V[v_01][col] = 1
        W[w_11][col] = 1   # C[1,c1]
        col += 1
        
        # M7: (b-d) * (g+h)
        # g = B[1,c0], h = B[1,c1]
        U[u_b][col] = 1
        U[u_d][col] = -1
        V[v_10][col] = 1
        V[v_11][col] = 1
        W[w_00][col] = 1   # C[0,c0]
        col += 1
        
        if time.time() > deadline:
            break
    
    # Last block: column 12
    if col < rank:
        c = 12
        v_0c = 2 * c
        v_1c = 2 * c + 1
        w_0c = 2 * c
        w_1c = 2 * c + 1
        
        # Term 1: a * B[0,c] -> C[0,c]
        U[u_a][col] = 1
        V[v_0c][col] = 1
        W[w_0c][col] = 1
        col += 1
        
        # Term 2: b * B[1,c] -> C[0,c]
        U[u_b][col] = 1
        V[v_1c][col] = 1
        W[w_0c][col] = 1
        col += 1
        
        # Term 3: c * B[0,c] -> C[1,c]
        U[u_c][col] = 1
        V[v_0c][col] = 1
        W[w_1c][col] = 1
        col += 1
        
        # Term 4: d * B[1,c] -> C[1,c]
        U[u_d][col] = 1
        V[v_1c][col] = 1
        W[w_1c][col] = 1
        col += 1

    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}
