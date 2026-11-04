DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    rank = 65
    
    U = [[0]*rank for _ in range(n*m)]  # 16 x 65
    V = [[0]*rank for _ in range(m*p)]  # 20 x 65
    W = [[0]*rank for _ in range(p*n)]  # 20 x 65
    
    # Strassen 2x2 factors
    U2 = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[0,0,1,1],[1,0,-1,0]]
    V2 = [[1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],[0,0,0,1],[1,1,0,0],[0,1,0,1]]
    W2 = [[1,0,0,1],[0,1,0,-1],[0,0,1,1],[1,1,0,0],[-1,0,1,-1],[0,0,0,1],[1,0,0,0]]
    
    a_comb = [{1:1,4:1},{3:1,4:1},{1:1},{4:1},{1:1,2:1},{3:1,4:1},{1:1,3:-1}]
    b_comb = [{1:1,4:1},{1:1},{2:1,4:-1},{3:1,1:-1},{4:1},{1:1,2:1},{4:1,2:1}]
    c_comb = [{1:1,4:1},{3:1,4:-1},{2:1,4:1},{1:1,3:1},{1:-1,2:1,4:-1},{4:1},{1:1}]
    
    block_rows = {1:0, 2:0, 3:2, 4:2}
    block_cols = {1:0, 2:2, 3:0, 4:2}
    
    for t in range(7):
        for s in range(7):
            r = t*7 + s
            for ai in range(2):
                for aj in range(2):
                    ca = U2[s][ai*2+aj]
                    if ca == 0: continue
                    for b, bc in a_comb[t].items():
                        if bc == 0: continue
                        row = block_rows[b] + ai
                        col = block_cols[b] + aj
                        U[row*m+col][r] += ca * bc
            for bi in range(2):
                for bj in range(2):
                    cb = V2[s][bi*2+bj]
                    if cb == 0: continue
                    for b, bc in b_comb[t].items():
                        if bc == 0: continue
                        row = block_rows[b] + bi
                        col = block_cols[b] + bj
                        V[row*p+col][r] += cb * bc
            for ci in range(2):
                for cj in range(2):
                    cc = W2[s][ci*2+cj]
                    if cc == 0: continue
                    for cb2, cc2 in c_comb[t].items():
                        if cc2 == 0: continue
                        row = block_rows[cb2] + ci
                        col = block_cols[cb2] + cj
                        W[col*n+row][r] += cc * cc2
    
    # Schoolbook for 5th column
    col5 = 4
    r = 49
    for i in range(n):
        for j in range(m):
            U[i*m+j][r] = 1
            V[j*p+col5][r] = 1
            W[col5*n+i][r] = 1
            r += 1
    
    return {"U": U, "V": V, "W": W}
