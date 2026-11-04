import numpy as np

def construct():
    n, m, p = 4, 4, 5
    
    def make_mat(rows, cols):
        return [[0] * cols for _ in range(rows)]

    # Strassen decomposition for 2x2x2
    # Returns U (4x7), V (4x7), W (4x7)
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    def strassen_222():
        U = make_mat(4, 7)
        V = make_mat(4, 7)
        W = make_mat(4, 7)
        
        # P1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # P2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[0][1] = -1; W[2][1] = 1
        
        # P3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = -1
        
        # P4 = A11(B10-B00)
        U[3][3] = 1
        V[0][3] = -1; V[2][3] = 1
        W[0][3] = 1; W[2][3] = 1
        
        # P5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = 1; W[3][4] = 1
        
        # P6 = (A10-A00)(B00+B01)
        U[0][5] = -1; U[2][5] = 1
        V[0][5] = 1; V[1][5] = 1
        W[0][5] = -1
        
        # P7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[3][6] = 1
        
        return U, V, W

    # Build 2x2x5 decomposition (Rank 18)
    # U: 4x18, V: 10x18, W: 10x18
    # U index: i*2+j
    # V index: j*5+k
    # W index: k*2+i
    def build_225():
        U222, V222, W222 = strassen_222()
        rank = 18
        U = make_mat(4, rank)
        V = make_mat(10, rank)
        W = make_mat(10, rank)
        
        # First 4 columns (k=0..3) use Strassen 2x2x2
        # Last 1 column (k=4) uses schoolbook 2x2x1
        for t in range(7):
            for i in range(4):
                U[i][t] = U222[i][t]
                U[i][t+7] = U222[i][t]
            
            for v_idx in range(4):
                val = V222[v_idx][t]
                if val != 0:
                    j = v_idx // 2
                    k = v_idx % 2
                    V[j*5 + k][t] = val
                    V[j*5 + k + 4][t+7] = val
            
            for w_idx in range(4):
                val = W222[w_idx][t]
                if val != 0:
                    k = w_idx // 2
                    i = w_idx % 2
                    W[k*2 + i][t] = val
                    W[(k+2)*2 + i][t+7] = val
        
        # Last 4 terms for k=4 (schoolbook 2x2x1)
        for i in range(2):
            for j in range(2):
                col = 14 + i*2 + j
                u_idx = i*2 + j
                v_idx = j*5 + 4
                w_idx = 4*2 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
        
        return U, V, W

    # Assemble 4x4x5 (Rank 72)
    U225, V225, W225 = build_225()
    rank = 72
    U = make_mat(16, rank)
    V = make_mat(20, rank)
    W = make_mat(20, rank)
    
    # Block 1: A00 * B_top -> C_top
    # A00: rows 0-1, cols 0-1. B_top: rows 0-1. C_top: rows 0-1.
    for t in range(18):
        for i in range(4):
            li = i // 2
            lj = i % 2
            U[li*4 + lj][t] = U225[i][t]
        for v_idx in range(10):
            val = V225[v_idx][t]
            if val != 0:
                j = v_idx // 5
                k = v_idx % 5
                V[j*5 + k][t] = val
        for w_idx in range(10):
            val = W225[w_idx][t]
            if val != 0:
                k = w_idx // 2
                i = w_idx % 2
                W[k*4 + i][t] = val

    # Block 2: A01 * B_bot -> C_top
    # A01: rows 0-1, cols 2-3. B_bot: rows 2-3. C_top: rows 0-1.
    for t in range(18):
        for i in range(4):
            li = i // 2
            lj = i % 2
            U[li*4 + lj + 2][t+18] = U225[i][t]
        for v_idx in range(10):
            val = V225[v_idx][t]
            if val != 0:
                j = v_idx // 5
                k = v_idx % 5
                V[(j+2)*5 + k][t+18] = val
        for w_idx in range(10):
            val = W225[w_idx][t]
            if val != 0:
                k = w_idx // 2
                i = w_idx % 2
                W[k*4 + i][t+18] = val

    # Block 3: A10 * B_top -> C_bot
    # A10: rows 2-3, cols 0-1. B_top: rows 0-1. C_bot: rows 2-3.
    for t in range(18):
        for i in range(4):
            li = i // 2
            lj = i % 2
            U[(li+2)*4 + lj][t+36] = U225[i][t]
        for v_idx in range(10):
            val = V225[v_idx][t]
            if val != 0:
                j = v_idx // 5
                k = v_idx % 5
                V[j*5 + k][t+36] = val
        for w_idx in range(10):
            val = W225[w_idx][t]
            if val != 0:
                k = w_idx // 2
                i = w_idx % 2
                W[k*4 + i + 2][t+36] = val

    # Block 4: A11 * B_bot -> C_bot
    # A11: rows 2-3, cols 2-3. B_bot: rows 2-3. C_bot: rows 2-3.
    for t in range(18):
        for i in range(4):
            li = i // 2
            lj = i % 2
            U[(li+2)*4 + lj + 2][t+54] = U225[i][t]
        for v_idx in range(10):
            val = V225[v_idx][t]
            if val != 0:
                j = v_idx // 5
                k = v_idx % 5
                V[(j+2)*5 + k][t+54] = val
        for w_idx in range(10):
            val = W225[w_idx][t]
            if val != 0:
                k = w_idx // 2
                i = w_idx % 2
                W[k*4 + i + 2][t+54] = val

    return {"U": U, "V": V, "W": W}
