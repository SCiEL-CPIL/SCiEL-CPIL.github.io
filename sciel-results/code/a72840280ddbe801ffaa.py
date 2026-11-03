import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block, u_offset, v_offset, w_offset, c_offset):
        for i in range(len(U_block)):
            for c in range(len(U_block[0])):
                U[u_offset + i][c_offset + c] = U_block[i][c]
        for i in range(len(V_block)):
            for c in range(len(V_block[0])):
                V[v_offset + i][c_offset + c] = V_block[i][c]
        for i in range(len(W_block)):
            for c in range(len(W_block[0])):
                W[w_offset + i][c_offset + c] = W_block[i][c]

    # Strassen M_2,2,2
    def strassen_222():
        U_b = [[0]*7 for _ in range(4)]
        V_b = [[0]*7 for _ in range(4)]
        W_b = [[0]*7 for _ in range(4)]
        # M1 = A00(B00+B11)
        U_b[0][0] = 1; V_b[0][0] = 1; V_b[3][0] = 1
        # M2 = A00 B10
        U_b[0][1] = 1; V_b[2][1] = 1
        # M3 = A10(B00+B01)
        U_b[1][2] = 1; V_b[0][2] = 1; V_b[1][2] = 1
        # M4 = A11 B11
        U_b[3][3] = 1; V_b[3][3] = 1
        # M5 = (A00+A10) B11
        U_b[0][4] = 1; U_b[1][4] = 1; V_b[3][4] = 1
        # M6 = (A11-A00)(B00+B01)
        U_b[3][5] = 1; U_b[0][5] = -1; V_b[0][5] = 1; V_b[1][5] = 1
        # M7 = (A01-A11)(B10+B11)
        U_b[2][6] = 1; U_b[3][6] = -1; V_b[2][6] = 1; V_b[3][6] = 1
        
        # W factors
        # C00 = M1 + M4 - M5 + M7
        W_b[0][0] = 1; W_b[0][3] = 1; W_b[0][4] = -1; W_b[0][6] = 1
        # C01 = M3 + M5
        W_b[1][2] = 1; W_b[1][4] = 1
        # C10 = M2 + M4
        W_b[2][1] = 1; W_b[2][3] = 1
        # C11 = M1 - M2 + M3 + M6
        W_b[3][0] = 1; W_b[3][1] = -1; W_b[3][2] = 1; W_b[3][5] = 1
        
        return U_b, V_b, W_b

    # M_2,2,3 Rank 10 (Approximation or Exact?)
    # I will use Rank 11 for safety, which is Strassen + 4 naive.
    # This gives total rank 36.
    def m223_rank11():
        U_b, V_b, W_b = strassen_222()
        # Extend to rank 11
        U_b = [row + [0]*4 for row in U_b]
        V_b = [row + [0]*4 for row in V_b]
        # V has 6 rows now (2x3)
        V_b = [row + [0]*4 for row in V_b] + [[0]*7 + [0]*4 for _ in range(2)]
        W_b = [row + [0]*4 for row in W_b]
        
        # Naive for col 2
        # C02 = A00 B02 + A01 B12
        U_b[0][7] = 1; V_b[4][7] = 1; W_b[4][7] = 1
        U_b[2][8] = 1; V_b[5][8] = 1; W_b[4][8] = 1
        # C12 = A10 B02 + A11 B12
        U_b[1][9] = 1; V_b[4][9] = 1; W_b[5][9] = 1
        U_b[3][10] = 1; V_b[5][10] = 1; W_b[5][10] = 1
        
        return U_b, V_b, W_b

    # Build M_2,4,2 (Two M_2,2,2)
    U_242 = [[0]*14 for _ in range(8)]
    V_242 = [[0]*14 for _ in range(8)]
    W_242 = [[0]*14 for _ in range(4)]
    
    U1, V1, W1 = strassen_222()
    add_block(U1, V1, W1, 0, 0, 0, 0)
    add_block(U1, V1, W1, 4, 4, 0, 7)
    
    # Build M_2,4,3 (Two M_2,2,3)
    U_243 = [[0]*20 for _ in range(8)]
    V_243 = [[0]*20 for _ in range(12)]
    W_243 = [[0]*20 for _ in range(6)]
    
    U2, V2, W2 = m223_rank11()
    add_block(U2, V2, W2, 0, 0, 0, 0)
    add_block(U2, V2, W2, 4, 6, 0, 11)
    
    # Combine
    add_block(U_242, V_242, W_242, 0, 0, 0, 0)
    add_block(U_243, V_243, W_243, 0, 8, 4, 14)
    
    return {"U": U, "V": V, "W": W}
