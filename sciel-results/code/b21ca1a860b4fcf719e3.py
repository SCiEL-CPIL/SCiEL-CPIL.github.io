import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    def get_2x3x3_rank17():
        rank = 17
        U = [[0] * rank for _ in range(6)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(6)]
        
        # Strassen for 2x2x2 block (A[0:2, 0:2], B[0:2, 0:2])
        # U indices: i*3+j
        # V indices: j*3+k
        # W indices: k*2+i
        
        # M1
        U[0][0] = 1; U[4][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2
        U[3][1] = 1; U[4][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # M3
        U[0][2] = 1
        V[3][2] = 1; V[4][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # M4
        U[4][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[4][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # M6
        U[3][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7
        U[1][6] = 1; U[4][6] = -1
        V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        # A2 B21 (2x1 * 1x2)
        # A2: rows 0,1 col 2. U indices 2, 5.
        # B21: row 2 cols 0,1. V indices 6, 7.
        # W: C00, C01, C10, C11. W indices 0, 2, 1, 3.
        
        # T1: A02 B20 -> C00
        U[2][7] = 1
        V[6][7] = 1
        W[0][7] = 1
        
        # T2: A02 B21 -> C01
        U[2][8] = 1
        V[7][8] = 1
        W[2][8] = 1
        
        # T3: A12 B20 -> C10
        U[5][9] = 1
        V[6][9] = 1
        W[1][9] = 1
        
        # T4: A12 B21 -> C11
        U[5][10] = 1
        V[7][10] = 1
        W[3][10] = 1
        
        # A1 B12 (2x2 * 2x1)
        # A1: rows 0,1 cols 0,1. U indices 0,1,3,4.
        # B12: rows 0,1 col 2. V indices 2, 5.
        # W: C02, C12. W indices 4, 5.
        
        # T5: A00 B02 -> C02
        U[0][11] = 1
        V[2][11] = 1
        W[4][11] = 1
        
        # T6: A01 B12 -> C02
        U[1][12] = 1
        V[5][12] = 1
        W[4][12] = 1
        
        # T7: A10 B02 -> C12
        U[3][13] = 1
        V[2][13] = 1
        W[5][13] = 1
        
        # T8: A11 B12 -> C12
        U[4][14] = 1
        V[5][14] = 1
        W[5][14] = 1
        
        # A2 B22 (2x1 * 1x1)
        # A2: U indices 2, 5.
        # B22: V index 8.
        # W: C02, C12. W indices 4, 5.
        
        # T9: A02 B22 -> C02
        U[2][15] = 1
        V[8][15] = 1
        W[4][15] = 1
        
        # T10: A12 B22 -> C12
        U[5][16] = 1
        V[8][16] = 1
        W[5][16] = 1
        
        return U, V, W

    U_block, V_block, W_block = get_2x3x3_rank17()
    
    rank_total = 17 * 3
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    for b in range(3):
        for i in range(n * m):
            U[i][b * 17 : (b + 1) * 17] = U_block[i]
        for j in range(m):
            for k_local in range(3):
                k_global = b * 3 + k_local
                v_idx = j * p + k_global
                v_block_idx = j * 3 + k_local
                V[v_idx][b * 17 : (b + 1) * 17] = V_block[v_block_idx]
        for k_local in range(3):
            for i in range(n):
                k_global = b * 3 + k_local
                w_idx = k_global * n + i
                w_block_idx = k_local * n + i
                W[w_idx][b * 17 : (b + 1) * 17] = W_block[w_block_idx]
                
    return {"U": U, "V": V, "W": W}
