import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen coefficients for 2x2 multiplication
    # X indices: 0=x00, 1=x01, 2=x10, 3=x11
    # Y indices: 0=y00, 1=y01, 2=y10, 3=y11
    strassen_U = [
        [1, 0, 0, 1],   # m1: x00 + x11
        [0, 0, 1, 1],   # m2: x10 + x11
        [1, 0, 0, 0],   # m3: x00
        [0, 0, 0, 1],   # m4: x11
        [1, 1, 0, 0],   # m5: x00 + x01
        [0, -1, 1, 0],  # m6: x10 - x01
        [0, 1, 0, -1],  # m7: x01 - x11
    ]
    
    strassen_V = [
        [1, 0, 0, 1],   # m1: y00 + y11
        [1, 0, 0, 0],   # m2: y00
        [0, 1, 0, -1],  # m3: y01 - y11
        [0, 0, 1, -1],  # m4: y10 - y00
        [0, 0, 0, 1],   # m5: y11
        [1, 0, 1, 0],   # m6: y00 + y10
        [0, 1, 0, 1],   # m7: y01 + y11
    ]
    
    # Result coefficients: result[r,c] = sum_t coeff[r,c][t] * m_t
    # result[0,0] = m1+m4-m5+m7
    # result[0,1] = m3+m5
    # result[1,0] = m2+m4
    # result[1,1] = m1-m2+m3+m6
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],   # result[0,0]
        [0, 0, 1, 0, 1, 0, 0],    # result[0,1]
        [0, 1, 0, 1, 0, 0, 0],    # result[1,0]
        [1, -1, 1, 0, 0, 1, 0],   # result[1,1]
    ]
    
    def add_strassen_block(M_row_off, M_col_off, N_row_off, N_col_off, out_row_off, out_col_off, base_col):
        for t in range(7):
            col = base_col + t
            # U side
            for xi in range(4):
                i = xi // 2
                j = xi % 2
                U_idx = (M_row_off + i) * 4 + (M_col_off + j)
                U[U_idx][col] += strassen_U[t][xi]
            # V side
            for yj in range(4):
                i = yj // 2
                j = yj % 2
                V_idx = (N_row_off + i) * 4 + (N_col_off + j)
                V[V_idx][col] += strassen_V[t][yj]
            # W side
            for ri in range(4):
                i = ri // 2
                j = ri % 2
                W_idx = (out_col_off + j) * 2 + (out_row_off + i)
                W[W_idx][col] += strassen_W[ri][t]
    
    # Block 1: M1 * N11 → output[:, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    # Block 2: M2 * N21 → output[:, 0:2]
    add_strassen_block(0, 2, 2, 0, 0, 0, 7)
    # Block 3: M1 * N12 → output[:, 2:4]
    add_strassen_block(0, 0, 0, 2, 0, 2, 14)
    # Block 4: M2 * N22 → output[:, 2:4]
    add_strassen_block(0, 2, 2, 2, 0, 2, 21)
    
    return {"U": U, "V": V, "W": W}
