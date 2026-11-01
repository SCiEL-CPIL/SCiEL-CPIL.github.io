import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 48 using block decomposition
    # Split m=5 into 2, 2, 1
    # C = A[:, 0:2]B[0:2, :] + A[:, 2:4]B[2:4, :] + A[:, 4:5]B[4:5, :]
    
    rank1 = 19 # 2x2x5 approx rank
    rank2 = 19
    rank3 = 10 # 2x1x5 rank
    total_rank = rank1 + rank2 + rank3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_222(u_offset, v_offset, w_offset, i_base, j_base, k_base):
        # Strassen for 2x2x2
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # U indices: i*m+j
        # V indices: j*p+k
        # W indices: k*n+i
        
        # M1
        U[i_base*m + 0][col_offset + 0] += 1
        U[i_base*m + 1][col_offset + 0] += 1
        V[j_base*p + 0][col_offset + 0] += 1
        V[j_base*p + 1][col_offset + 0] += 1
        W[k_base*n + 0][col_offset + 0] += 1
        W[k_base*n + 1][col_offset + 0] += 1
        
        # M2
        U[i_base*m + 1][col_offset + 1] += 1
        V[j_base*p + 0][col_offset + 1] += 1
        W[k_base*n + 0][col_offset + 1] += 1
        
        # M3
        U[i_base*m + 0][col_offset + 2] += 1
        V[j_base*p + 1][col_offset + 2] += 1
        W[k_base*n + 1][col_offset + 2] += 1
        
        # M4
        U[i_base*m + 1][col_offset + 3] += 1
        V[j_base*p + 0][col_offset + 3] += 1
        W[k_base*n + 0][col_offset + 3] += 1
        
        # M5
        U[i_base*m + 0][col_offset + 4] += 1
        V[j_base*p + 1][col_offset + 4] += 1
        W[k_base*n + 1][col_offset + 4] += 1
        
        # M6
        U[i_base*m + 0][col_offset + 5] += 1
        U[i_base*m + 1][col_offset + 5] -= 1
        V[j_base*p + 0][col_offset + 5] += 1
        V[j_base*p + 1][col_offset + 5] += 1
        W[k_base*n + 1][col_offset + 5] += 1
        
        # M7
        U[i_base*m + 0][col_offset + 6] += 1
        U[i_base*m + 1][col_offset + 6] -= 1
        V[j_base*p + 1][col_offset + 6] += 1
        W[k_base*n + 1][col_offset + 6] += 1

    # Helper to add 2x2x3 schoolbook block
    def add_school_223(u_offset, v_offset, w_offset, i_base, j_base, k_base):
        # 2x2x3 schoolbook
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    idx = i*2 + j*6 + k # Local index? No, just use global col
                    U[i_base*m + i][col_offset + i*6 + j*3 + k] = 1
                    V[j_base*p + j][col_offset + i*6 + j*3 + k] = 1
                    W[k_base*n + k][col_offset + i*6 + j*3 + k] = 1

    # Helper to add 2x1x5 schoolbook block
    def add_school_215(u_offset, v_offset, w_offset, i_base, j_base, k_base):
        # 2x1x5 schoolbook
        for i in range(2):
            for k in range(5):
                idx = i*5 + k
                U[i_base*m + i][col_offset + idx] = 1
                V[j_base*p + 0][col_offset + idx] = 1
                W[k_base*n + k][col_offset + idx] = 1

    # Block 1: i=0..1, j=0..1, k=0..4
    # Split k into 0..1 and 2..4
    # 2x2x2 Strassen for k=0,1
    add_strassen_222(0, 0, 0, i_base=0, j_base=0, k_base=0)
    col_offset += 7
    
    # 2x2x3 Schoolbook for k=2,3,4
    add_school_223(7, 7, 7, i_base=0, j_base=0, k_base=2)
    col_offset += 12
    
    # Block 2: i=0..1, j=2..3, k=0..4
    # 2x2x2 Strassen for k=0,1
    add_strassen_222(19, 19, 19, i_base=0, j_base=2, k_base=0)
    col_offset += 7
    
    # 2x2x3 Schoolbook for k=2,3,4
    add_school_223(26, 26, 26, i_base=0, j_base=2, k_base=2)
    col_offset += 12
    
    # Block 3: i=0..1, j=4..4, k=0..4
    # 2x1x5 Schoolbook
    add_school_215(38, 38, 38, i_base=0, j_base=4, k_base=0)
    col_offset += 10
    
    return {"U": U, "V": V, "W": W}
