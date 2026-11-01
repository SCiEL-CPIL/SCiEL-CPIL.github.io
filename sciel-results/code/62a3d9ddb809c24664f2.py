import numpy as np

def construct():
    n, m, p = 2, 2, 15
    # We will use 7 blocks of 2x2x2 (Strassen, rank 7) and 1 block of 2x2x1 (Schoolbook, rank 4)
    # Total rank = 7 * 7 + 4 = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Strassen decomposition for 2x2x2
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # U indices: (i, j) -> i*m+j
    # V indices: (j, k) -> j*p+k
    # W indices: (k, i) -> k*n+i
    
    strassen_U = [
        [1, 0, 1, 0, 0, 0, 0], # i=0, j=0: A00 in P1, P3
        [0, 0, 0, 0, 1, 0, 1], # i=0, j=1: A01 in P5, P7
        [0, 1, 0, 0, 0, 1, 0], # i=1, j=0: A10 in P2, P6
        [1, 1, 0, 1, 0, 0, -1] # i=1, j=1: A11 in P1, P2, P4, P7
    ]
    
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0: B00 in P1, P2, P4, P6
        [1, 0, 1, 0, 0, 1, 0], # j=0, k=1: B01 in P1, P3, P6
        [0, 0, -1, 1, 1, 0, 1], # j=1, k=0: B10 in P3, P4, P5, P7
        [0, 0, 0, 0, 1, 0, 1] # j=1, k=1: B11 in P1, P3, P5, P7
    ]
    # Wait, let's re-verify V indices.
    # V is (j, k).
    # P1: B00, B11. V[0,0]=1, V[1,1]=1.
    # P2: B00. V[0,0]=1.
    # P3: B01, -B11. V[0,1]=1, V[1,1]=-1.
    # P4: B10, -B00. V[1,0]=1, V[0,0]=-1.
    # P5: B11. V[1,1]=1.
    # P6: B00, B01. V[0,0]=1, V[0,1]=1.
    # P7: B10, B11. V[1,0]=1, V[1,1]=1.
    
    # Let's rebuild V carefully.
    # Rows are (j, k). 00, 01, 10, 11.
    # Cols are 1..7.
    
    V_block = [[0]*7 for _ in range(4)]
    # P1
    V_block[0][0] += 1 # B00
    V_block[3][0] += 1 # B11
    # P2
    V_block[0][1] += 1 # B00
    # P3
    V_block[1][2] += 1 # B01
    V_block[3][2] += -1 # B11
    # P4
    V_block[2][3] += 1 # B10
    V_block[0][3] += -1 # B00
    # P5
    V_block[3][4] += 1 # B11
    # P6
    V_block[0][5] += 1 # B00
    V_block[1][5] += 1 # B01
    # P7
    V_block[2][6] += 1 # B10
    V_block[3][6] += 1 # B11

    # W indices: (k, i). 00, 01, 10, 11.
    # P1: C00, C11. W[0,0]=1, W[1,1]=1.
    # P2: C10, -C11. W[1,0]=1, W[1,1]=-1.
    # P3: C01, C11. W[0,1]=1, W[1,1]=1.
    # P4: C00, C10. W[0,0]=1, W[1,0]=1.
    # P5: -C00, C01. W[0,0]=-1, W[0,1]=1.
    # P6: C11. W[1,1]=1.
    # P7: C00. W[0,0]=1.
    
    W_block = [[0]*7 for _ in range(4)]
    # P1
    W_block[0][0] += 1 # C00
    W_block[3][0] += 1 # C11
    # P2
    W_block[2][1] += 1 # C10
    W_block[3][1] += -1 # C11
    # P3
    W_block[1][2] += 1 # C01
    W_block[3][2] += 1 # C11
    # P4
    W_block[0][3] += 1 # C00
    W_block[2][3] += 1 # C10
    # P5
    W_block[0][4] += -1 # C00
    W_block[1][4] += 1 # C01
    # P6
    W_block[3][5] += 1 # C11
    # P7
    W_block[0][6] += 1 # C00

    # Schoolbook for 2x2x1
    # Rank 4
    # U: 4x4 identity? No.
    # U rows: 00, 01, 10, 11.
    # V rows: 0k, 1k.
    # W rows: k0, k1.
    
    # Term 1: A00 B0k -> C0k. U[0,0]=1, V[0,k]=1, W[k,0]=1.
    # Term 2: A01 B1k -> C0k. U[0,1]=1, V[1,k]=1, W[k,0]=1.
    # Term 3: A10 B0k -> C1k. U[1,0]=1, V[0,k]=1, W[k,1]=1.
    # Term 4: A11 B1k -> C1k. U[1,1]=1, V[1,k]=1, W[k,1]=1.
    
    # We have 7 Strassen blocks (k=0..13) and 1 Schoolbook block (k=14).
    
    for block_idx in range(7):
        k_start = block_idx * 2
        for u_row in range(4):
            for c in range(7):
                U[u_row][col_offset + c] = strassen_U[u_row][c]
        for v_row in range(4):
            j = v_row // 2
            k_local = v_row % 2
            k_global = k_start + k_local
            v_idx = j * p + k_global
            for c in range(7):
                V[v_idx][col_offset + c] = V_block[v_row][c]
        for w_row in range(4):
            k_local = w_row // 2
            i = w_row % 2
            k_global = k_start + k_local
            w_idx = k_global * n + i
            for c in range(7):
                W[w_idx][col_offset + c] = W_block[w_row][c]
        col_offset += 7
        
    # Schoolbook block at k=14
    k_start = 14
    # U is identity for the 4 rows
    U[0][col_offset] = 1
    U[1][col_offset+1] = 1
    U[2][col_offset+2] = 1
    U[3][col_offset+3] = 1
    
    # V
    # Term 1: V[0, 14] = 1
    V[0 * p + k_start][col_offset] = 1
    # Term 2: V[1, 14] = 1
    V[1 * p + k_start][col_offset+1] = 1
    # Term 3: V[0, 14] = 1
    V[0 * p + k_start][col_offset+2] = 1
    # Term 4: V[1, 14] = 1
    V[1 * p + k_start][col_offset+3] = 1
    
    # W
    # Term 1: W[14, 0] = 1
    W[k_start * n + 0][col_offset] = 1
    # Term 2: W[14, 0] = 1
    W[k_start * n + 0][col_offset+1] = 1
    # Term 3: W[14, 1] = 1
    W[k_start * n + 1][col_offset+2] = 1
    # Term 4: W[14, 1] = 1
    W[k_start * n + 1][col_offset+3] = 1
    
    return {"U": U, "V": V, "W": W}
