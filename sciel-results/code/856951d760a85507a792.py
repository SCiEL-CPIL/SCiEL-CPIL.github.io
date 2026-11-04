import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local Strassen matrices for 2x2x2
    # U_local: 4x7
    # V_local: 4x7
    # W_local: 4x7
    # Indices 0..3 correspond to (11, 12, 21, 22)
    
    U_local = np.zeros((4, 7))
    V_local = np.zeros((4, 7))
    W_local = np.zeros((4, 7))
    
    # M1: (A11+A22)(B11+B22)
    U_local[0, 0] = 1; U_local[2, 0] = 1
    V_local[0, 0] = 1; V_local[2, 0] = 1
    W_local[0, 0] = 1; W_local[3, 0] = 1
    
    # M2: (A21+A22)B11
    U_local[1, 1] = 1; U_local[2, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1
    
    # M3: A11(B12-B22)
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[2, 2] = -1
    W_local[1, 2] = 1; W_local[3, 2] = 1
    
    # M4: A22(B21-B11)
    U_local[2, 3] = 1
    V_local[1, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1; W_local[2, 3] = 1
    
    # M5: (A11+A12)B22
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[2, 4] = 1
    W_local[0, 4] = -1; W_local[1, 4] = 1
    
    # M6: (A21-A11)(B11+B12)
    U_local[1, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    
    # M7: (A12-A22)(B21+B22)
    U_local[1, 6] = 1; U_local[2, 6] = -1
    V_local[1, 6] = 1; V_local[2, 6] = 1
    W_local[0, 6] = 1
    
    # Blocks:
    # 1: A_left (j=0,1), B_11 (j=0,1, k=0,1) -> C_col 0,1
    # 2: A_right (j=2,3), B_21 (j=2,3, k=0,1) -> C_col 0,1
    # 3: A_left (j=0,1), B_12 (j=0,1, k=2,3) -> C_col 2,3
    # 4: A_right (j=2,3), B_22 (j=2,3, k=2,3) -> C_col 2,3
    
    # Mapping local indices (0,1,2,3) to global indices
    # Local A idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> i*2+j
    # Local B idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> j*2+k
    # Local C idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> k*2+i
    
    # Block 1: A j=0,1; B j=0,1, k=0,1
    # A global idx: i*4 + j
    # B global idx: j*4 + k
    # C global idx: k*2 + i
    
    col_offset = 0
    for r in range(7):
        # U
        for la in range(4):
            val = U_local[la, r]
            if val != 0:
                # la: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                li = la // 2
                lj = la % 2
                ga = li * 4 + (lj + 0) # j offset 0
                U[ga, col_offset + r] += val
        # V
        for lb in range(4):
            val = V_local[lb, r]
            if val != 0:
                lj = lb // 2
                lk = lb % 2
                gb = (lj + 0) * 4 + (lk + 0) # j offset 0, k offset 0
                V[gb, col_offset + r] += val
        # W
        for lc in range(4):
            val = W_local[lc, r]
            if val != 0:
                lk = lc // 2
                li = lc % 2
                gc = (lk + 0) * 2 + li # k offset 0
                W[gc, col_offset + r] += val
    col_offset += 7
    
    # Block 2: A j=2,3; B j=2,3, k=0,1
    for r in range(7):
        for la in range(4):
            val = U_local[la, r]
            if val != 0:
                li = la // 2
                lj = la % 2
                ga = li * 4 + (lj + 2) # j offset 2
                U[ga, col_offset + r] += val
        for lb in range(4):
            val = V_local[lb, r]
            if val != 0:
                lj = lb // 2
                lk = lb % 2
                gb = (lj + 2) * 4 + (lk + 0) # j offset 2, k offset 0
                V[gb, col_offset + r] += val
        for lc in range(4):
            val = W_local[lc, r]
            if val != 0:
                lk = lc // 2
                li = lc % 2
                gc = (lk + 0) * 2 + li # k offset 0
                W[gc, col_offset + r] += val
    col_offset += 7
    
    # Block 3: A j=0,1; B j=0,1, k=2,3
    for r in range(7):
        for la in range(4):
            val = U_local[la, r]
            if val != 0:
                li = la // 2
                lj = la % 2
                ga = li * 4 + (lj + 0) # j offset 0
                U[ga, col_offset + r] += val
        for lb in range(4):
            val = V_local[lb, r]
            if val != 0:
                lj = lb // 2
                lk = lb % 2
                gb = (lj + 0) * 4 + (lk + 2) # j offset 0, k offset 2
                V[gb, col_offset + r] += val
        for lc in range(4):
            val = W_local[lc, r]
            if val != 0:
                lk = lc // 2
                li = lc % 2
                gc = (lk + 2) * 2 + li # k offset 2
                W[gc, col_offset + r] += val
    col_offset += 7
    
    # Block 4: A j=2,3; B j=2,3, k=2,3
    for r in range(7):
        for la in range(4):
            val = U_local[la, r]
            if val != 0:
                li = la // 2
                lj = la % 2
                ga = li * 4 + (lj + 2) # j offset 2
                U[ga, col_offset + r] += val
        for lb in range(4):
            val = V_local[lb, r]
            if val != 0:
                lj = lb // 2
                lk = lb % 2
                gb = (lj + 2) * 4 + (lk + 2) # j offset 2, k offset 2
                V[gb, col_offset + r] += val
        for lc in range(4):
            val = W_local[lc, r]
            if val != 0:
                lk = lc // 2
                li = lc % 2
                gc = (lk + 2) * 2 + li # k offset 2
                W[gc, col_offset + r] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
