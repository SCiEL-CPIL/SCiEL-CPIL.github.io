import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, leave 5s margin
    
    n, m, p = 4, 4, 7
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for 4x4x4
    # First, define 2x2x2 Strassen
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    # M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    # M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    # M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    # M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    # M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    # M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Construct 4x4x4 Strassen
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for c_block in range(7):
        for c_internal in range(7):
            c_total = c_block * 7 + c_internal
            # U4
            for i_block in range(2):
                for i_internal in range(2):
                    i = i_block * 2 + i_internal
                    for j_block in range(2):
                        for j_internal in range(2):
                            j = j_block * 2 + j_internal
                            u_block = U2[i_block * 2 + j_block][c_block]
                            u_internal = U2[i_internal * 2 + j_internal][c_internal]
                            if u_block != 0 and u_internal != 0:
                                U4[i * 4 + j][c_total] = u_block * u_internal
            # V4
            for j_block in range(2):
                for j_internal in range(2):
                    j = j_block * 2 + j_internal
                    for k_block in range(2):
                        for k_internal in range(2):
                            k = k_block * 2 + k_internal
                            v_block = V2[j_block * 2 + k_block][c_block]
                            v_internal = V2[j_internal * 2 + k_internal][c_internal]
                            if v_block != 0 and v_internal != 0:
                                V4[j * 4 + k][c_total] = v_block * v_internal
            # W4
            for k_block in range(2):
                for k_internal in range(2):
                    k = k_block * 2 + k_internal
                    for i_block in range(2):
                        for i_internal in range(2):
                            i = i_block * 2 + i_internal
                            w_block = W2[k_block * 2 + i_block][c_block]
                            w_internal = W2[k_internal * 2 + i_internal][c_internal]
                            if w_block != 0 and w_internal != 0:
                                W4[k * 4 + i][c_total] = w_block * w_internal
    
    # Schoolbook for 4x4x3
    U3 = [[0]*48 for _ in range(16)]
    V3 = [[0]*48 for _ in range(12)]
    W3 = [[0]*48 for _ in range(12)]
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                c = i * 12 + j * 3 + k
                U3[i * 4 + j][c] = 1
                V3[j * 3 + k][c] = 1
                W3[k * 4 + i][c] = 1
    
    # Combine
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
        for c in range(48):
            U[i][49 + c] = U3[i][c]
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 7 + k][c] = V4[j * 4 + k][c]
        for k in range(3):
            for c in range(48):
                V[j * 7 + 4 + k][49 + c] = V3[j * 3 + k][c]
    
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k * 4 + i][c] = W4[k * 4 + i][c]
        for i in range(4):
            for c in range(48):
                W[4 * 4 + k + i][49 + c] = W3[k * 4 + i][c]
    
    if time.time() > deadline:
        # Should not happen, but just in case
        pass
        
    return {"U": U, "V": V, "W": W}
