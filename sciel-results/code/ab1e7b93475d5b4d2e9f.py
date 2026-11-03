import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen factors for 2x2x2
    # U_str[i_loc, j_loc, s]
    U_str = [[[0]*7 for _ in range(2)] for _ in range(2)]
    # V_str[j_loc, k_loc, s]
    V_str = [[[0]*7 for _ in range(2)] for _ in range(2)]
    # W_str[k_loc, i_loc, s]
    W_str = [[[0]*7 for _ in range(2)] for _ in range(2)]

    # M1
    U_str[0][0][0] += 1; U_str[1][1][0] += 1
    V_str[0][0][0] += 1; V_str[1][1][0] += 1
    W_str[0][0][0] += 1; W_str[1][1][0] += 1
    # M2
    U_str[1][0][1] += 1; U_str[1][1][1] += 1
    V_str[0][0][1] += 1
    W_str[1][0][1] += 1; W_str[1][1][1] -= 1
    # M3
    U_str[0][0][2] += 1
    V_str[1][0][2] += 1; V_str[1][1][2] -= 1
    W_str[0][1][2] += 1; W_str[1][1][2] += 1
    # M4
    U_str[1][1][3] += 1
    V_str[0][1][3] += 1; V_str[0][0][3] -= 1
    W_str[0][0][3] += 1; W_str[1][0][3] += 1
    # M5
    U_str[0][0][4] += 1; U_str[0][1][4] += 1
    V_str[1][1][4] += 1
    W_str[0][0][4] -= 1; W_str[0][1][4] += 1
    # M6
    U_str[1][0][5] += 1; U_str[0][0][5] -= 1
    V_str[0][0][5] += 1; V_str[0][1][5] += 1
    # M7
    U_str[0][1][6] += 1; U_str[1][1][6] -= 1
    V_str[1][0][6] += 1; V_str[1][1][6] += 1
    W_str[0][0][6] -= 1; W_str[1][1][6] += 1

    for b in range(8):
        b0 = b & 1
        b1 = (b >> 1) & 1
        b2 = (b >> 2) & 1
        
        for i_loc in range(2):
            for j_loc in range(2):
                i = i_loc
                j = 2 * j_loc + b0
                idx_u = i * m + j
                for s in range(7):
                    U[idx_u][7 * b + s] = U_str[i_loc][j_loc][s]
        
        for j_loc in range(2):
            for k_loc in range(2):
                j = 2 * j_loc + b0
                k = 4 * k_loc + 2 * b1 + b2
                idx_v = j * p + k
                for s in range(7):
                    V[idx_v][7 * b + s] = V_str[j_loc][k_loc][s]
        
        for k_loc in range(2):
            for i_loc in range(2):
                k = 4 * k_loc + 2 * b1 + b2
                i = i_loc
                idx_w = k * n + i
                for s in range(7):
                    W[idx_w][7 * b + s] = W_str[k_loc][i_loc][s]

    return {"U": U, "V": V, "W": W}
