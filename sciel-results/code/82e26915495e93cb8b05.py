DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7-term coefficients for 2x2 matrix multiplication
    # M = [[m00, m01], [m10, m11]], N = [[n00, n01], [n10, n11]]
    # P = M * N = [[p00, p01], [p10, p11]]
    
    # U: coefficients for M (first matrix)
    U_strassen = [
        {(0,0): 1, (1,1): 1},   # P1: m00 + m11
        {(1,0): 1, (1,1): 1},   # P2: m10 + m11
        {(0,0): 1},              # P3: m00
        {(1,1): 1},              # P4: m11
        {(0,0): 1, (0,1): 1},   # P5: m00 + m01
        {(1,0): 1, (0,0): -1},  # P6: m10 - m00
        {(0,1): 1, (1,1): -1},  # P7: m01 - m11
    ]
    
    # V: coefficients for N (second matrix)
    V_strassen = [
        {(0,0): 1, (1,1): 1},   # P1: n00 + n11
        {(0,0): 1},              # P2: n00
        {(0,1): 1, (1,1): -1},  # P3: n01 - n11
        {(1,0): 1},              # P4: n10
        {(1,1): 1},              # P5: n11
        {(0,0): 1, (1,0): 1},   # P6: n00 + n10
        {(1,0): 1, (1,1): 1},   # P7: n10 + n11
    ]
    
    # W: coefficients for P (output)
    # p00 = P1 + P4 - P5 + P7
    # p01 = P3 + P5
    # p10 = P2 + P4
    # p11 = P1 - P2 + P3 + P6
    W_strassen = [
        {(0,0): 1, (1,1): 1},   # P1
        {(1,0): 1, (1,1): -1},  # P2
        {(0,1): 1, (1,1): 1},   # P3
        {(0,0): 1, (1,0): 1},   # P4
        {(0,0): -1, (0,1): 1},  # P5
        {(1,1): 1},              # P6
        {(0,0): 1},              # P7
    ]
    
    col = 0
    for a in range(2):  # A column block: 0 for j∈{0,1}, 1 for j∈{2,3}
        for b in range(3):  # B column block: 0 for k∈{0,1}, 1 for k∈{2,3}, 2 for k∈{4,5}
            for s in range(7):  # Strassen term
                # Update U: A[i, a*2 + j_loc]
                for (i_loc, j_loc), coeff in U_strassen[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V: B[a*2 + j_loc, b*2 + k_loc]
                for (j_loc, k_loc), coeff in V_strassen[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W: C[i_loc, b*2 + k_loc] stored as W[k*n + i]
                for (i_loc, k_loc), coeff in W_strassen[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
