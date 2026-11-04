# EVOLVE-BLOCK-START
def _strassen_2x2_decomp():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    
    # M3
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # M6
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[3][6] += 1
    
    return U, V, W

def _strassen_4x4_decomp():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using Strassen.
    U shape: (16, 49)
    V shape: (16, 49)
    W shape: (16, 49)
    """
    U_4 = [[0]*49 for _ in range(16)]
    V_4 = [[0]*49 for _ in range(16)]
    W_4 = [[0]*49 for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2_decomp()
    
    mappings = [
        ([(1, 0, 0), (1, 1, 1)], [(1, 0, 0), (1, 1, 1)], [(1, 0, 0), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(1, 0, 0)], [(1, 0, 1)]),
        ([(1, 0, 0)], [(1, 0, 1), (-1, 1, 1)], [(1, 0, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 1, 0), (-1, 0, 0)], [(1, 0, 0), (1, 0, 1)]),
        ([(1, 0, 0), (1, 0, 1)], [(1, 1, 1)], [(-1, 0, 0), (1, 0, 1)]),
        ([(1, 0, 1), (-1, 0, 0)], [(1, 0, 0), (1, 0, 1)], [(1, 1, 1)]),
        ([(1, 0, 1), (-1, 1, 1)], [(1, 1, 0), (1, 1, 1)], [(1, 0, 0), (1, 1, 1)]),
    ]
    
    for k in range(7):
        base_col = k * 7
        a_maps, b_maps, c_maps = mappings[k]
        
        for c_idx in range(7):
            col = base_col + c_idx
            
            for i2 in range(2):
                for j2 in range(2):
                    u_val = U2[i2*2+j2][c_idx]
                    if u_val != 0:
                        for coeff, i1, j1 in a_maps:
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U_4[i*4+j][col] += u_val * coeff
            
            for j2 in range(2):
                for k2 in range(2):
                    v_val = V2[j2*2+k2][c_idx]
                    if v_val != 0:
                        for coeff, j1, k1 in b_maps:
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V_4[j*4+k][col] += v_val * coeff
            
            for k2 in range(2):
                for i2 in range(2):
                    w_val = W2[k2*2+i2][c_idx]
                    if w_val != 0:
                        for coeff, i1, k1 in c_maps:
                            i = i1 * 2 + i2
                            k = k1 * 2 + k2
                            W_4[k*4+i][col] += w_val * coeff
                            
    return U_4, V_4, W_4

def _schoolbook_4x4x1_decomp():
    """
    Returns U, V, W for 4x4x1 matrix multiplication.
    U shape: (16, 16)
    V shape: (4, 16)
    W shape: (4, 16)
    """
    U = [[0]*16 for _ in range(16)]
    V = [[0]*16 for _ in range(4)]
    W = [[0]*16 for _ in range(4)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
            
    return U, V, W

def construct():
    n, m, p = 4, 4, 9
    rank = 114
    
    U1, V1, W1 = _strassen_4x4_decomp()
    U2, V2, W2 = _strassen_4x4_decomp()
    U3, V3, W3 = _schoolbook_4x4x1_decomp()
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Sub 1: Rank 49
    for c in range(49):
        for r_u in range(16):
            U[r_u][c] = U1[r_u][c]
        for r_v in range(16):
            j = r_v // 4
            k = r_v % 4
            r_v_global = j * 9 + k
            V[r_v_global][c] = V1[r_v][c]
        for r_w in range(16):
            k = r_w // 4
            i = r_w % 4
            r_w_global = k * 4 + i
            W[r_w_global][c] = W1[r_w][c]
            
    # Sub 2: Rank 49
    for c in range(49):
        for r_u in range(16):
            U[r_u][c+49] = U2[r_u][c]
        for r_v in range(16):
            j = r_v // 4
            k_local = r_v % 4
            k_global = k_local + 4
            r_v_global = j * 9 + k_global
            V[r_v_global][c+49] = V2[r_v][c]
        for r_w in range(16):
            k_local = r_w // 4
            i = r_w % 4
            k_global = k_local + 4
            r_w_global = k_global * 4 + i
            W[r_w_global][c+49] = W2[r_w][c]
            
    # Sub 3: Rank 16
    for c in range(16):
        for r_u in range(16):
            U[r_u][c+98] = U3[r_u][c]
        for r_v in range(4):
            r_v_global = r_v * 9 + 8
            V[r_v_global][c+98] = V3[r_v][c]
        for r_w in range(4):
            r_w_global = 8 * 4 + r_w
            W[r_w_global][c+98] = W3[r_w][c]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
