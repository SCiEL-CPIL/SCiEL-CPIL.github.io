"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen factors
    # U2[x, t], V2[y, t], W2[z, t]
    # x, y, z are 0..3 (row*2+col)
    # t is 0..6
    U2 = [
        [1, 0, 1, 0, 1, -1, 1], # x=0 (00)
        [0, 0, 0, 0, 1, 0, 1],  # x=1 (01)
        [0, 1, 0, 0, 0, 1, 0],  # x=2 (10)
        [1, 1, 0, 1, 0, 0, -1]  # x=3 (11)
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # y=0 (00)
        [0, 0, 1, 0, 0, 1, 0],  # y=1 (01)
        [0, 0, 0, 1, 0, 0, 1],  # y=2 (10)
        [1, 0, -1, 0, 1, 0, 1]  # y=3 (11)
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # z=0 (00)
        [0, 0, 1, 0, 1, 0, 0],  # z=1 (01)
        [0, 1, 0, 1, 0, 0, 0],  # z=2 (10)
        [1, -1, 1, 0, 0, 1, 0]  # z=3 (11)
    ]
    
    # Macro coefficients
    # coeff_A[s][b]: coefficient of A_block b in S_s
    coeff_A = [
        [1, 0, 0, 1], # S1
        [0, 0, 1, 1], # S2
        [1, 0, 0, 0], # S3
        [0, 0, 0, 1], # S4
        [1, 1, 0, 0], # S5
        [-1, 0, 1, 0],# S6
        [0, 1, 0, -1] # S7
    ]
    
    # coeff_B[s][b]: coefficient of B_block b in T_s
    coeff_B = [
        [1, 0, 0, 1], # T1
        [1, 0, 0, 0], # T2
        [0, 1, 0, -1],# T3
        [0, 0, 1, -1],# T4
        [0, 0, 0, 1], # T5
        [1, 1, 0, 0], # T6
        [0, 0, 1, 1]  # T7
    ]
    
    # coeff_C[s][bc]: coefficient of P_{s+1} in C_block bc
    coeff_C = [
        [1, 0, 0, 1], # P1 -> C11, C22
        [0, 0, 1, -1],# P2 -> C21, C22
        [0, 1, 0, 1], # P3 -> C12, C22
        [1, 0, 1, 0], # P4 -> C11, C21
        [-1, 1, 0, 0],# P5 -> C11, C12
        [0, 0, 0, 1], # P6 -> C22
        [1, 0, 0, 0]  # P7 -> C11
    ]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # Update U
            for x in range(4):
                u_val = U2[x][t]
                if u_val == 0: continue
                rx = x // 2
                cx = x % 2
                for b in range(4):
                    c_val = coeff_A[s][b]
                    if c_val == 0: continue
                    i = (b // 2) * 2 + rx
                    j = (b % 2) * 2 + cx
                    a_idx = i * 4 + j
                    U[a_idx][r] += c_val * u_val
            
            # Update V
            for y in range(4):
                v_val = V2[y][t]
                if v_val == 0: continue
                ry = y // 2
                cy = y % 2
                for b in range(4):
                    c_val = coeff_B[s][b]
                    if c_val == 0: continue
                    j = (b // 2) * 2 + ry
                    k = (b % 2) * 2 + cy
                    b_idx = j * 4 + k
                    V[b_idx][r] += c_val * v_val
            
            # Update W
            for z in range(4):
                w_val = W2[z][t]
                if w_val == 0: continue
                rz = z // 2
                cz = z % 2
                for bc in range(4):
                    c_val = coeff_C[s][bc]
                    if c_val == 0: continue
                    i = (bc // 2) * 2 + rz
                    k = (bc % 2) * 2 + cz
                    c_idx = k * 4 + i
                    W[c_idx][r] += c_val * w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
