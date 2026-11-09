"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return (U, V, W) for 2x2 x 2x2 matrix multiplication, rank 7.
    U: 4 rows x 7 cols, V: 4 rows x 7 cols, W: 4 rows x 7 cols.
    U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a0+a3)(b0+b3) -> c0 += 1, c3 += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (a2+a3)b0 -> c2 += 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # p3 = a0(b1-b3) -> c1 += 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # p4 = a3(b2-b0) -> c2 += 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1
    
    # p5 = (a0+a1)b3 -> c1 += 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1
    
    # p6 = (a2-a0)(b0+b1) -> c3 += 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # p7 = (a1-a3)(b2+b3) -> c3 += 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[3][6] = 1
    
    return U, V, W

def construct():
    """Construct rank-minimal decomposition for 4x4 x 4x7 matrix multiplication."""
    # Build 4x4x4 decomposition via tensor product of two 2x2x2 Strassen (rank 7 x 7 = 49)
    U1, V1, W1 = _strassen_2x2()
    r1 = 7
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for ra in range(r1):
        for rb in range(r1):
            col = ra * r1 + rb
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    u_val = U1[i1*2+j1][ra] * U1[i2*2+j2][rb]
                                    if u_val != 0:
                                        U4[(i1*2+i2)*4 + (j1*2+j2)][col] += u_val
                                    
                                    v_val = V1[j1*2+k1][ra] * V1[j2*2+k2][rb]
                                    if v_val != 0:
                                        V4[(j1*2+j2)*4 + (k1*2+k2)][col] += v_val
                                    
                                    w_val = W1[k1*2+i1][ra] * W1[k2*2+i2][rb]
                                    if w_val != 0:
                                        W4[(k1*2+k2)*4 + (i1*2+i2)][col] += w_val
    
    # Build 4x4x3 schoolbook decomposition (rank 48)
    U43 = [[0]*48 for _ in range(16)]
    V43 = [[0]*48 for _ in range(12)]
    W43 = [[0]*48 for _ in range(12)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U43[i*4+j][col] = 1
                V43[j*3+k][col] = 1
                W43[k*4+i][col] = 1
                col += 1
    
    # Combine: total rank = 97
    total_rank = 49 + 48
    U_final = [[0]*total_rank for _ in range(16)]
    V_final = [[0]*total_rank for _ in range(28)]
    W_final = [[0]*total_rank for _ in range(28)]
    
    # 4x4x4 part: columns 0-48
    for i in range(16):
        for r in range(49):
            U_final[i][r] = U4[i][r]
            V_final[i][r] = V4[i][r]
            W_final[i][r] = W4[i][r]
    
    # 4x4x3 part: columns 49-96
    for i in range(16):
        for r in range(48):
            U_final[i][49+r] = U43[i][r]
    
    for j in range(4):
        for k in range(3):
            for r in range(48):
                V_final[j*7 + (k+4)][49+r] = V43[j*3+k][r]
    
    for k in range(3):
        for i in range(4):
            for r in range(48):
                W_final[(k+4)*4 + i][49+r] = W43[k*4+i][r]
    
    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-END
