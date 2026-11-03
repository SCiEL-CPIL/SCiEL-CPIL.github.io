"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2():
    """Return rank-7 decomposition for 2x2x2 matrix multiplication.
    U[i*2+j, alpha], V[j*2+k, alpha], W[k*2+i, alpha]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _rank49_4x4x4():
    """Return rank-49 decomposition for 4x4x4 matrix multiplication."""
    n = 4
    U = [[0]*49 for _ in range(n*n)]
    V = [[0]*49 for _ in range(n*n)]
    W = [[0]*49 for _ in range(n*n)]
    
    U2, V2, W2 = _strassen_2x2()
    
    A_blocks = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    
    B_blocks = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]
    
    C_blocks = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],
        [(2, 1), (4, 1)],
        [(1, 1), (3, 1)],
        [(0, 1), (1, -1), (2, 1), (5, 1)],
    ]
    
    for s in range(7):
        for t in range(7):
            alpha = s * 7 + t
            for (block_idx, coeff) in A_blocks[s]:
                br = block_idx // 2
                bc = block_idx % 2
                for di in range(2):
                    for dj in range(2):
                        i = br * 2 + di
                        j = bc * 2 + dj
                        u_idx = i * 4 + j
                        u2_idx = di * 2 + dj
                        U[u_idx][alpha] += coeff * U2[u2_idx][t]
            for (block_idx, coeff) in B_blocks[s]:
                br = block_idx // 2
                bc = block_idx % 2
                for dj in range(2):
                    for dk in range(2):
                        j = br * 2 + dj
                        k = bc * 2 + dk
                        v_idx = j * 4 + k
                        v2_idx = dj * 2 + dk
                        V[v_idx][alpha] += coeff * V2[v2_idx][t]
            for c_idx in range(4):
                for (sp, coeff) in C_blocks[c_idx]:
                    if sp == s:
                        cr = c_idx // 2
                        cc = c_idx % 2
                        for dk in range(2):
                            for di in range(2):
                                k = cr * 2 + dk
                                i = cc * 2 + di
                                w_idx = k * 4 + i
                                w2_idx = dk * 2 + di
                                W[w_idx][alpha] += coeff * W2[w2_idx][t]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U4, V4, W4 = _rank49_4x4x4()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(16):
        for alpha in range(49):
            U[i][alpha] = U4[i][alpha]
            U[i][49 + alpha] = U4[i][alpha]
    
    for j in range(4):
        for k in range(4):
            v_idx = j * 8 + k
            v4_idx = j * 4 + k
            for alpha in range(49):
                V[v_idx][alpha] = V4[v4_idx][alpha]
    
    for j in range(4):
        for k in range(4):
            v_idx = j * 8 + (k + 4)
            v4_idx = j * 4 + k
            for alpha in range(49):
                V[v_idx][49 + alpha] = V4[v4_idx][alpha]
    
    for k in range(4):
        for i in range(4):
            w_idx = k * 4 + i
            for alpha in range(49):
                W[w_idx][alpha] = W4[w_idx][alpha]
    
    for k in range(4):
        for i in range(4):
            w_idx = (k + 4) * 4 + i
            w4_idx = k * 4 + i
            for alpha in range(49):
                W[w_idx][49 + alpha] = W4[w4_idx][alpha]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
