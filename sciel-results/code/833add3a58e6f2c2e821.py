import time
import sys

DIMENSIONS = [2, 4, 5]

def add_strassen_block(U, V, W, a_idx, b_idx, c_idx, start):
    # a_idx: [a00, a01, a10, a11]
    # b_idx: [b00, b01, b10, b11]
    # c_idx: [c00, c01, c10, c11]
    
    # M1 = (a00 + a11)(b00 + b11)
    U[a_idx[0]][start] += 1
    U[a_idx[3]][start] += 1
    V[b_idx[0]][start] += 1
    V[b_idx[3]][start] += 1
    W[c_idx[0]][start] += 1
    W[c_idx[3]][start] += 1
    
    # M2 = (a10 + a11)b00
    U[a_idx[2]][start+1] += 1
    U[a_idx[3]][start+1] += 1
    V[b_idx[0]][start+1] += 1
    W[c_idx[1]][start+1] += 1 # Wait, Z10 = M2 + M4. So W[c10] gets M2.
    # Correction:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Let's map carefully.
    # M1 contributes to Z00 (+1) and Z11 (+1)
    # M2 contributes to Z10 (+1) and Z11 (-1)
    # M3 contributes to Z01 (+1) and Z11 (+1)
    # M4 contributes to Z00 (+1) and Z10 (+1)
    # M5 contributes to Z00 (-1) and Z01 (+1)
    # M6 contributes to Z11 (+1)
    # M7 contributes to Z00 (+1)

    # M1
    W[c_idx[0]][start] += 1
    W[c_idx[3]][start] += 1

    # M2
    U[a_idx[2]][start+1] += 1
    U[a_idx[3]][start+1] += 1
    V[b_idx[0]][start+1] += 1
    W[c_idx[2]][start+1] += 1
    W[c_idx[3]][start+1] -= 1

    # M3
    U[a_idx[0]][start+2] += 1
    V[b_idx[1]][start+2] -= 1
    V[b_idx[3]][start+2] += 1
    W[c_idx[1]][start+2] += 1
    W[c_idx[3]][start+2] += 1

    # M4
    U[a_idx[3]][start+3] += 1
    V[b_idx[2]][start+3] -= 1
    V[b_idx[0]][start+3] += 1
    W[c_idx[0]][start+3] += 1
    W[c_idx[2]][start+3] += 1

    # M5
    U[a_idx[0]][start+4] += 1
    U[a_idx[1]][start+4] += 1
    V[b_idx[3]][start+4] += 1
    W[c_idx[0]][start+4] -= 1
    W[c_idx[1]][start+4] += 1

    # M6
    U[a_idx[2]][start+5] -= 1
    U[a_idx[0]][start+5] += 1
    V[b_idx[0]][start+5] += 1
    V[b_idx[1]][start+5] += 1
    W[c_idx[3]][start+5] += 1

    # M7
    U[a_idx[1]][start+6] -= 1
    U[a_idx[3]][start+6] += 1
    V[b_idx[2]][start+6] += 1
    V[b_idx[3]][start+6] += 1
    W[c_idx[0]][start+6] += 1

def add_schoolbook_block(U, V, W, a_idx, b_idx, c_idx, start):
    # a_idx: [a00, a01, a10, a11]
    # b_idx: [b00, b10]
    # c_idx: [c00, c10]
    
    # M1 = a00 b00
    U[a_idx[0]][start] += 1
    V[b_idx[0]][start] += 1
    W[c_idx[0]][start] += 1

    # M2 = a01 b10
    U[a_idx[1]][start+1] += 1
    V[b_idx[1]][start+1] += 1
    W[c_idx[0]][start+1] += 1

    # M3 = a10 b00
    U[a_idx[2]][start+2] += 1
    V[b_idx[0]][start+2] += 1
    W[c_idx[1]][start+2] += 1

    # M4 = a11 b10
    U[a_idx[3]][start+3] += 1
    V[b_idx[1]][start+3] += 1
    W[c_idx[1]][start+3] += 1

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: Strassen A_L B_T1
    a1 = [0, 1, 4, 5]
    b1 = [0, 1, 5, 6]
    c1 = [0, 2, 1, 3]
    add_strassen_block(U, V, W, a1, b1, c1, 0)
    
    # Block 2: Strassen A_L B_T2
    a2 = [0, 1, 4, 5]
    b2 = [2, 3, 7, 8]
    c2 = [4, 6, 5, 7]
    add_strassen_block(U, V, W, a2, b2, c2, 7)
    
    # Block 3: Schoolbook A_L B_T3
    a3 = [0, 1, 4, 5]
    b3 = [4, 9]
    c3 = [8, 9]
    add_schoolbook_block(U, V, W, a3, b3, c3, 14)
    
    # Block 4: Strassen A_R B_B1
    a4 = [2, 3, 6, 7]
    b4 = [10, 11, 15, 16]
    c4 = [0, 2, 1, 3]
    add_strassen_block(U, V, W, a4, b4, c4, 18)
    
    # Block 5: Strassen A_R B_B2
    a5 = [2, 3, 6, 7]
    b5 = [12, 13, 17, 18]
    c5 = [4, 6, 5, 7]
    add_strassen_block(U, V, W, a5, b5, c5, 25)
    
    # Block 6: Schoolbook A_R B_B3
    a6 = [2, 3, 6, 7]
    b6 = [14, 19]
    c6 = [8, 9]
    add_schoolbook_block(U, V, W, a6, b6, c6, 32)
    
    return {"U": U, "V": V, "W": W}
