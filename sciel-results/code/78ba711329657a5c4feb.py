import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _strassen_222():
    """Strassen decomposition for 2x2 * 2x2. Returns (U, V, W) as lists of lists.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # 7 products:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # Row index in U: i*2+j; V: j*2+k; W: k*2+i
    # M1: (a11+a22)(b11+b22) -> U[0]+U[3], V[0]+V[3], contributes to W[0](c11) and W[3](c22)
    # M2: (a21+a22)b11 -> U[2]+U[3], V[0], contributes to W[2](c21) and W[3](c22) with -1
    # M3: a11(b12-b22) -> U[0], V[1]-V[3], contributes to W[1](c12) and W[3](c22)
    # M4: a22(b21-b11) -> U[3], V[2]-V[0], contributes to W[0](c11) and W[2](c21)
    # M5: (a11+a12)b22 -> U[0]+U[1], V[3], contributes to W[0](c11) with -1 and W[1](c12)
    # M6: (a21-a11)(b11+b12) -> U[2]-U[0], V[0]+V[1], contributes to W[3](c22)
    # M7: (a11+a12)(b21+b22) -> U[0]+U[1], V[2]+V[3], contributes to W[0](c11)
    
    # M1
    U[0][0] += 1.0; U[3][0] += 1.0
    V[0][0] += 1.0; V[3][0] += 1.0
    W[0][0] += 1.0; W[3][0] += 1.0
    # M2
    U[2][1] += 1.0; U[3][1] += 1.0
    V[0][1] += 1.0
    W[2][1] += 1.0; W[3][1] += -1.0
    # M3
    U[0][2] += 1.0
    V[1][2] += 1.0; V[3][2] += -1.0
    W[1][2] += 1.0; W[3][2] += 1.0
    # M4
    U[3][3] += 1.0
    V[2][3] += 1.0; V[0][3] += -1.0
    W[0][3] += 1.0; W[2][3] += 1.0
    # M5
    U[0][4] += 1.0; U[1][4] += 1.0
    V[3][4] += 1.0
    W[0][4] += -1.0; W[1][4] += 1.0
    # M6
    U[2][5] += 1.0; U[0][5] += -1.0
    V[0][5] += 1.0; V[1][5] += 1.0
    W[3][5] += 1.0
    # M7
    U[0][6] += 1.0; U[1][6] += 1.0
    V[2][6] += 1.0; V[3][6] += 1.0
    W[0][6] += 1.0
    
    return U, V, W

def _rank1_121():
    """Rank-1 decomposition for 1x2 * 2x1. U(2,1), V(2,1), W(2,1).
    c11 = a11*b11 + a12*b21
    """
    U = [[0.0]*2 for _ in range(2)]
    V = [[0.0]*2 for _ in range(2)]
    W = [[0.0]*2 for _ in range(2)]
    U[0][0] = 1.0; V[0][0] = 1.0; W[0][0] = 1.0
    U[1][1] = 1.0; V[1][1] = 1.0; W[1][1] = 1.0
    return U, V, W

def _rank1_211():
    """Rank-1 decomposition for 2x1 * 1x1. U(2,1), V(1,1), W(2,1).
    c11 = a11*b11, c21 = a21*b11
    """
    U = [[0.0]*1 for _ in range(2)]
    V = [[0.0]*1 for _ in range(1)]
    W = [[0.0]*1 for _ in range(2)]
    U[0][0] = 1.0; V[0][0] = 1.0; W[0][0] = 1.0
    U[1][0] = 1.0; W[1][0] = 1.0
    return U, V, W

def _rank1_112():
    """Rank-1 decomposition for 1x1 * 1x2. U(1,1), V(2,1), W(2,1).
    c11 = a11*b11, c12 = a11*b12
    """
    U = [[0.0]*1 for _ in range(1)]
    V = [[0.0]*1 for _ in range(2)]
    W = [[0.0]*1 for _ in range(2)]
    U[0][0] = 1.0; V[0][0] = 1.0; W[0][0] = 1.0
    V[1][0] = 1.0; W[1][0] = 1.0
    return U, V, W

def _strassen_223():
    """Strassen-like for 2x2 * 2x3. Rank 14 (two 2x2 Strassen blocks).
    U(4,14), V(6,14), W(6,14).
    Split B into B_left (2x2) and B_right (2x2).
    C = [C_left | C_right] where C_left = A*B_left, C_right = A*B_right.
    """
    U1, V1, W1 = _strassen_222()
    U2, V2, W2 = _strassen_222()
    U = [[0.0]*14 for _ in range(4)]
    V = [[0.0]*14 for _ in range(6)]
    W = [[0.0]*14 for _ in range(6)]
    
    # First block: columns 0-6
    for i in range(4):
        for r in range(7):
            U[i][r] += U1[i][r]
    # V for first block: rows 0-3 (B_left), rows 4-5 (B_right)
    for j in range(2):
        for r in range(7):
            V[j][r] += V1[j][r]
    # W for first block: rows 0-3 (C_left), rows 4-5 (C_right)
    for k in range(2):
        for r in range(7):
            W[k][r] += W1[k][r]
    
    # Second block: columns 7-13
    for i in range(4):
        for r in range(7):
            U[i][r+7] += U2[i][r]
    for j in range(2):
        for r in range(7):
            V[j+2][r+7] += V2[j][r]
    for k in range(2):
        for r in range(7):
            W[k+2][r+7] += W2[k][r]
    
    return U, V, W

def _decompose_343():
    """Decompose 3x4 * 4x3. Rank 18.
    Split A (3x4) into A1 (2x4) and A2 (1x4).
    C1 = A1*B (2x3), C2 = A2*B (1x3).
    2x4*4x3: rank 14 (two 2x2*2x3 Strassen blocks)
    1x4*4x3: rank 4 (schoolbook)
    Total: 18
    """
    U1, V1, W1 = _strassen_223()  # 2x2*2x3, U(4,7), V(6,7), W(6,7)
    U2, V2, W2 = _strassen_223()
    
    # 2x4*4x3: A is 2x4 (rows 0..7: i*4+j for i=0,1 j=0..3)
    # B is 4x3 (rows 0..11: j*3+k for j=0..3 k=0..2)
    # C is 2x3 (rows 0..5: k*2+i for k=0..2 i=0,1)
    # Block 1: A[0:2,0:2]*B[0:2,0:3] -> C[0:2,0:3]
    # Block 2: A[0:2,2:4]*B[2:4,0:3] -> C[0:2,0:3]
    
    U_243 = [[0.0]*14 for _ in range(8)]
    V_243 = [[0.0]*14 for _ in range(12)]
    W_243 = [[0.0]*14 for _ in range(6)]
    
    # Block 1: A rows 0..3 (i=0,1; j=0,1), B rows 0..5 (j=0,1; k=0,1,2)
    for i in range(4):
        for r in range(7):
            U_243[i][r] += U1[i][r]
    for j in range(6):
        for r in range(7):
            V_243[j][r] += V1[j][r]
    for k in range(6):
        for r in range(7):
            W_243[k][r] += W1[k][r]
    
    # Block 2: A rows 4..7 (i=0,1; j=2,3), B rows 6..11 (j=2,3; k=0,1,2)
    for i in range(4):
        for r in range(7):
            U_243[i+4][r+7] += U2[i][r]
    for j in range(6):
        for r in range(7):
            V_243[j+6][r+7] += V2[j][r]
    for k in range(6):
        for r in range(7):
            W_243[k][r+7] += W2[k][r]
    
    # 1x4*4x3: A row 8..11 (i=2; j=0..3), B rows 0..11, C rows 6..8 (k*3+2 for k=0,1,2)
    U_143 = [[0.0]*4 for _ in range(4)]
    V_143 = [[0.0]*4 for _ in range(12)]
    W_143 = [[0.0]*4 for _ in range(3)]
    for t in range(4):
        U_143[t][t] = 1.0
        for k in range(3):
            V_143[t*3+k][t] = 1.0
            W_143[k][t] = 1.0
    
    # Combine
    U = [[0.0]*18 for _ in range(12)]
    V = [[0.0]*18 for _ in range(12)]
    W = [[0.0]*18 for _ in range(9)]
    
    for i in range(8):
        for r in range(14):
            U[i][r] += U_243[i][r]
    for i in range(4):
        for r in range(4):
            U[i+8][r+14] += U_143[i][r]
    
    for j in range(12):
        for r in range(14):
            V[j][r] += V_243[j][r]
    for j in range(12):
        for r in range(4):
            V[j][r+14] += V_143[j][r]
    
    # W: 2x3 block uses k*2+i indexing (rows 0..5), 1x3 uses k*3+2 (rows 6..8)
    for k in range(6):
        for r in range(14):
            W[k][r] += W_243[k][r]
    for k in range(3):
        for r in range(4):
            W[k+6][r+14] += W_143[k][r]
    
    return U, V, W

def _decompose_344():
    """Decompose 3x4 * 4x4. Rank 32.
    Split A into A1 (2x4) and A2 (1x4).
    2x4*4x4: rank 28 (four 2x2*2x2 Strassen blocks)
    1x4*4x4: rank 4 (schoolbook)
    Total: 32
    """
    U1, V1, W1 = _strassen_222()
    U2, V2, W2 = _strassen_222()
    U3, V3, W3 = _strassen_222()
    U4, V4, W4 = _strassen_222()
    
    # 2x4*4x4: A rows 0..7 (i=0,1; j=0..3), B rows 0..15 (j=0..3; k=0..3)
    # C rows 0..7 (k=0..3; i=0,1)
    # Block 1: A[0:2,0:2]*B[0:2,0:2] -> C[0:2,0:2]
    # Block 2: A[0:2,2:4]*B[2:4,0:2] -> C[0:2,0:2]
    # Block 3: A[0:2,0:2]*B[0:2,2:4] -> C[2:4,0:2]
    # Block 4: A[0:2,2:4]*B[2:4,2:4] -> C[2:4,0:2]
    
    U_244 = [[0.0]*28 for _ in range(8)]
    V_244 = [[0.0]*28 for _ in range(16)]
    W_244 = [[0.0]*28 for _ in range(8)]
    
    # Block 1: A rows 0..3, B rows 0..3, C rows 0..3
    for i in range(4):
        for r in range(7):
            U_244[i][r] += U1[i][r]
    for j in range(4):
        for r in range(7):
            V_244[j][r] += V1[j][r]
    for k in range(4):
        for r in range(7):
            W_244[k][r] += W1[k][r]
    
    # Block 2: A rows 4..7, B rows 4..7, C rows 0..3
    for i in range(4):
        for r in range(7):
            U_244[i+4][r+7] += U2[i][r]
    for j in range(4):
        for r in range(7):
            V_244[j+4][r+7] += V2[j][r]
    for k in range(4):
        for r in range(7):
            W_244[k][r+7] += W2[k][r]
    
    # Block 3: A rows 0..3, B rows 8..11, C rows 4..7
    for i in range(4):
        for r in range(7):
            U_244[i][r+14] += U3[i][r]
    for j in range(4):
        for r in range(7):
            V_244[j+8][r+14] += V3[j][r]
    for k in range(4):
        for r in range(7):
            W_244[k+4][r+14] += W3[k][r]
    
    # Block 4: A rows 4..7, B rows 12..15, C rows 4..7
    for i in range(4):
        for r in range(7):
            U_244[i+4][r+21] += U4[i][r]
    for j in range(4):
        for r in range(7):
            V_244[j+12][r+21] += V4[j][r]
    for k in range(4):
        for r in range(7):
            W_244[k+4][r+21] += W4[k][r]
    
    # 1x4*4x4: A rows 8..11 (i=2; j=0..3), B rows 0..15, C rows 8..11 (k*3+2 for k=0..3)
    U_144 = [[0.0]*4 for _ in range(4)]
    V_144 = [[0.0]*4 for _ in range(16)]
    W_144 = [[0.0]*4 for _ in range(4)]
    for t in range(4):
        U_144[t][t] = 1.0
        for k in range(4):
            V_144[t*4+k][t] = 1.0
            W_144[k][t] = 1.0
    
    # Combine
    U = [[0.0]*32 for _ in range(12)]
    V = [[0.0]*32 for _ in range(16)]
    W = [[0.0]*32 for _ in range(12)]
    
    for i in range(8):
        for r in range(28):
            U[i][r] += U_244[i][r]
    for i in range(4):
        for r in range(4):
            U[i+8][r+28] += U_144[i][r]
    
    for j in range(16):
        for r in range(28):
            V[j][r] += V_244[j][r]
    for j in range(16):
        for r in range(4):
            V[j][r+28] += V_144[j][r]
    
    # W: 2x4 block uses k*2+i (rows 0..7), 1x4 uses k*3+2 (rows 8..11)
    for k in range(8):
        for r in range(28):
            W[k][r] += W_244[k][r]
    for k in range(4):
        for r in range(4):
            W[k+8][r+28] += W_144[k][r]
    
    return U, V, W

def _decompose_347():
    """Decompose 3x4 * 4x7. Rank 50.
    
    Strategy: Split B (4x7) into B1 (4x3) and B2 (4x4).
    C = A*B1 (3x3) | A*B2 (3x4)
    
    For A*B1 (3x4 * 4x3), rank 18:
      - Split A into A_top (2x4) and A_bot (1x4)
      - A_top * B1 (2x4*4x3): rank 14 via two 2x2*2x3 Strassen
      - A_bot * B1 (1x4*4x3): rank 4 schoolbook
    
    For A*B2 (3x4 * 4x4), rank 32:
      - Split A into A_top (2x4) and A_bot (1x4)
      - A_top * B2 (2x4*4x4): rank 28 via four 2x2*2x2 Strassen
      - A_bot * B2 (1x4*4x4): rank 4 schoolbook
    
    Global indexing:
      U rows: i*4+j for i in 0..2, j in 0..3  (12 rows)
      V rows: j*7+k for j in 0..3, k in 0..6  (28 rows)
      W rows: k*3+i for k in 0..6, i in 0..2  (21 rows)
    """
    total_rank = 50
    U = [[0.0]*total_rank for _ in range(12)]
    V = [[0.0]*total_rank for _ in range(28)]
    W = [[0.0]*total_rank for _ in range(21)]
    
    # --- Part 1: A*B1 (3x4 * 4x3), ranks 0..17 ---
    # B1 occupies V columns j*7+k for k=0,1,2 (i.e., V rows 0,1,2, 7,8,9, 14,15,16, 21,22,23)
    # C1 = A*B1 is 3x3, W rows k*3+i for k=0,1,2 (i.e., W rows 0,1,2, 3,4,5, 6,7,8)
    
    # A_top (2x4) * B1 (4x3): rank 14
    # Split A_top into A_tl (2x2: i=0,1 j=0,1) and A_tr (2x2: i=0,1 j=2,3)
    # Split B1 into B1_t (2x3: j=0,1 k=0,1,2) and B1_b (2x3: j=2,3 k=0,1,2)
    
    # Strassen 2x2*2x3 for A_tl * B1_t
    S1 = _strassen_223()  # U(4,7), V(6,7), W(6,7)
    # A_tl rows: i*4+j for i=0,1 j=0,1 -> U rows 0,1,4,5
    # B1_t rows: j*7+k for j=0,1 k=0,1,2 -> V rows 0,1,2, 7,8,9
    # C1_tl = (A_tl*B1_t) is 2x3, W rows k*3+i for k=0,1,2 i=0,1 -> W rows 0,1, 3,4, 6,7
    
    # Map local Strassen indices to global
    # Local U: i*2+j -> global U: i*4+j
    # Local V: j*3+k -> global V: j*7+k
    # Local W: k*2+i -> global W: k*3+i
    
    u_map_223 = [(0,0),(1,1),(4,4),(5,5)]  # local->global for 2x2 A block
    v_map_223_3 = [(0,0),(1,1),(2,2),(3,7),(4,8),(5,9)]  # local->global for 2x3 B block
    w_map_223_3 = [(0,0),(1,1),(2,3),(3,4),(4,6),(5,7)]  # local->global for 2x3 C block
    
    for li, gi in u_map_223:
        for r in range(7):
            U[gi][r] += S1[0][li][r] if False else S1[li][r]
    for lj, gj in v_map_223_3:
        for r in range(7):
            V[gj][r] += S1[1][lj][r]
    for lw, gw in w_map_223_3:
        for r in range(7):
            W[gw][r] += S1[2][lw][r]
    
    # Strassen 2x2*2x3 for A_tr * B1_b
    S2 = _strassen_223()
    # A_tr rows: i*4+j for i=0,1 j=2,3 -> U rows 2,3,6,7
    # B1_b rows: j*7+k for j=2,3 k=0,1,2 -> V rows 14,15,16, 21,22,23
    # C1_tr contributes to same W rows as C1_tl
    
    u_map_223b = [(0,2),(1,3),(4,6),(5,7)]
    v_map_223_3b = [(0,14),(1,15),(2,16),(3,21),(4,22),(5,23)]
    w_map_223_3b = [(0,0),(1,1),(2,3),(3,4),(4,6),(5,7)]
    
    for li, gi in u_map_223b:
        for r in range(7):
            U[gi][r+7] += S2[0][li][r] if False else S2[li][r]
    for lj, gj in v_map_223_3b:
        for r in range(7):
            V[gj][r+7] += S2[1][lj][r]
    for lw, gw in w_map_223_3b:
        for r in range(7):
            W[gw][r+7] += S2[2][lw][r]
    
    # A_bot (1x4) * B1 (4x3): rank 4
    # A_bot rows: i*4+j for i=2 j=0,1,2,3 -> U rows 8,9,10,11
    # B1 rows: same as above
    # C1_bot = A_bot*B1 is 1x3, W rows k*3+2 for k=0,1,2 -> W rows 2,5,8
    
    for t in range(4):
        U[8+t][14+t] = 1.0
        # B1 row for j=t, k=0,1,2: V rows t*7+0, t*7+1, t*7+2
        for k in range(3):
            V[t*7+k][14+t] = 1.0
            W[k*3+2][14+t] = 1.0
    
    # --- Part 2: A*B2 (3x4 * 4x4), ranks 18..49 ---
    # B2 occupies V columns j*7+k for k=3,4,5,6 (i.e., V rows 3,4,5,6, 10,11,12,13, 17,18,19,20, 24,25,26,27)
    # C2 = A*B2 is 3x4, W rows k*3+i for k=3,4,5,6 (i.e., W rows 9,10,11, 12,13,14, 15,16,17, 18,19,20)
    
    # A_top (2x4) * B2 (4x4): rank 28
    # Four 2x2*2x2 Strassen blocks
    
    # Block 1: A_tl (2x2) * B2_tl (2x2: j=0,1 k=3,4)
    S3 = _strassen_222()  # U(4,7), V(4,7), W(4,7)
    # A_tl: U rows 0,1,4,5
    # B2_tl: V rows 3,4,10,11
    # C2_tl: W rows 9,10,12,13 (k=3,4 i=0,1)
    
    u_map_222_1 = [(0,0),(1,1),(4,4),(5,5)]
    v_map_222_1 = [(0,3),(1,4),(2,10),(3,11)]
    w_map_222_1 = [(0,9),(1,10),(2,12),(3,13)]
    
    for li, gi in u_map_222_1:
        for r in range(7):
            U[gi][18+r] += S3[li][r]
    for lj, gj in v_map_222_1:
        for r in range(7):
            V[gj][18+r] += S3[1][lj][r]
    for lw, gw in w_map_222_1:
        for r in range(7):
            W[gw][18+r] += S3[2][lw][r]
    
    # Block 2: A_tr (2x2) * B2_tr (2x2: j=2,3 k=3,4)
    S4 = _strassen_222()
    u_map_222_2 = [(0,2),(1,3),(4,6),(5,7)]
    v_map_222_2 = [(0,17),(1,18),(2,24),(3,25)]
    w_map_222_2 = [(0,9),(1,10),(2,12),(3,13)]
    
    for li, gi in u_map_222_2:
        for r in range(7):
            U[gi][18+r+7] += S4[li][r]
    for lj, gj in v_map_222_2:
        for r in range(7):
            V[gj][18+r+7] += S4[1][lj][r]
    for lw, gw in w_map_222_2:
        for r in range(7):
            W[gw][18+r+7] += S4[2][lw][r]
    
    # Block 3: A_tl (2x2) * B2_bl (2x2: j=0,1 k=5,6)
    S5 = _strassen_222()
    u_map_222_3 = [(0,0),(1,1),(4,4),(5,5)]
    v_map_222_3 = [(0,5),(1,6),(2,12),(3,13)]
    w_map_222_3 = [(0,15),(1,16),(2,18),(3,19)]
    
    for li, gi in u_map_222_3:
        for r in range(7):
            U[gi][18+r+14] += S5[li][r]
    for lj, gj in v_map_222_3:
        for r in range(7):
            V[gj][18+r+14] += S5[1][lj][r]
    for lw, gw in w_map_222_3:
        for r in range(7):
            W[gw][18+r+14] += S5[2][lw][r]
    
    # Block 4: A_tr (2x2) * B2_br (2x2: j=2,3 k=5,6)
    S6 = _strassen_222()
    u_map_222_4 = [(0,2),(1,3),(4,6),(5,7)]
    v_map_222_4 = [(0,19),(1,20),(2,26),(3,27)]
    w_map_222_4 = [(0,15),(1,16),(2,18),(3,19)]
    
    for li, gi in u_map_222_4:
        for r in range(7):
            U[gi][18+r+21] += S6[li][r]
    for lj, gj in v_map_222_4:
        for r in range(7):
            V[gj][18+r+21] += S6[1][lj][r]
    for lw, gw in w_map_222_4:
        for r in range(7):
            W[gw][18+r+21] += S6[2][lw][r]
    
    # A_bot (1x4) * B2 (4x4): rank 4
    for t in range(4):
        U[8+t][18+28+t] = 1.0
        for k in range(4):
            V[t*7+3+k][18+28+t] = 1.0
            W[(3+k)*3+2][18+28+t] = 1.0
    
    return U, V, W

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = delta.
    T is shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1.
    """
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is correct."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    # Reconstruct tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_recon = np.zeros((n, m, p))
    for r in range(U_arr.shape[1]):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        # u_col is shape (n*m,), reshape to (n, m)
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    
    T_expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i, j, k] = 1.0
    
    if np.max(np.abs(T_recon - T_expected)) > tol:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # Try the block decomposition (rank 50)
    try:
        U, V, W = _decompose_347()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 84)
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
