import numpy as np
import time

def construct():
    t0 = time.time()
    tlimit = 110

    n, m, p = 4, 5, 5
    total_r = 79  # 49 + 16 + 14

    U = [[0] * total_r for _ in range(n * m)]
    V = [[0] * total_r for _ in range(m * p)]
    W = [[0] * total_r for _ in range(p * n)]

    # 2x2x2 Strassen basis
    sU = [[0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0]]
    sV = [[0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0]]
    sW = [[0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0]]
    sU[0][0]=1; sU[3][0]=1; sV[0][0]=1; sV[3][0]=1; sW[0][0]=1; sW[3][0]=1
    sU[2][1]=1; sU[3][1]=1; sV[0][1]=1; sW[2][1]=1; sW[3][1]=-1
    sU[0][2]=1; sV[1][2]=1; sV[3][2]=-1; sW[1][2]=1; sW[3][2]=1
    sU[3][3]=1; sV[2][3]=1; sV[0][3]=-1; sW[0][3]=1; sW[2][3]=1
    sU[0][4]=1; sU[1][4]=1; sV[3][4]=1; sW[0][4]=-1; sW[1][4]=1
    sU[0][5]=-1; sU[2][5]=1; sV[0][5]=1; sV[1][5]=1; sW[3][5]=1
    sU[1][6]=1; sU[3][6]=-1; sV[2][6]=1; sV[3][6]=1; sW[0][6]=1

    # 4x4x4 via double Strassen: 7 outer x 7 inner = 49
    outer = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

    col = 0
    for ac, bc, cc in outer:
        for kk in range(7):
            c_idx = col + kk
            for (br, bc_), ca in ac.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        if sU[li][kk]:
                            U[(2*br+lr)*5 + (2*bc_+lc)][c_idx] += ca * sU[li][kk]
            for (br, bc_), cb in bc.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        if sV[li][kk]:
                            V[(2*br+lr)*5 + (2*bc_+lc)][c_idx] += cb * sV[li][kk]
            for (br, bc_), cw in cc.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        if sW[li][kk]:
                            W[(2*bc_+lc)*4 + (2*br+lr)][c_idx] += cw * sW[li][kk]
        col += 7

    # C1 += A2 * B3: rank 16
    for i in range(4):
        for k in range(4):
            U[i*5+4][col] = 1
            V[4*5+k][col] = 1
            W[k*4+i][col] = 1
            col += 1

    # C2 = A1*B2 + A2*B4 via 2x2 Strassen on row-blocks: rank 14
    # A1 rows 0-3, A2 row 4; B2 cols 0-3, B4 col 4; C2 rows 0-3
    outer_c2 = [
        ({(0,0):1,(1,0):1}, {(0,0):1,(0,1):1}, {(0,0):1,(0,1):1}),
        ({(1,0):1}, {(0,0):1}, {(0,1):1,(0,0):-1}),
        ({(0,0):1}, {(0,1):1,(0,0):-1}, {(0,0):1,(0,1):1}),
        ({(1,0):1}, {(0,0):1,(0,1):-1}, {(0,0):1,(0,1):1}),
        ({(0,0):1,(0,1):1}, {(0,0):1}, {(0,0):-1,(0,1):1}),
        ({(0,0):-1,(0,1):1}, {(0,0):1,(0,1):1}, {(0,0):1}),
        ({(0,1):1,(1,0):-1}, {(0,0):1,(0,1):1}, {(0,0):1}),
    ]
    for ac, bc, cc in outer_c2:
        for kk in range(7):
            c_idx = col + kk
            for (ar, ac_), ca in ac.items():
                if sU[ar][kk]:
                    U[ar*5 + ac_][c_idx] += ca * sU[ar][kk]
            for (br, bc_), cb in bc.items():
                if sV[br][kk]:
                    V[br*5 + bc_][c_idx] += cb * sV[br][kk]
            for (cr, cc_), cw in cc.items():
                if sW[cr][kk]:
                    W[4*4 + cr][c_idx] += cw * sW[cr][kk]
        col += 7

    if time.time() - t0 > tlimit:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
