import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 4, 5, 5

    # Block decomposition:
    # A (4×5) = [A1 (4×4) | A2 (4×1)]
    # B (5×5) = [[B1 (4×4), B2 (4×1)], [B3 (1×4), B4 (1×1)]]
    # C (4×5) = [C1 (4×4) | C2 (4×1)]
    #
    # C1 = A1*B1 + A2*B3   (rank 49 + 16 = 65)
    # C2 = A1*B2 + A2*B4   (rank 16 + 4 = 20)
    # Total rank = 85

    rank = 85
    U = [[0] * rank for _ in range(20)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(20)]

    # Strassen 2×2×2 primitive
    su = [[0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0]]
    sv = [[0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0]]
    sw = [[0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0]]

    su[0][0]=1; su[3][0]=1; sv[0][0]=1; sv[3][0]=1; sw[0][0]=1; sw[3][0]=1
    su[2][1]=1; su[3][1]=1; sv[0][1]=1; sw[2][1]=1; sw[3][1]=-1
    su[0][2]=1; sv[1][2]=1; sv[3][2]=-1; sw[1][2]=1; sw[3][2]=1
    su[3][3]=1; sv[2][3]=1; sv[0][3]=-1; sw[0][3]=1; sw[2][3]=1
    su[0][4]=1; su[1][4]=1; sv[3][4]=1; sw[0][4]=-1; sw[1][4]=1
    su[0][5]=-1; su[2][5]=1; sv[0][5]=1; sv[1][5]=1; sw[3][5]=1
    su[1][6]=1; su[3][6]=-1; sv[2][6]=1; sv[3][6]=1; sw[0][6]=1

    # 4×4×4 via 2×2 Strassen blocks
    # Outer 2×2 Strassen: 7 terms, each inner 2×2 Strassen: 7 terms = 49
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
        for k in range(7):
            ci = col + k
            for (br,bc_),ca in ac.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr*2+lc
                        if su[li][k]:
                            U[(2*br+lr)*5+(2*bc_+lc)][ci] += ca*su[li][k]
            for (br,bc_),cb in bc.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr*2+lc
                        if sv[li][k]:
                            V[(2*br+lr)*5+(2*bc_+lc)][ci] += cb*sv[li][k]
            for (br,bc_),cc_ in cc.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr*2+lc
                        if sw[li][k]:
                            W[(2*bc_+lc)*4+(2*br+lr)][ci] += cc_*sw[li][k]
        col += 7

    # C1 += A2*B3: rank 16
    for i in range(4):
        for k in range(4):
            U[i*5+4][col] = 1
            V[4*5+k][col] = 1
            W[k*4+i][col] = 1
            col += 1

    # C2 += A1*B2: rank 16
    for i in range(4):
        for j in range(4):
            U[i*5+j][col] = 1
            V[j*5+4][col] = 1
            W[4*4+i][col] = 1
            col += 1

    # C2 += A2*B4: rank 4
    for i in range(4):
        U[i*5+4][col] = 1
        V[4*5+4][col] = 1
        W[4*4+i][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
