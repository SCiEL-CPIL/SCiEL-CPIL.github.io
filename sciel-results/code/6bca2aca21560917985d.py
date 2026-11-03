"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen 2x2 coefficients
    alpha = [
        {(0,0):1, (1,1):1},           # M1
        {(1,0):1, (1,1):1},           # M2
        {(0,0):1},                    # M3
        {(1,1):1},                    # M4
        {(0,0):1, (0,1):1},           # M5
        {(1,0):1, (0,0):-1},          # M6
        {(0,1):1, (1,1):-1},          # M7
    ]
    
    beta = [
        {(0,0):1, (1,1):1},           # M1
        {(0,0):1},                    # M2
        {(0,1):1, (1,1):-1},          # M3
        {(1,0):1, (0,0):-1},          # M4
        {(1,1):1},                    # M5
        {(0,0):1, (0,1):1},           # M6
        {(1,0):1, (1,1):1},           # M7
    ]
    
    coeff_C = {
        (0,0): {0:1, 3:1, 4:-1, 6:1},   # C11 = M1+M4-M5+M7
        (0,1): {2:1, 4:1},               # C12 = M3+M5
        (1,0): {1:1, 3:1},               # C21 = M2+M4
        (1,1): {0:1, 1:-1, 2:1, 5:1},   # C22 = M1-M2+M3+M6
    }
    
    col = 0
    
    # First half: B cols 0-3
    for t in range(7):
        for a in range(2):
            for b in range(2):
                for d in range(2):
                    if time.time() > deadline:
                        break
                    for (br, bc), ca in alpha[t].items():
                        row = br * 2 + a
                        colA = bc * 2 + b
                        U[row * m + colA][col] += ca
                    for (br, bc), cb in beta[t].items():
                        rowB = br * 2 + b
                        colB = bc * 2 + d
                        V[rowB * m + colB][col] += cb
                    for (br, bc), cc in coeff_C.items():
                        if t in cc:
                            rowC = br * 2 + a
                            colC = bc * 2 + d
                            W[colC * n + rowC][col] += cc[t]
                    col += 1
    
    # Second half: B cols 4-7
    for t in range(7):
        for a in range(2):
            for b in range(2):
                for d in range(2):
                    if time.time() > deadline:
                        break
                    for (br, bc), ca in alpha[t].items():
                        row = br * 2 + a
                        colA = bc * 2 + b
                        U[row * m + colA][col] += ca
                    for (br, bc), cb in beta[t].items():
                        rowB = br * 2 + b
                        colB = bc * 2 + d + 4
                        V[rowB * m + colB][col] += cb
                    for (br, bc), cc in coeff_C.items():
                        if t in cc:
                            rowC = br * 2 + a
                            colC = bc * 2 + d + 4
                            W[colC * n + rowC][col] += cc[t]
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
