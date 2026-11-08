"""Program for matrix multiplication 3,4,8 using rank-88 decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 factors
    sU = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    sV = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    sW = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Part 1: i=0, schoolbook for 1x4x8 (32 terms)
    for j in range(4):
        for k in range(8):
            U[j][col] = 1
            V[j * p + k][col] = 1
            W[k * n][col] = 1
            col += 1
    
    # Part 2: i=1,2, Strassen for 2x4x8
    for jb in range(2):
        j0 = jb * 2
        for kb in range(4):
            k0 = kb * 2
            for t in range(7):
                for li in range(2):
                    for lj in range(2):
                        l = li * 2 + lj
                        val = sU[t][l]
                        if val != 0:
                            U[(1 + li) * m + (j0 + lj)][col] += val
                for lj in range(2):
                    for lk in range(2):
                        l = lj * 2 + lk
                        val = sV[t][l]
                        if val != 0:
                            V[(j0 + lj) * p + (k0 + lk)][col] += val
                for li in range(2):
                    for lk in range(2):
                        l = li * 2 + lk
                        val = sW[t][l]
                        if val != 0:
                            W[(k0 + lk) * n + (1 + li)][col] += val
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
