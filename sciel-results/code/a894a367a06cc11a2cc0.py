"""Rewritten program for matrix multiplication 3,4,8 using rank-88 decomposition."""
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
    # M1 = (A00+A11)(B00+B11), M2 = (A10+A11)B00, M3 = A00(B10-B11),
    # M4 = A11(B00-B10), M5 = (A00+A01)B11, M6 = (A10-A00)(B00+B01),
    # M7 = (A01-A11)(B10+B11)
    sU = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    sV = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    sW = [
        [1, 0, 0, 1],  # M1: C00+=1, C11+=1
        [0, 0, 1, -1], # M2: C10+=1, C11-=1
        [0, 1, 0, 1],  # M3: C01+=1, C11+=1
        [1, 0, 1, 0],  # M4: C00+=1, C10+=1
        [-1, 1, 0, 0], # M5: C00-=1, C01+=1
        [0, 0, 0, 1],  # M6: C11+=1
        [1, 0, 0, 0]   # M7: C00+=1
    ]
    
    # Part 1: i=0, schoolbook for 1x4x8 (32 terms)
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: i=1,2, Strassen for 2x4x8
    # Split j into [0,1] and [2,3], split k into [0,1],[2,3],[4,5],[6,7]
    for jb in range(2):  # j block
        j0 = jb * 2
        for kb in range(4):  # k block
            k0 = kb * 2
            for t in range(7):
                # U: local (li, lj) -> global (i=1+li, j=j0+lj)
                for li in range(2):
                    for lj in range(2):
                        l = li * 2 + lj
                        val = sU[t][l]
                        if val != 0:
                            U[(1 + li) * m + (j0 + lj)][col] += val
                # V: local (lj, lk) -> global (j=j0+lj, k=k0+lk)
                for lj in range(2):
                    for lk in range(2):
                        l = lj * 2 + lk
                        val = sV[t][l]
                        if val != 0:
                            V[(j0 + lj) * p + (k0 + lk)][col] += val
                # W: local (li, lk) -> global (k=k0+lk, i=1+li)
                for li in range(2):
                    for lk in range(2):
                        l = li * 2 + lk
                        val = sW[t][l]
                        if val != 0:
                            W[(k0 + lk) * n + (1 + li)][col] += val
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
