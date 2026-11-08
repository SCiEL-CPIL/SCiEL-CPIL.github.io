"""
Program for matrix multiplication 3,4,8.
Uses a rank-88 decomposition via block partitioning:
- 1x4x8 slice (i=0): rank 32 (trivial)
- 2x4x8 slice (i=1,2): rank 56 (Strassen on 2x2x2 blocks)
Total: 88
"""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    R = 88

    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]

    c = 0

    # Part A: i=0 slice, trivial rank-32
    for j in range(m):
        for k in range(p):
            U[j][c] = 1.0
            V[j * p + k][c] = 1.0
            W[k * n][c] = 1.0
            c += 1

    # Part B: i=1,2 slice via Strassen on 2x2x2 sub-blocks
    # Strassen 2x2x2 factors (local index l: 0=(0,0),1=(0,1),2=(1,0),3=(1,1))
    SU = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]]
    SV = [[1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],[0,0,0,1],[1,1,0,0],[0,0,1,1]]
    SW = [[1,0,0,1],[0,0,1,-1],[0,1,0,1],[1,0,1,0],[-1,1,0,0],[0,0,0,1],[1,0,0,0]]

    for jb in range(2):
        jo = jb * 2
        for kb in range(4):
            ko = kb * 2
            for t in range(7):
                for l in range(4):
                    val = SU[t][l]
                    if val:
                        gi = 1 + l // 2
                        gj = jo + l % 2
                        U[gi * m + gj][c] += float(val)
                for l in range(4):
                    val = SV[t][l]
                    if val:
                        gj = jo + l // 2
                        gk = ko + l % 2
                        V[gj * p + gk][c] += float(val)
                for l in range(4):
                    val = SW[t][l]
                    if val:
                        gi = 1 + l // 2
                        gk = ko + l % 2
                        W[gk * n + gi][c] += float(val)
                c += 1
                if time.time() - t0 > 145:
                    return {"U": U, "V": V, "W": W}
            if time.time() - t0 > 145:
                return {"U": U, "V": V, "W": W}
        if time.time() - t0 > 145:
            return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
