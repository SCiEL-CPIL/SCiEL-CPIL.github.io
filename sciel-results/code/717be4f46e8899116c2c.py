import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    def check_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    # Build rank-17 decomposition using Strassen hybrid
    rank = 17
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]

    # Strassen 2x2x2 core: i in {0,1}, j in {0,1}, k in {0,1}
    # Using standard Strassen formulas (0-indexed):
    # M1 = (A[1][1]-A[0][0])*(B[1][0]+B[1][1])
    # M2 = (A[1][0]+A[1][1])*B[0][0]
    # M3 = (A[0][0]+A[0][1])*B[1][1]
    # M4 = (A[1][1]-A[0][0])*(B[0][0]+B[0][1])
    # M5 = (A[0][0]+A[1][0])*(B[0][0]+B[0][1])
    # M6 = (A[0][1]-A[1][1])*(B[1][0]+B[1][1])
    # M7 = (A[1][0]-A[0][0])*(B[0][0]+B[0][1])
    # C[0][0] = M1 + M4 - M5 + M7
    # C[0][1] = M3 + M5
    # C[1][0] = M2 + M4
    # C[1][1] = M1 - M2 + M3 + M6

    c = 0
    # M1
    U[4][c] += 1; U[0][c] -= 1  # A[1,1] - A[0,0]
    V[3][c] += 1; V[4][c] += 1  # B[1,0] + B[1,1]
    W[0][c] += 1; W[3][c] += 1  # C[0,0] + C[1,1]
    c += 1

    # M2
    U[3][c] += 1; U[4][c] += 1  # A[1,0] + A[1,1]
    V[0][c] += 1  # B[0,0]
    W[1][c] += 1  # C[1,0]
    c += 1

    # M3
    U[0][c] += 1; U[1][c] += 1  # A[0,0] + A[0,1]
    V[4][c] += 1  # B[1,1]
    W[2][c] += 1; W[3][c] += 1  # C[0,1] + C[1,1]
    c += 1

    # M4
    U[4][c] += 1; U[0][c] -= 1  # A[1,1] - A[0,0]
    V[0][c] += 1; V[1][c] += 1  # B[0,0] + B[0,1]
    W[0][c] += 1; W[1][c] += 1  # C[0,0] + C[1,0]
    c += 1

    # M5
    U[0][c] += 1; U[3][c] += 1  # A[0,0] + A[1,0]
    V[0][c] += 1; V[1][c] += 1  # B[0,0] + B[0,1]
    W[2][c] += 1; W[3][c] += 1  # C[0,1] + C[1,1]
    c += 1

    # M6
    U[1][c] += 1; U[4][c] -= 1  # A[0,1] - A[1,1]
    V[3][c] += 1; V[4][c] += 1  # B[1,0] + B[1,1]
    W[3][c] += 1  # C[1,1]
    c += 1

    # M7
    U[3][c] += 1; U[0][c] -= 1  # A[1,0] - A[0,0]
    V[0][c] += 1; V[1][c] += 1  # B[0,0] + B[0,1]
    W[0][c] += 1  # C[0,0]
    c += 1

    # j=2 terms: (i, 2, k) for i in {0,1}, k in {0,1,2}
    for k in range(3):
        U[2][c] += 1  # i=0, j=2
        U[5][c] += 1  # i=1, j=2
        V[6+k][c] += 1  # j=2, k
        W[k*2+0][c] += 1  # k, i=0
        W[k*2+1][c] += 1  # k, i=1
        c += 1

    # (i, j, 2) for j in {0,1}, i in {0,1}
    for j in range(2):
        U[j][c] += 1  # i=0, j
        U[3+j][c] += 1  # i=1, j
        V[j*3+2][c] += 1  # j, k=2
        W[4][c] += 1  # k=2, i=0
        W[5][c] += 1  # k=2, i=1
        c += 1

    # Verify
    if c <= rank and check_decomposition(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    rank_schoolbook = n * m * p
    U_s = [[0]*rank_schoolbook for _ in range(n*m)]
    V_s = [[0]*rank_schoolbook for _ in range(m*p)]
    W_s = [[0]*rank_schoolbook for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    return {"U": U_s, "V": V_s, "W": W_s}
