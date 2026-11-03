import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    deadline = time.time() + 140
    
    # Rank 56 solution using Strassen for 2x2x2 subproblems
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    for s in range(2):  # s=0,1: split A columns / B rows into 2x2 blocks
        for j in range(4):  # j=0..3: split B columns into 2x2 blocks
            
            # A_s: rows 0-1, cols 2s, 2s+1 of A (2x4)
            # U index: i*m + j
            a00 = 0 * m + (2*s + 0)
            a01 = 0 * m + (2*s + 1)
            a10 = 1 * m + (2*s + 0)
            a11 = 1 * m + (2*s + 1)
            
            # B_{s,j}: rows 2s, 2s+1; cols 2j, 2j+1 of B (4x8)
            # V index: j*p + k
            b00 = (2*s + 0) * p + (2*j + 0)
            b01 = (2*s + 0) * p + (2*j + 1)
            b10 = (2*s + 1) * p + (2*j + 0)
            b11 = (2*s + 1) * p + (2*j + 1)
            
            # C block: rows 0-1, cols 2j, 2j+1
            # W index: k*n + i (trace convention)
            c00 = (2*j + 0) * n + 0
            c01 = (2*j + 1) * n + 0
            c10 = (2*j + 0) * n + 1
            c11 = (2*j + 1) * n + 1
            
            # Strassen U (A-side) coefficients
            # m1 = (a00+a11)*(b00+b11)
            U[a00][col] += 1
            U[a11][col] += 1
            # m2 = (a10+a11)*b00
            U[a10][col+1] += 1
            U[a11][col+1] += 1
            # m3 = a00*(b01-b11)
            U[a00][col+2] += 1
            # m4 = a11*b10
            U[a11][col+3] += 1
            # m5 = (a00+a01)*b11
            U[a00][col+4] += 1
            U[a01][col+4] += 1
            # m6 = (a10-a00)*(b00+b01)
            U[a10][col+5] += 1
            U[a00][col+5] += -1
            # m7 = (a01-a11)*(b10+b11)
            U[a01][col+6] += 1
            U[a11][col+6] += -1
            
            # Strassen V (B-side) coefficients
            V[b00][col] += 1
            V[b11][col] += 1
            V[b00][col+1] += 1
            V[b01][col+2] += 1
            V[b11][col+2] += -1
            V[b10][col+3] += 1
            V[b00][col+3] += -1
            V[b11][col+4] += 1
            V[b00][col+5] += 1
            V[b01][col+5] += 1
            V[b10][col+6] += 1
            V[b11][col+6] += 1
            
            # Strassen W (C-side) coefficients
            # c00 = m1 + m4 - m5 + m7
            W[c00][col] += 1
            W[c00][col+3] += 1
            W[c00][col+4] += -1
            W[c00][col+6] += 1
            # c01 = m3 + m5
            W[c01][col+2] += 1
            W[c01][col+4] += 1
            # c10 = m2 + m4
            W[c10][col+1] += 1
            W[c10][col+3] += 1
            # c11 = m1 - m2 + m3 + m6
            W[c11][col] += 1
            W[c11][col+1] += -1
            W[c11][col+2] += 1
            W[c11][col+5] += 1
            
            col += 7
            
    if time.time() > deadline:
        # Fallback to schoolbook if somehow this takes too long (it won't)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
