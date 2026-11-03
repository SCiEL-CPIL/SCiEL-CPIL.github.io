import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

def _validate(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    R = len(U[0]) if U else 0
    if len(V[0]) != R or len(W[0]) != R:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _strassen_2x2():
    """Return U, V, W for rank-7 Strassen decomposition of 2x2x2."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = -1
    
    # r=3: M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[1][6] = 1
    
    return U, V, W

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook fallback: rank 36
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Build a rank-30 decomposition using Strassen for 2x2*2x2 blocks
    # and schoolbook for smaller blocks.
    #
    # A is 3x3, B is 3x4, C is 3x4
    # Block A: [[A00(2x2), A01(2x1)], [A10(1x2), A11(1x1)]]
    # Block B: [[B00(2x2), B01(2x2)], [B10(1x2), B11(1x2)]]
    # Block C: [[C00(2x2), C01(2x2)], [C10(1x2), C11(1x2)]]
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Index mappings:
    # A[i,j] -> index i*3+j
    #   A00: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    #   A01: (0,2)->2, (1,2)->5
    #   A10: (2,0)->6, (2,1)->7
    #   A11: (2,2)->8
    #
    # B[j,k] -> index j*4+k
    #   B00: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    #   B01: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    #   B10: (2,0)->8, (2,1)->9
    #   B11: (2,2)->10, (2,3)->11
    #
    # C[k,i] -> index k*3+i
    #   C00: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    #   C01: (0,2)->2, (0,3)->5, (1,2)->6, (1,3)->7
    #   C10: (2,0)->8, (2,1)->9
    #   C11: (2,2)->10, (2,3)->11
    
    R = 0
    U = [[0]*30 for _ in range(n*m)]
    V = [[0]*30 for _ in range(m*p)]
    W = [[0]*30 for _ in range(p*n)]
    
    sU, sV, sW = _strassen_2x2()
    
    # A00 local indices: [0,1,3,4] corresponding to A[0,0],A[0,1],A[1,0],A[1,1]
    # B00 local indices: [0,1,4,5] corresponding to B[0,0],B[0,1],B[1,0],B[1,1]
    # C00 local indices: [0,1,3,4] corresponding to C[0,0],C[0,1],C[1,0],C[1,1]
    # Strassen U/V/W use local 2x2 indices: 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)
    
    # C00 = A00*B00 (Strassen) + A01*B10 (schoolbook)
    # A00*B00:
    a00_idx = [0,1,3,4]  # A[0,0], A[0,1], A[1,0], A[1,1]
    b00_idx = [0,1,4,5]  # B[0,0], B[0,1], B[1,0], B[1,1]
    c00_idx = [0,1,3,4]  # C[0,0], C[0,1], C[1,0], C[1,1]
    
    for r in range(7):
        for li in range(4):
            if sU[li][r] != 0:
                U[a00_idx[li]][R] = sU[li][r]
        for lj in range(4):
            if sV[lj][r] != 0:
                V[b00_idx[lj]][R] = sV[lj][r]
        for lk in range(4):
            if sW[lk][r] != 0:
                W[c00_idx[lk]][R] = sW[lk][r]
        R += 1
    
    # A01*B10: 4 schoolbook terms
    # A01: A[0,2]=2, A[1,2]=5
    # B10: B[2,0]=8, B[2,1]=9
    # C00: C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4
    # A[0,2]*B[2,0] -> C[0,0]
    # A[0,2]*B[2,1] -> C[0,1]
    # A[1,2]*B[2,0] -> C[1,0]
    # A[1,2]*B[2,1] -> C[1,1]
    terms_a01_b10 = [(2,8,0), (2,9,1), (5,8,3), (5,9,4)]
    for (a, b, c) in terms_a01_b10:
        U[a][R] = 1
        V[b][R] = 1
        W[c][R] = 1
        R += 1
    
    # C01 = A00*B01 (Strassen) + A01*B11 (schoolbook)
    # A00*B01:
    a00_idx = [0,1,3,4]
    b01_idx = [2,3,6,7]  # B[0,2], B[0,3], B[1,2], B[1,3]
    c01_idx = [2,5,6,7]  # C[0,2], C[0,3], C[1,2], C[1,3]
    
    for r in range(7):
        for li in range(4):
            if sU[li][r] != 0:
                U[a00_idx[li]][R] = sU[li][r]
        for lj in range(4):
            if sV[lj][r] != 0:
                V[b01_idx[lj]][R] = sV[lj][r]
        for lk in range(4):
            if sW[lk][r] != 0:
                W[c01_idx[lk]][R] = sW[lk][r]
        R += 1
    
    # A01*B11: 4 schoolbook terms
    # A01: A[0,2]=2, A[1,2]=5
    # B11: B[2,2]=10, B[2,3]=11
    # C01: C[0,2]=2, C[0,3]=5, C[1,2]=6, C[1,3]=7
    # A[0,2]*B[2,2] -> C[0,2]
    # A[0,2]*B[2,3] -> C[0,3]
    # A[1,2]*B[2,2] -> C[1,2]
    # A[1,2]*B[2,3] -> C[1,3]
    terms_a01_b11 = [(2,10,2), (2,11,5), (5,10,6), (5,11,7)]
    for (a, b, c) in terms_a01_b11:
        U[a][R] = 1
        V[b][R] = 1
        W[c][R] = 1
        R += 1
    
    # C10 = A10*B00 (schoolbook) + A11*B10 (schoolbook)
    # A10: A[2,0]=6, A[2,1]=7
    # B00: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # C10: C[2,0]=8, C[2,1]=9
    # A[2,0]*B[0,0] -> C[2,0]
    # A[2,0]*B[1,0] -> C[2,0]  (wait, need to check)
    # Actually, C[k,i] = sum_j A[i,j]*B[j,k]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # So A10*B00 contributes:
    # A[2,0]*B[0,0] -> C[2,0]
    # A[2,0]*B[0,1] -> C[2,1]
    # A[2,1]*B[1,0] -> C[2,0]
    # A[2,1]*B[1,1] -> C[2,1]
    terms_a10_b00 = [(6,0,8), (6,1,9), (7,4,8), (7,5,9)]
    for (a, b, c) in terms_a10_b00:
        U[a][R] = 1
        V[b][R] = 1
        W[c][R] = 1
        R += 1
    
    # A11*B10: 2 schoolbook terms
    # A11: A[2,2]=8
    # B10: B[2,0]=8, B[2,1]=9
    # C10: C[2,0]=8, C[2,1]=9
    # A[2,2]*B[2,0] -> C[2,0]
    # A[2,2]*B[2,1] -> C[2,1]
    terms_a11_b10 = [(8,8,8), (8,9,9)]
    for (a, b, c) in terms_a11_b10:
        U[a][R] = 1
        V[b][R] = 1
        W[c][R] = 1
        R += 1
    
    # C11 = A10*B01 (schoolbook) + A11*B11 (schoolbook)
    # A10: A[2,0]=6, A[2,1]=7
    # B01: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # C11: C[2,2]=10, C[2,3]=11
    # A[2,0]*B[0,2] -> C[2,2]
    # A[2,0]*B[0,3] -> C[2,3]
    # A[2,1]*B[1,2] -> C[2,2]
    # A[2,1]*B[1,3] -> C[2,3]
    terms_a10_b01 = [(6,2,10), (6,3,11), (7,6,10), (7,7,11)]
    for (a, b, c) in terms_a10_b01:
        U[a][R] = 1
        V[b][R] = 1
        W[c][R] = 1
        R += 1
    
    # A11*B11: 2 schoolbook terms
    # A11: A[2,2]=8
    # B11: B[2,2]=10, B[2,3]=11
    # C11: C[2,2]=10, C[2,3]=11
    # A[2,2]*B[2,2] -> C[2,2]
    # A[2,2]*B[2,3] -> C[2,3]
    terms_a11_b11 = [(8,10,10), (8,11,11)]
    for (a, b, c) in terms_a11_b11:
        U[a][R] = 1
        V[b][R] = 1
        W[c][R] = 1
        R += 1
    
    # Total rank should be 7+4+7+4+4+2+4+2 = 34
    # Wait, let me recount:
    # C00: 7 (Strassen) + 4 (schoolbook) = 11
    # C01: 7 (Strassen) + 4 (schoolbook) = 11
    # C10: 4 (schoolbook) + 2 (schoolbook) = 6
    # C11: 4 (schoolbook) + 2 (schoolbook) = 6
    # Total: 34
    
    # That's still less than 36! Let me verify.
    
    # Trim to actual R
    U_final = [row[:R] for row in U]
    V_final = [row[:R] for row in V]
    W_final = [row[:R] for row in W]
    
    result = {"U": U_final, "V": V_final, "W": W_final}
    
    if _validate(U_final, V_final, W_final, n, m, p):
        if R < best_rank:
            best_rank = R
            best_result = result
    
    return best_result
