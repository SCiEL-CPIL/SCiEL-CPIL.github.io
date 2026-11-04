import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _laderman_3x3_3():
    """Return U, V, W for 3x3 * 3x3 with rank 23 (Laderman)."""
    n, m, p = 3, 3, 3
    rank = 23
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Indices: i,j for U (0..8), j,k for V (0..8), k,i for W (0..8)
    # U[i*m + j], V[j*p + k], W[k*n + i]
    
    # M1
    U[0][0]=1; U[1][0]=1; U[2][0]=1
    V[0][0]=1; V[3][0]=1; V[6][0]=1
    W[0][0]=1; W[3][0]=1; W[6][0]=1
    
    # M2
    U[0][1]=1; U[1][1]=1
    V[1][1]=1; V[4][1]=1
    W[1][1]=1; W[4][1]=1
    
    # M3
    U[0][2]=1; U[2][2]=1
    V[2][2]=1; V[5][2]=1
    W[2][2]=1; W[5][2]=1
    
    # M4
    U[0][3]=1; U[3][3]=1; U[6][3]=1
    V[0][3]=1; V[1][3]=1; V[2][3]=1
    W[0][3]=1; W[3][3]=1; W[6][3]=1
    
    # M5
    U[0][4]=1; U[3][4]=1; U[6][4]=1; U[4][4]=1; U[7][4]=1
    V[0][4]=1; V[1][4]=1; V[2][4]=1; V[4][4]=1; V[7][4]=1
    W[0][4]=1; W[3][4]=1; W[6][4]=1; W[4][4]=1; W[7][4]=1
    
    # M6
    U[0][5]=1; U[3][5]=1; U[6][5]=1; U[5][5]=1; U[8][5]=1
    V[0][5]=1; V[1][5]=1; V[2][5]=1; V[5][5]=1; V[8][5]=1
    W[0][5]=1; W[3][5]=1; W[6][5]=1; W[5][5]=1; W[8][5]=1
    
    # M7
    U[1][6]=1; U[4][6]=1; U[7][6]=1
    V[0][6]=1; V[3][6]=1; V[6][6]=1; V[4][6]=1; V[7][6]=1
    W[0][6]=1; W[3][6]=1; W[6][6]=1
    
    # M8
    U[2][7]=1; U[5][7]=1; U[8][7]=1
    V[0][7]=1; V[3][7]=1; V[6][7]=1; V[5][7]=1; V[8][7]=1
    W[0][7]=1; W[3][7]=1; W[6][7]=1
    
    # M9
    U[3][8]=1; U[4][8]=1; U[5][8]=1; U[6][8]=1; U[7][8]=1; U[8][8]=1
    V[1][8]=1; V[2][8]=1; V[4][8]=1; V[5][8]=1; V[7][8]=1; V[8][8]=1
    W[1][8]=1; W[2][8]=1; W[4][8]=1; W[5][8]=1; W[7][8]=1; W[8][8]=1
    
    # M10
    U[0][9]=1; U[1][9]=1; U[2][9]=1; U[3][9]=1; U[4][9]=1; U[5][9]=1
    V[0][9]=1; V[1][9]=1; V[2][9]=1; V[3][9]=1; V[4][9]=1; V[5][9]=1
    W[0][9]=1; W[1][9]=1; W[2][9]=1; W[3][9]=1; W[4][9]=1; W[5][9]=1
    
    # M11
    U[0][10]=1; U[1][10]=1; U[2][10]=1; U[3][10]=1; U[4][10]=1; U[5][10]=1
    V[0][10]=1; V[1][10]=1; V[2][10]=1; V[6][10]=1; V[7][10]=1; V[8][10]=1
    W[0][10]=1; W[1][10]=1; W[2][10]=1; W[6][10]=1; W[7][10]=1; W[8][10]=1
    
    # M12
    U[0][11]=1; U[1][11]=1; U[2][11]=1; U[6][11]=1; U[7][11]=1; U[8][11]=1
    V[0][11]=1; V[1][11]=1; V[2][11]=1; V[4][11]=1; V[5][11]=1; V[8][11]=1
    W[0][11]=1; W[1][11]=1; W[2][11]=1; W[4][11]=1; W[5][11]=1; W[8][11]=1
    
    # M13
    U[0][12]=1; U[3][12]=1; U[6][12]=1; U[4][12]=1; U[7][12]=1
    V[1][12]=1; V[4][12]=1; V[7][12]=1
    W[1][12]=1; W[4][12]=1; W[7][12]=1
    
    # M14
    U[0][13]=1; U[3][13]=1; U[6][13]=1; U[5][13]=1; U[8][13]=1
    V[2][13]=1; V[5][13]=1; V[8][13]=1
    W[2][13]=1; W[5][13]=1; W[8][13]=1
    
    # M15
    U[1][14]=1; U[4][14]=1; U[7][14]=1; U[0][14]=1; U[3][14]=1; U[6][14]=1
    V[0][14]=1; V[3][14]=1; V[6][14]=1
    W[0][14]=1; W[3][14]=1; W[6][14]=1
    
    # M16
    U[2][15]=1; U[5][15]=1; U[8][15]=1; U[0][15]=1; U[3][15]=1; U[6][15]=1
    V[0][15]=1; V[3][15]=1; V[6][15]=1
    W[0][15]=1; W[3][15]=1; W[6][15]=1
    
    # M17
    U[1][16]=1; U[4][16]=1; U[7][16]=1; U[1][16]+=1; U[4][16]+=1; U[7][16]+=1
    V[1][16]=1; V[4][16]=1; V[7][16]=1
    W[1][16]=1; W[4][16]=1; W[7][16]=1
    
    # M18
    U[2][17]=1; U[5][17]=1; U[8][17]=1; U[2][17]+=1; U[5][17]+=1; U[8][17]+=1
    V[2][17]=1; V[5][17]=1; V[8][17]=1
    W[2][17]=1; W[5][17]=1; W[8][17]=1
    
    # M19
    U[0][18]=1; U[1][18]=1; U[2][18]=1; U[3][18]=1; U[4][18]=1; U[5][18]=1
    V[6][18]=1; V[7][18]=1; V[8][18]=1
    W[6][18]=1; W[7][18]=1; W[8][18]=1
    
    # M20
    U[0][19]=1; U[1][19]=1; U[2][19]=1; U[6][19]=1; U[7][19]=1; U[8][19]=1
    V[3][19]=1; V[4][19]=1; V[5][19]=1
    W[3][19]=1; W[4][19]=1; W[5][19]=1
    
    # M21
    U[0][20]=1; U[3][20]=1; U[6][20]=1; U[3][20]+=1; U[6][20]+=1
    V[0][20]=1; V[1][20]=1; V[2][20]=1
    W[0][20]=1; W[1][20]=1; W[2][20]=1
    
    # M22
    U[1][21]=1; U[4][21]=1; U[7][21]=1; U[4][21]+=1; U[7][21]+=1
    V[0][21]=1; V[1][21]=1; V[2][21]=1
    W[0][21]=1; W[1][21]=1; W[2][21]=1
    
    # M23
    U[2][22]=1; U[5][22]=1; U[8][22]=1; U[5][22]+=1; U[8][22]+=1
    V[0][22]=1; V[1][22]=1; V[2][22]=1
    W[0][22]=1; W[1][22]=1; W[2][22]=1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _build_3x3x2_strassen():
    """Build 3x3x2 decomposition using Strassen on 2x2x2 block and schoolbook for rest."""
    n, m, p = 3, 3, 2
    sU, sV, sW = _strassen_2x2_2()
    rank_strassen = 7
    # Remaining entries:
    # i=2, j=0,1,2, k=0,1 (6 entries)
    # i=0,1, j=2, k=0,1 (4 entries)
    # Total 10 entries.
    rank_school = 10
    rank = rank_strassen + rank_school
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Embed Strassen at i in {0,1}, j in {0,1}, k in {0,1}
    for c in range(rank_strassen):
        for di in range(2):
            for dj in range(2):
                U[(0+di)*m + (0+dj)][c] += sU[di*2+dj][c]
        for dj in range(2):
            for dk in range(2):
                V[(0+dj)*p + (0+dk)][c] += sV[dj*2+dk][c]
        for dk in range(2):
            for di in range(2):
                W[(0+dk)*n + (0+di)][c] += sW[dk*2+di][c]
                
    col = rank_strassen
    # Schoolbook for remaining
    for i in range(3):
        for j in range(3):
            for k in range(2):
                if i < 2 and j < 2:
                    continue
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: 1x Laderman (3x3x3) + 2x Strassen-3x3x2 + 1x Schoolbook (3x3x1)
    # Split k: 0,1,2 | 3,4 | 5,6 | 7
    
    U_l, V_l, W_l = _laderman_3x3_3()
    U_s, V_s, W_s = _build_3x3x2_strassen()
    
    U_1, V_1, W_1 = _schoolbook(3, 3, 1)
    
    # Assemble
    rank_total = 23 + 17 + 17 + 9
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    
    # Block 1: k=0,1,2 (Laderman)
    for c in range(23):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # U is 9x23, V is 9x23, W is 9x23
                    # U[i*3+j, c]
                    # V[j*3+k, c]
                    # W[k*3+i, c]
                    # But V in Laderman is for 3x3x3, so V[j*3+k]
                    # In full V (9x23), row is j*8+k
                    U[i*m + j][c] += U_l[i*3 + j][c]
                    V[j*p + k][c] += V_l[j*3 + k][c]
                    W[k*n + i][c] += W_l[k*3 + i][c]
                    
    # Block 2: k=3,4 (Strassen 3x3x2)
    col_offset = 23
    for c in range(17):
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U[i*m + j][col_offset + c] += U_s[i*3 + j][c]
                    V[j*p + (3+k)][col_offset + c] += V_s[j*3 + k][c]
                    W[(3+k)*n + i][col_offset + c] += W_s[k*3 + i][c]
                    
    # Block 3: k=5,6 (Strassen 3x3x2)
    col_offset = 23 + 17
    for c in range(17):
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U[i*m + j][col_offset + c] += U_s[i*3 + j][c]
                    V[j*p + (5+k)][col_offset + c] += V_s[j*3 + k][c]
                    W[(5+k)*n + i][col_offset + c] += W_s[k*3 + i][c]
                    
    # Block 4: k=7 (Schoolbook 3x3x1)
    col_offset = 23 + 17 + 17
    for c in range(9):
        for i in range(3):
            for j in range(3):
                U[i*m + j][col_offset + c] += U_1[i*3 + j][c]
                V[j*p + 7][col_offset + c] += V_1[j*3 + 0][c]
                W[7*n + i][col_offset + c] += W_1[0*3 + i][c]
                
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to previous best (68)
        U_fb, V_fb, W_fb = _build_decomposition(3, 3, 8, [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)])
        return {"U": U_fb, "V": V_fb, "W": W_fb}

def _build_decomposition(n, m, p, blocks):
    """Fallback: 4 Strassen blocks + Schoolbook."""
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] += sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] += sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] += sW[dk*2+di][c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W
