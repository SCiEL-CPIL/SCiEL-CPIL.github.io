"""Optimized tensor decomposition for 2x4x5 matrix multiplication.

Uses block-Strassen: split middle dim 4 into two 2-blocks, apply Strassen
(rank 7) to 2x2x2 sub-problems and schoolbook (rank 12) to 2x2x3 parts.
Total rank: 2*(7+12) = 38, improving over schoolbook rank 40.
"""

import time

DIMENSIONS = [2, 4, 5]

def _strassen_2x2x2():
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    # M1 = (X00+X11)*(Y00+Y11)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # M2 = (X10+X11)*Y00
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1
    # M3 = X00*(Y01-Y11)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1
    # M4 = X11*(Y10-Y00)
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[1][3]=1
    # M5 = (X00+X01)*Y11
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[3][4]=1; W[0][4]=-1
    # M6 = (X10-X00)*(Y00+Y01)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    # M7 = (X01-X11)*(Y10+Y11)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[2][6]=1
    return U, V, W

def _schoolbook_2x2x3():
    n,m,p = 2,2,3
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col]=1; V[j*p+k][col]=1; W[k*n+i][col]=1; col+=1
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    total_rank = 2*(7+12)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    sU, sV, sW = _strassen_2x2x2()
    bU, bV, bW = _schoolbook_2x2x3()
    
    def embed_strassen(offset, u_rows, v_rows, w_rows):
        for c in range(7):
            for idx in range(4):
                if sU[idx][c]: U[u_rows[idx]][offset+c] += sU[idx][c]
                if sV[idx][c]: V[v_rows[idx]][offset+c] += sV[idx][c]
                if sW[idx][c]: W[w_rows[idx]][offset+c] += sW[idx][c]
    
    def embed_sb(offset, u_rows, v_rows, w_rows):
        for c in range(12):
            for idx in range(4):
                if bU[idx][c]: U[u_rows[idx]][offset+c] += bU[idx][c]
            for idx in range(6):
                if bV[idx][c]: V[v_rows[idx]][offset+c] += bV[idx][c]
            for idx in range(6):
                if bW[idx][c]: W[w_rows[idx]][offset+c] += bW[idx][c]
    
    u_A1 = [i*4+j for i in range(2) for j in range(2)]
    u_A2 = [i*4+(2+j) for i in range(2) for j in range(2)]
    v_B1a = [j*5+k for j in range(2) for k in range(2)]
    v_B1b = [j*5+k for j in range(2) for k in range(2,5)]
    v_B2a = [(2+j)*5+k for j in range(2) for k in range(2)]
    v_B2b = [(2+j)*5+k for j in range(2) for k in range(2,5)]
    w_01 = [k*2+i for k in range(2) for i in range(2)]
    w_234 = [k*2+i for k in range(2,5) for i in range(2)]
    
    embed_strassen(0, u_A1, v_B1a, w_01)
    embed_sb(7, u_A1, v_B1b, w_234)
    embed_strassen(19, u_A2, v_B2a, w_01)
    embed_sb(26, u_A2, v_B2b, w_234)
    
    if time.time() - start_time > 110:
        pass
    
    return {"U": U, "V": V, "W": W}
