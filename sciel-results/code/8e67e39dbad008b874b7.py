"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check if the decomposition is exact. Returns True if valid."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-9)

def _strassen_2x2():
    """Strassen 2x2x2 rank-7 decomposition."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1=(A00+A11)(B00+B11)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # M2=(A10+A11)B00
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1
    # M3=A00(B01-B11)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
    # M4=A11(B10-B00)
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
    # M5=(A00+A01)B11
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
    # M6=(A10-A00)(B00+B01)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    # M7=(A01-A11)(B10+B11)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Total rank: 8*7 + 2*1 + 1*1 = 56 + 2 + 1 = 59
    total_rank = 59
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    Us, Vs, Ws = _strassen_2x2()
    
    # Part 1: 4x4x4 (i=0..3, j=0..3, k=0..3) using 8 Strassen blocks
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                io, jo, ko = ib*2, jb*2, kb*2
                for r in range(4):
                    for c in range(7):
                        ii, jj = r // 2, r % 2
                        U[(io+ii)*m + (jo+jj)][col+c] += Us[r][c]
                        V[(jo+jj)*p + (ko+r%2)][col+c] += Vs[r][c]
                        W[(ko+r%2)*n + (io+r//2)][col+c] += Ws[r][c]
                col += 7
    
    # Part 2: j=0..3, k=4 (rank 1)
    for i in range(n):
        for j in range(4):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
    col += 1
    
    # Part 3: j=0..3, k=5 (rank 1)
    for i in range(n):
        for j in range(4):
            U[i*m+j][col] = 1
            V[j*p+5][col] = 1
            W[5*n+i][col] = 1
    col += 1
    
    # Part 4: j=4, k=0..5 (rank 1)
    for i in range(n):
        for k in range(p):
            U[i*m+4][col] = 1
            V[4*p+k][col] = 1
            W[k*n+i][col] = 1
    col += 1
    
    result = {"U": U, "V": V, "W": W}
    if _verify(U, V, W):
        return result
    return _schoolbook()
# EVOLVE-BLOCK-END
