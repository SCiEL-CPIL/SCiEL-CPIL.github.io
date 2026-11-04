"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U, V, W each of shape (4, 7).
    A flat: (i,j) -> i*2+j: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    B flat: (j,k) -> j*2+k: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    C flat (trace): (k,i) -> k*2+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00: k=0,i=0
    W[3][0] = 1  # C11: k=1,i=1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # C10: k=0,i=1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1  # C01: k=1,i=0
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00: k=0,i=0
    W[1][3] = 1  # C10: k=0,i=1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C00: k=0,i=0
    W[2][4] = 1   # C01: k=1,i=0
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1   # C11: k=1,i=1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1   # C00: k=0,i=0
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Use Strassen for the 2x2x2 sub-block (i<2, j<2, k<2)
    # Schoolbook for all other (i,j,k) triples
    # Strassen rank: 7, Schoolbook for 2x2x2 would be 8, saving 1
    # Total rank = 75 - 8 + 7 = 74
    
    rank_total = 74
    U = [[0] * rank_total for _ in range(n * m)]  # 15 x 74
    V = [[0] * rank_total for _ in range(m * p)]  # 25 x 74
    W = [[0] * rank_total for _ in range(p * n)]  # 15 x 74
    
    col = 0
    
    # Embed Strassen 2x2x2
    sU, sV, sW = _strassen_2x2x2()
    for r in range(7):
        for idx in range(4):
            if sU[idx][r] != 0:
                U[idx][col] = sU[idx][r]
            if sV[idx][r] != 0:
                V[idx][col] = sV[idx][r]
            if sW[idx][r] != 0:
                W[idx][col] = sW[idx][r]
        col += 1
    
    # Schoolbook for remaining triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    assert col == rank_total, f"Expected {rank_total}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
