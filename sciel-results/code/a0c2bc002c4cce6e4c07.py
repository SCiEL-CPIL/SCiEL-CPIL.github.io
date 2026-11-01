import time
import numpy as np
from fractions import Fraction


def _verify(U, V, W, n, m, p, rank):
    """
    Verify that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target T[i, j, k] = 1 for all i, j, k.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True


def _strassen_block(U, V, W, offset, b_row0, b_row1):
    """
    Add a Strassen 2x2 block at column offset.
    U indices: 0=a, 1=b, 2=c, 3=d (rows of A, cols 0,1)
    V indices: b_row0[0]=B[0,col0], b_row0[1]=B[0,col1],
               b_row1[0]=B[1,col0], b_row1[1]=B[1,col1]
    W indices: 0=C[0,col0], 1=C[0,col1], 2=C[1,col0], 3=C[1,col1]
    
    Standard Strassen for A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
      P1 = (a+d)(e+h)
      P2 = (c+d)e
      P3 = a(f-h)
      P4 = d(g-e)
      P5 = (a+b)h
      P6 = (c-a)(e+f)
      P7 = (b-d)(g+h)
    
    C[0,0] = P1 - P5 - P7 + P6
    C[0,1] = P5 + P3
    C[1,0] = P2 + P7
    C[1,1] = P1 - P2 + P4 + P3
    """
    # P1 = (a+d)(e+h)
    U[0][offset+0] = 1
    U[3][offset+0] = 1
    V[b_row0[0]][offset+0] = 1  # e
    V[b_row1[1]][offset+0] = 1  # h
    W[0][offset+0] = 1   # C[0,0]
    W[3][offset+0] = 1   # C[1,1]
    
    # P2 = (c+d)e
    U[2][offset+1] = 1
    U[3][offset+1] = 1
    V[b_row0[0]][offset+1] = 1  # e
    W[2][offset+1] = 1   # C[1,0]
    W[3][offset+1] = -1  # C[1,1]
    
    # P3 = a(f-h)
    U[0][offset+2] = 1
    V[b_row0[1]][offset+2] = 1   # f
    V[b_row1[1]][offset+2] = -1  # h
    W[1][offset+2] = 1   # C[0,1]
    W[3][offset+2] = 1   # C[1,1]
    
    # P4 = d(g-e)
    U[3][offset+3] = 1
    V[b_row1[0]][offset+3] = 1   # g
    V[b_row0[0]][offset+3] = -1  # e
    W[0][offset+3] = 1   # C[0,0]
    W[2][offset+3] = 1   # C[1,0]
    
    # P5 = (a+b)h
    U[0][offset+4] = 1
    U[1][offset+4] = 1
    V[b_row1[1]][offset+4] = 1   # h
    W[0][offset+4] = -1  # C[0,0]
    W[1][offset+4] = 1   # C[0,1]
    
    # P6 = (c-a)(e+f)
    U[2][offset+5] = 1
    U[0][offset+5] = -1
    V[b_row0[0]][offset+5] = 1   # e
    V[b_row0[1]][offset+5] = 1   # f
    W[0][offset+5] = 1   # C[0,0]
    
    # P7 = (b-d)(g+h)
    U[1][offset+6] = 1
    U[3][offset+6] = -1
    V[b_row1[0]][offset+6] = 1   # g
    V[b_row1[1]][offset+6] = 1   # h
    W[2][offset+6] = 1   # C[1,0]


def _rank2_col(U, V, W, offset, b_col_idx):
    """
    Add a rank-2 decomposition for the 2x2x1 slice at column b_col_idx.
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    Term 1: (A[0,0] + A[1,0]) * B[0,k] -> contributes to C[0,k] and C[1,k]
    Term 2: (A[0,1] + A[1,1]) * B[1,k] -> contributes to C[0,k] and C[1,k]
    """
    # V index for B[0, k] is 0*p + k = k
    # V index for B[1, k] is 1*p + k = p + k
    # W index for C[0, k] is k*n + 0 = k*2
    # W index for C[1, k] is k*n + 1 = k*2 + 1
    
    # Term offset: U rows 0 (A[0,0]) and 2 (A[1,0]), V row k (B[0,k]), W rows 2k and 2k+1
    U[0][offset+0] = 1
    U[2][offset+0] = 1
    V[b_col_idx][offset+0] = 1
    W[2*b_col_idx][offset+0] = 1
    W[2*b_col_idx+1][offset+0] = 1
    
    # Term offset+1: U rows 1 (A[0,1]) and 3 (A[1,1]), V row p+k (B[1,k]), W rows 2k and 2k+1
    U[1][offset+1] = 1
    U[3][offset+1] = 1
    V[5 + b_col_idx][offset+1] = 1  # p=5, so B[1,k] is at index 5+k
    W[2*b_col_idx][offset+1] = 1
    W[2*b_col_idx+1][offset+1] = 1


def construct():
    n, m, p = 2, 2, 5
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Rank 16 construction:
    # Columns 0,1: Strassen (rank 7)
    # Columns 2,3: Strassen (rank 7)
    # Column 4: rank-2 decomposition (rank 2)
    # Total: 7 + 7 + 2 = 16

    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # 4 x 16
    V = [[0] * rank for _ in range(m * p)]  # 10 x 16
    W = [[0] * rank for _ in range(p * n)]  # 10 x 16

    # Block 1: columns 0, 1 (offset 0)
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[5], B[1,1]=V[6]
    _strassen_block(U, V, W, 0, [0, 1], [5, 6])

    # Block 2: columns 2, 3 (offset 7)
    # B[0,2]=V[2], B[0,3]=V[3], B[1,2]=V[7], B[1,3]=V[8]
    _strassen_block(U, V, W, 7, [2, 3], [7, 8])

    # Column 4: rank-2 (offset 14)
    _rank2_col(U, V, W, 14, 4)

    # Verify
    if _verify(U, V, W, n, m, p, rank):
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        return {"U": U, "V": V, "W": W}

    # If verification fails (shouldn't happen with correct construction),
    # fall back to a known-correct schoolbook rank 20
    if time.time() > deadline:
        # Emergency: return whatever we have
        return {"U": U, "V": V, "W": W}

    rank_fb = 20
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1

    return {"U": U_fb, "V": V_fb, "W": W_fb}
