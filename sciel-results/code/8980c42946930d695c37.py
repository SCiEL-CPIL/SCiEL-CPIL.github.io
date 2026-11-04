"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 145.0

    n, m, p = 2, 4, 6
    nm, mp, pn = n*m, m*p, p*n  # 8, 24, 12

    # Rank 42 decomposition based on Strassen's algorithm
    # M_{2,4,6} = 3 * M_{2,4,2} = 3 * (2 * M_{2,2,2}) = 6 * 7 = 42
    rank = 42

    U = [[Fraction(0)] * rank for _ in range(nm)]
    V = [[Fraction(0)] * rank for _ in range(mp)]
    W = [[Fraction(0)] * rank for _ in range(pn)]

    # Strassen 2x2x2 decomposition helper
    # a_idx: [a00, a01, a10, a11] -> U indices
    # b_idx: [b00, b01, b10, b11] -> V indices
    # w_idx: [w00, w01, w10, w11] -> W indices
    # c: starting column in rank
    def strassen(a_idx, b_idx, w_idx, c):
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5 = (a00+a01)b11
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    # A is 2x4, A[i,j] -> U[i*m+j]
    # Left cols (j=0,1): a00=0, a01=1, a10=4, a11=5
    # Right cols (j=2,3): a00=2, a01=3, a10=6, a11=7
    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]

    # B is 4x6, B[j,k] -> V[j*p+k]
    # W is k*n+i, W[k*n+i]

    # Block (k=0,1):
    #   B top (j=0,1): B[0,0]=0,B[0,1]=1,B[1,0]=6,B[1,1]=7 -> [0,1,6,7]
    #   B bot (j=2,3): B[2,0]=12,B[2,1]=13,B[3,0]=18,B[3,1]=19 -> [12,13,18,19]
    #   W for C[0,0],C[0,1],C[1,0],C[1,1]: W[0],W[2],W[1],W[3]
    strassen(a_L, [0, 1, 6, 7], [0, 2, 1, 3], 0)
    strassen(a_R, [12, 13, 18, 19], [0, 2, 1, 3], 7)

    # Block (k=2,3):
    #   B top: B[0,2]=2,B[0,3]=3,B[1,2]=8,B[1,3]=9 -> [2,3,8,9]
    #   B bot: B[2,2]=14,B[2,3]=15,B[3,2]=20,B[3,3]=21 -> [14,15,20,21]
    #   W for C[0,2],C[0,3],C[1,2],C[1,3]: W[4],W[6],W[5],W[7]
    strassen(a_L, [2, 3, 8, 9], [4, 6, 5, 7], 14)
    strassen(a_R, [14, 15, 20, 21], [4, 6, 5, 7], 21)

    # Block (k=4,5):
    #   B top: B[0,4]=4,B[0,5]=5,B[1,4]=10,B[1,5]=11 -> [4,5,10,11]
    #   B bot: B[2,4]=16,B[2,5]=17,B[3,4]=22,B[3,5]=23 -> [16,17,22,23]
    #   W for C[0,4],C[0,5],C[1,4],C[1,5]: W[8],W[10],W[9],W[11]
    strassen(a_L, [4, 5, 10, 11], [8, 10, 9, 11], 28)
    strassen(a_R, [16, 17, 22, 23], [8, 10, 9, 11], 35)

    # Verify
    def check():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    if not check():
        # Fallback: schoolbook rank 48
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(nm)]
        V = [[Fraction(0)] * rank for _ in range(mp)]
        W = [[Fraction(0)] * rank for _ in range(pn)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
