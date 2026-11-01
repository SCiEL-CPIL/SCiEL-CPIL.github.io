# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    The decomposition is exact and rank-minimal (rank 7 is the known lower bound).
    
    Interface:
        U: shape (4, 7) - factors for A (index i*2+j)
        V: shape (4, 7) - factors for B (index j*2+k)
        W: shape (4, 7) - factors for C (index k*2+i, official trace convention)
    """
    # Rank 7 Strassen decomposition (optimal for 2x2 matrix multiplication)
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # Index conventions:
    # U[i*2+j][r] corresponds to A[i,j]
    # V[j*2+k][r] corresponds to B[j,k]
    # W[k*2+i][r] corresponds to C[i,k] output (official trace convention)

    # Strassen's 7 products:
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b01 - b11)
    # m4 = a11(b10 - b00)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)

    # m1: U[0][0]=1, U[3][0]=1, V[0][0]=1, V[3][0]=1, W[0][0]=1, W[3][0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # m2: U[2][1]=1, U[3][1]=1, V[0][1]=1, W[1][1]=1, W[3][1]=-1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # m3: U[0][2]=1, V[1][2]=1, V[3][2]=-1, W[2][2]=1, W[3][2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # m4: U[3][3]=1, V[2][3]=1, V[0][3]=-1, W[0][3]=1, W[1][3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # m5: U[0][4]=1, U[1][4]=1, V[3][4]=1, W[0][4]=-1, W[2][4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # m6: U[2][5]=1, U[0][5]=-1, V[0][5]=1, V[1][5]=1, W[3][5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # m7: U[1][6]=1, U[3][6]=-1, V[2][6]=1, V[3][6]=1, W[0][6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
