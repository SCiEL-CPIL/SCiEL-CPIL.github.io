"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 19
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: 2x2x2 Strassen (j in 0,1; k in 0,1)
    # M1 = (A00+A11)(B00+B11)
    U[0, col] = 1; U[1*m+1, col] = 1
    V[0, col] = 1; V[1, col] = 1
    W[0, col] = 1; W[1*n+1, col] = 1
    col += 1
    # M2 = (A10+A11)B00
    U[1*m, col] = 1; U[1*m+1, col] = 1
    V[0, col] = 1
    W[1*n, col] = 1
    col += 1
    # M3 = A00(B01-B11)
    U[0, col] = 1
    V[1, col] = 1; V[1*m+1, col] = -1 # Wait, V index is j*p+k. j=0, k=1 -> 0*6+1=1. j=1, k=1 -> 1*6+1=7.
    # Correction: V index is j*p+k.
    # j=0, k=1 -> 1. j=1, k=1 -> 7.
    V[1, col] = 1; V[7, col] = -1
    W[1*n, col] = 1
    col += 1
    # M4 = A11(B10-B00)
    U[1*m+1, col] = 1
    V[6, col] = 1; V[0, col] = -1 # j=1, k=0 -> 6. j=0, k=0 -> 0.
    W[1*n+1, col] = 1
    col += 1
    # M5 = (A00+A01)B11
    U[0, col] = 1; U[1, col] = 1
    V[7, col] = 1 # j=1, k=1 -> 7
    W[1*n+1, col] = 1
    col += 1
    # M6 = (A10-A00)(B00+B01)
    U[1*m, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[1*n, col] = 1
    col += 1
    # M7 = (A01-A11)(B10+B11)
    U[1, col] = 1; U[1*m+1, col] = -1
    V[6, col] = 1; V[7, col] = 1
    W[1*n+1, col] = 1
    col += 1
    
    # Part 2: 2x3x4 (j in 2,3,4; k in 2,3,4,5)
    # Local indices: jl in 0,1,2 -> jl+2; kl in 0,1,2,3 -> kl+2
    # Rank 12 algorithm
    # P1 = A00 B00
    U[2*m+2, col] = 1
    V[2*6+2, col] = 1
    W[2*2+0, col] = 1
    col += 1
    # P2 = A01 B10
    U[3*m+2, col] = 1 # j=3, i=0 -> 3*5+0? No, U index is i*m+j.
    # U index: i*m+j. i=0, j=3 -> 3.
    U[3, col] = 1
    V[3*6+2, col] = 1 # j=3, k=2 -> 20
    W[2*2+0, col] = 1
    col += 1
    # P3 = A02 B20
    U[4, col] = 1 # i=0, j=4
    V[4*6+2, col] = 1 # j=4, k=2 -> 26
    W[2*2+0, col] = 1
    col += 1
    # P4 = A10 B01
    U[5+2, col] = 1 # i=1, j=2 -> 7
    U[7, col] = 1
    V[2*6+3, col] = 1 # j=2, k=3 -> 15
    W[3*2+1, col] = 1
    col += 1
    # P5 = A11 B11
    U[5+3, col] = 1 # i=1, j=3 -> 8
    U[8, col] = 1
    V[3*6+3, col] = 1 # j=3, k=3 -> 21
    W[3*2+1, col] = 1
    col += 1
    # P6 = A12 B21
    U[5+4, col] = 1 # i=1, j=4 -> 9
    U[9, col] = 1
    V[4*6+3, col] = 1 # j=4, k=3 -> 27
    W[3*2+1, col] = 1
    col += 1
    # P7 = (A00+A10) B02
    U[2, col] = 1 # i=0, j=2
    U[7, col] = 1 # i=1, j=2
    V[2*6+4, col] = 1 # j=2, k=4 -> 16
    W[4*2+0, col] = 1
    col += 1
    # P8 = (A01+A11) B12
    U[3, col] = 1 # i=0, j=3
    U[8, col] = 1 # i=1, j=3
    V[3*6+4, col] = 1 # j=3, k=4 -> 22
    W[4*2+0, col] = 1
    col += 1
    # P9 = (A02+A12) B22
    U[4, col] = 1 # i=0, j=4
    U[9, col] = 1 # i=1, j=4
    V[4*6+4, col] = 1 # j=4, k=4 -> 28
    W[4*2+0, col] = 1
    col += 1
    # P10 = (A00+A01) B03
    U[2, col] = 1
    U[3, col] = 1
    V[2*6+5, col] = 1 # j=2, k=5 -> 17
    W[5*2+0, col] = 1
    col += 1
    # P11 = (A10+A11) B13
    U[7, col] = 1
    U[8, col] = 1
    V[3*6+5, col] = 1 # j=3, k=5 -> 23
    W[5*2+0, col] = 1
    col += 1
    # P12 = (A02+A12) B23
    U[4, col] = 1
    U[9, col] = 1
    V[4*6+5, col] = 1 # j=4, k=5 -> 29
    W[5*2+0, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
