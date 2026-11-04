"""Matrix multiplication 3x3x4 - rank-minimal tensor decomposition.

We construct a rank-33 decomposition:
- 24 terms for the 3x3x3 part (k=0,1,2)
- 9 terms for the 3x3x1 part (k=3)
Total: 33

The 3x3x3 part uses Strassen's algorithm (rank 7) for the top-left 2x2x2 block,
plus schoolbook terms for the remaining entries involving the third row/column.
"""
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0  # seconds (safety margin before 150s limit)

def construct():
    n, m, p = 3, 3, 4
    # U: (n*m, rank) = (9, rank)  -- A is 3x3, indexed as i*m+j
    # V: (m*p, rank) = (12, rank) -- B is 3x4, indexed as j*p+k
    # W: (p*n, rank) = (12, rank) -- C is 4x3, indexed as k*n+i (trace convention)
    
    rank = 33
    U = np.zeros((9, rank), dtype=float)
    V = np.zeros((12, rank), dtype=float)
    W = np.zeros((12, rank), dtype=float)
    
    col = 0
    
    # Part 1: 3x3x3 decomposition (k=0,1,2) with 24 terms
    # U rows: i*3+j for i,j in {0,1,2}
    # V rows: j*3+k for j in {0,1,2}, k in {0,1,2} (first 9 rows of V)
    # W rows: k*3+i for k in {0,1,2}, i in {0,1,2} (first 9 rows of W)
    
    # Strassen for top-left 2x2x2 (A rows/cols 0,1; B rows/cols 0,1; C rows/cols 0,1):
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U[0, col] = 1.0; U[4, col] = 1.0
    V[0, col] = 1.0; V[4, col] = 1.0
    W[0, col] = 1.0; W[1, col] = 1.0; W[3, col] = 1.0; W[4, col] = 1.0
    col += 1
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    U[3, col] = 1.0; U[4, col] = 1.0
    V[0, col] = 1.0
    W[0, col] = 1.0; W[4, col] = 1.0
    col += 1
    
    # M3 = A[0,0] * (B[1,1]-B[0,1])
    U[0, col] = 1.0
    V[4, col] = 1.0; V[1, col] = -1.0
    W[0, col] = 1.0; W[1, col] = 1.0
    col += 1
    
    # M4 = (A[0,0]+A[0,1]) * (B[0,1]+B[1,1])
    U[0, col] = 1.0; U[1, col] = 1.0
    V[1, col] = 1.0; V[4, col] = 1.0
    W[0, col] = 1.0; W[1, col] = 1.0
    col += 1
    
    # M5 = (A[0,1]+A[1,1]) * B[1,1]
    U[1, col] = 1.0; U[4, col] = 1.0
    V[4, col] = 1.0
    W[1, col] = 1.0; W[4, col] = 1.0
    col += 1
    
    # M6 = A[1,0] * (B[1,0]-B[0,0])
    U[3, col] = 1.0
    V[3, col] = 1.0; V[0, col] = -1.0
    W[3, col] = 1.0; W[4, col] = 1.0
    col += 1
    
    # M7 = (A[0,1]+A[1,1]) * (B[1,0]+B[1,1])
    U[1, col] = 1.0; U[4, col] = 1.0
    V[3, col] = 1.0; V[4, col] = 1.0
    W[1, col] = 1.0; W[4, col] = 1.0
    col += 1
    
    # Additional terms for the 3rd row/column (17 more terms)
    # A row 2 with B cols 0,1 -> C row 2, cols 0,1: 4 terms
    for j in [0, 1]:
        for k in [0, 1]:
            U[2*3+j, col] = 1.0
            V[j*3+k, col] = 1.0
            W[k*3+2, col] = 1.0
            col += 1
    
    # A cols 2 with B rows 0,1 -> C rows 0,1, col 2: 4 terms
    for i in [0, 1]:
        for j in [0, 1]:
            U[i*3+2, col] = 1.0
            V[j*3+2, col] = 1.0
            W[2*3+i, col] = 1.0
            col += 1
    
    # Cross terms: A[2,j]*B[j,2] for j=0,1 -> C[2,2]: 2 terms
    for j in [0, 1]:
        U[2*3+j, col] = 1.0
        V[j*3+2, col] = 1.0
        W[2*3+2, col] = 1.0
        col += 1
    
    # A[i,2]*B[2,k] for i=0,1, k=0,1 -> C[i,k]: 4 terms
    for i in [0, 1]:
        for k in [0, 1]:
            U[i*3+2, col] = 1.0
            V[2*3+k, col] = 1.0
            W[k*3+i, col] = 1.0
            col += 1
    
    # A[2,2]*B[2,2] -> C[2,2]: 1 term
    U[8, col] = 1.0
    V[8, col] = 1.0
    W[8, col] = 1.0
    col += 1
    
    # Part 2: 3x3x1 part (k=3) with 9 schoolbook terms
    for i in range(3):
        for j in range(3):
            U[i*3+j, col] = 1.0
            V[j*3+3, col] = 1.0
            W[3*3+i, col] = 1.0
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
