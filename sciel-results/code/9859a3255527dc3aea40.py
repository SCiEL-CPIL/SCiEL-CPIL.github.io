import time
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 17
    rank = 17
    
    # Initialize matrices
    # U shape: (n*m, rank) = (6, 17)
    # V shape: (m*p, rank) = (9, 17)
    # W shape: (p*n, rank) = (6, 17)
    # Note: The problem statement says W[k*n+i, r].
    # Standard tensor decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]?
    # No, the standard form is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # Tensor T_{i,j,k} has T_{i,j,k} = 1 if j is the linking index.
    # Decomposition: T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}?
    # Let's check the interface description:
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the output is indexed by k (col of B) and i (row of A).
    # So W is a matrix of size (p*n, rank).
    # U is (n*m, rank). Index i*m+j.
    # V is (m*p, rank). Index j*p+k.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x2x2 block (A[0:2, 0:2] * B[0:2, 0:2])
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen's algorithm for 2x2x2
    # Multiplications:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # U indices for A parts:
    # A11 -> i=0, j=0 -> idx 0
    # A12 -> i=0, j=1 -> idx 1
    # A21 -> i=1, j=0 -> idx 2
    # A22 -> i=1, j=1 -> idx 3
    
    # V indices for B parts:
    # B11 -> j=0, k=0 -> idx 0
    # B12 -> j=0, k=1 -> idx 1
    # B21 -> j=1, k=0 -> idx 2
    # B22 -> j=1, k=1 -> idx 3
    
    # W indices for C parts (output k*n + i):
    # C11 -> k=0, i=0 -> idx 0
    # C12 -> k=1, i=0 -> idx 1
    # C21 -> k=0, i=1 -> idx 2
    # C22 -> k=1, i=1 -> idx 3
    
    def add_strassen():
        nonlocal col
        # M1
        U[0][col] = 1; U[3][col] = 1
        V[0][col] = 1; V[3][col] = 1
        W[0][col] = 1; W[1][col] = 1; W[2][col] = 1; W[3][col] = 1
        col += 1
        # M2
        U[2][col] = 1; U[3][col] = 1
        V[0][col] = 1
        W[2][col] = 1; W[3][col] = 1
        col += 1
        # M3
        U[0][col] = 1
        V[1][col] = 1; V[3][col] = -1
        W[1][col] = 1
        col += 1
        # M4
        U[3][col] = 1
        V[2][col] = 1; V[0][col] = -1
        W[2][col] = 1
        col += 1
        # M5
        U[0][col] = 1; U[1][col] = 1
        V[3][col] = 1
        W[1][col] = 1; W[3][col] = 1
        col += 1
        # M6
        U[2][col] = 1; U[0][col] = -1
        V[0][col] = 1; V[1][col] = 1
        W[0][col] = 1
        col += 1
        # M7
        U[1][col] = 1; U[3][col] = -1
        V[2][col] = 1; V[3][col] = 1
        W[0][col] = 1; W[2][col] = 1
        col += 1

    add_strassen()
    
    # Part 2: 2x1x2 block (A[0:2, 2] * B[2, 0:2])
    # A col 2 is index j=2.
    # A13 -> i=0, j=2 -> idx 2
    # A23 -> i=1, j=2 -> idx 5
    # B row 2 is index j=2.
    # B31 -> j=2, k=0 -> idx 6
    # B32 -> j=2, k=1 -> idx 7
    # Output C[0:2, 0:2]
    # C11 (k=0,i=0) idx 0
    # C12 (k=1,i=0) idx 1
    # C21 (k=0,i=1) idx 2
    # C22 (k=1,i=1) idx 3
    
    # M8: A13 * B31 -> C11
    U[2][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    col += 1
    # M9: A13 * B32 -> C12
    U[2][col] = 1
    V[7][col] = 1
    W[1][col] = 1
    col += 1
    # M10: A23 * B31 -> C21
    U[5][col] = 1
    V[6][col] = 1
    W[2][col] = 1
    col += 1
    # M11: A23 * B32 -> C22
    U[5][col] = 1
    V[7][col] = 1
    W[3][col] = 1
    col += 1
    
    # Part 3: 2x2x1 block (A[0:2, 0:2] * B[0:2, 2])
    # A rows 0:2, cols 0:2.
    # B rows 0:2, col 2 (k=2).
    # B13 -> j=0, k=2 -> idx 2
    # B23 -> j=1, k=2 -> idx 5
    # Output C[0:2, 2]
    # C13 (k=2,i=0) idx 4
    # C23 (k=2,i=1) idx 5
    
    # M12: A11 * B13 -> C13
    U[0][col] = 1
    V[2][col] = 1
    W[4][col] = 1
    col += 1
    # M13: A12 * B23 -> C13
    U[1][col] = 1
    V[5][col] = 1
    W[4][col] = 1
    col += 1
    # M14: A21 * B13 -> C23
    U[2][col] = 1
    V[2][col] = 1
    W[5][col] = 1
    col += 1
    # M15: A22 * B23 -> C23
    U[3][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    # Part 4: 2x1x1 block (A[0:2, 2] * B[2, 2])
    # A13 -> idx 2, A23 -> idx 5
    # B33 -> j=2, k=2 -> idx 8
    # Output C[0:2, 2]
    # C13 idx 4, C23 idx 5
    
    # M16: A13 * B33 -> C13
    U[2][col] = 1
    V[8][col] = 1
    W[4][col] = 1
    col += 1
    # M17: A23 * B33 -> C23
    U[5][col] = 1
    V[8][col] = 1
    W[5][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
