"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition using Strassen for 2x2x2 subblock and schoolbook for the rest
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for C[0:2, 0:2] using A[0:2, 0:2] and B[0:2, 0:2]
    # Indices: i in 0,1; j in 0,1; k in 0,1
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Column 0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1; W[6][0] = 1; W[9][0] = 1
    # W indices: k*n+i
    # C00 (k=0,i=0) -> W[0*3+0] = W[0]
    # C01 (k=1,i=0) -> W[1*3+0] = W[3]
    # C10 (k=0,i=1) -> W[0*3+1] = W[1]
    # C11 (k=1,i=1) -> W[1*3+1] = W[4]
    # Wait, W[k*n+i]
    # C00: k=0, i=0 -> W[0]
    # C01: k=1, i=0 -> W[3]
    # C10: k=0, i=1 -> W[1]
    # C11: k=1, i=1 -> W[4]
    
    # M1 contributes to C00, C01, C10, C11
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-verify the W indices.
    # W[k*n + i]
    # k=0, i=0 -> 0
    # k=1, i=0 -> 3
    # k=2, i=0 -> 6
    # k=3, i=0 -> 9
    # k=4, i=0 -> 12
    # k=0, i=1 -> 1
    # k=1, i=1 -> 4
    # k=2, i=1 -> 7
    # k=3, i=1 -> 10
    # k=4, i=1 -> 13
    # k=0, i=2 -> 2
    # k=1, i=2 -> 5
    # k=2, i=2 -> 8
    # k=3, i=2 -> 11
    # k=4, i=2 -> 14
    
    # M1: (A00+A11)(B00+B11)
    # U: A00 (0), A11 (3)
    # V: B00 (0), B11 (4)
    # W: C00 (0), C10 (1), C01 (3), C11 (4) with coefficients 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = 1; W[4][0] = 1
    
    # M2: (A10+A11)B00
    # U: A10 (3), A11 (4)
    # V: B00 (0)
    # W: C10 (1), C11 (4) with coefficients 1
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[4][1] = 1
    
    # M3: A00(B01-B11)
    # U: A00 (0)
    # V: B01 (1), B11 (4) with coeff -1
    # W: C01 (3), C11 (4) with coeff 1
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[3][2] = 1; W[4][2] = 1
    
    # M4: A11(B10-B00)
    # U: A11 (4)
    # V: B10 (3), B00 (0) with coeff -1
    # W: C00 (0), C10 (1) with coeff 1
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A00+A01)B11
    # U: A00 (0), A01 (1)
    # V: B11 (4)
    # W: C00 (0) with coeff -1, C01 (3) with coeff 1
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # U: A10 (3), A00 (0) with coeff -1
    # V: B00 (0), B01 (1)
    # W: C11 (4) with coeff 1
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # U: A01 (1), A11 (4) with coeff -1
    # V: B10 (3), B11 (4)
    # W: C00 (0) with coeff 1
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # Schoolbook part for the remaining 37 terms
    # Terms: A[i,j] * B[j,k] for all (i,j,k) NOT in the Strassen set
    # Strassen set: i in 0,1; j in 0,1; k in 0,1
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in Strassen set
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
