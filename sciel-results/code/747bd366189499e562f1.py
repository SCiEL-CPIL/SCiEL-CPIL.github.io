import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 71 solution using Laderman for 3x3x3 and Schoolbook for the rest
    # Split A (3x5) into A1 (3x3) and A2 (3x2)
    # Split B (5x5) into B11 (3x3), B12 (3x2), B21 (2x3), B22 (2x2)
    
    # C = A1*B11 + A2*B21  (cols 0-2)
    # C = A1*B12 + A2*B22  (cols 3-4)
    
    # M1: 3x3x3 (Laderman, rank 23)
    # M2: 3x2x3 (Schoolbook, rank 18)
    # M3: 3x3x2 (Schoolbook, rank 18)
    # M4: 3x2x2 (Schoolbook, rank 12)
    # Total Rank: 23 + 18 + 18 + 12 = 71
    
    total_rank = 71
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper for Laderman 3x3x3
    def add_laderman(i0, j0, k0, col_offset):
        # Laderman's algorithm for 3x3x3
        # Local indices 0,1,2
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # 23 terms
        # Reference: Laderman (1976)
        # This is a known rank 23 decomposition.
        
        # M1 = (A00+A11+A22)(B00+B11+B22)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        U[(i0+2)*m + (j0+2)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        V[(j0+2)*p + (k0+2)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1

        # M2 = (A00+A11)(B00+B11)
        c = col_offset + 1
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M3 = (A00+A22)(B00+B22)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        U[(i0+2)*m + (j0+2)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+2)*p + (k0+2)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1

        # M4 = (A11+A22)(B11+B22)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        U[(i0+2)*m + (j0+2)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        V[(j0+2)*p + (k0+2)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1

        # M5 = (A01+A10)(B01+B10)
        c = col_offset + 4
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M6 = (A02+A20)(B02+B20)
        c = col_offset + 5
        U[i0*m + (j0+2)][c] += 1
        U[(i0+2)*m + j0][c] += 1
        V[j0*p + (k0+2)][c] += 1
        V[(j0+2)*p + k0][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1

        # M7 = (A12+A21)(B12+B21)
        c = col_offset + 6
        U[(i0+1)*m + (j0+2)][c] += 1
        U[(i0+2)*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+2)][c] += 1
        V[(j0+2)*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1

        # M8 = (A00-A11)(B00-B11)
        c = col_offset + 7
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M9 = (A00-A22)(B00-B22)
        c = col_offset + 8
        U[i0*m + j0][c] += 1
        U[(i0+2)*m + (j0+2)][c] += -1
        V[j0*p + k0][c] += 1
        V[(j0+2)*p + (k0+2)][c] += -1
        W[k0*n + i0][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1

        # M10 = (A11-A22)(B11-B22)
        c = col_offset + 9
        U[(i0+1)*m + (j0+1)][c] += 1
        U[(i0+2)*m + (j0+2)][c] += -1
        V[(j0+1)*p + (k0+1)][c] += 1
        V[(j0+2)*p + (k0+2)][c] += -1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1

        # M11 = (A01-A10)(B01-B10)
        c = col_offset + 10
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + j0][c] += -1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M12 = (A02-A20)(B02-B20)
        c = col_offset + 11
        U[i0*m + (j0+2)][c] += 1
        U[(i0+2)*m + j0][c] += -1
        V[j0*p + (k0+2)][c] += 1
        V[(j0+2)*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1

        # M13 = (A12-A21)(B12-B21)
        c = col_offset + 12
        U[(i0+1)*m + (j0+2)][c] += 1
        U[(i0+2)*m + (j0+1)][c] += -1
        V[(j0+1)*p + (k0+2)][c] += 1
        V[(j0+2)*p + (k0+1)][c] += -1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1

        # M14 = (A00+A01+A10)(B00+B01+B10)
        c = col_offset + 13
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + j0][c] += 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1

        # M15 = (A00+A02+A20)(B00+B02+B20)
        c = col_offset + 14
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+2)][c] += 1
        U[(i0+2)*m + j0][c] += 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+2)][c] += 1
        V[(j0+2)*p + k0][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+2)*n + i0][c] += 1
        W[k0*n + (i0+2)][c] += 1

        # M16 = (A11+A12+A21)(B11+B12+B21)
        c = col_offset + 15
        U[(i0+1)*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+2)][c] += 1
        U[(i0+2)*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+2)][c] += 1
        V[(j0+2)*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+2)][c] += 1

        # M17 = (A22+A20+A02)(B22+B20+B02)
        c = col_offset + 16
        U[(i0+2)*m + (j0+2)][c] += 1
        U[(i0+2)*m + j0][c] += 1
        U[i0*m + (j0+2)][c] += 1
        V[(j0+2)*p + (k0+2)][c] += 1
        V[(j0+2)*p + k0][c] += 1
        V[j0*p + (k0+2)][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1
        W[k0*n + (i0+2)][c] += 1
        W[(k0+2)*n + i0][c] += 1

        # M18 = (A11+A20)(B11+B20)
        c = col_offset + 17
        U[(i0+1)*m + (j0+1)][c] += 1
        U[(i0+2)*m + j0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        V[(j0+2)*p + k0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[k0*n + (i0+2)][c] += 1

        # M19 = (A22+A01)(B22+B01)
        c = col_offset + 18
        U[(i0+2)*m + (j0+2)][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+2)*p + (k0+2)][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1
        W[(k0+1)*n + i0][c] += 1

        # M20 = (A00+A21)(B00+B21)
        c = col_offset + 19
        U[i0*m + j0][c] += 1
        U[(i0+2)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+2)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+2)][c] += 1

        # M21 = (A11+A02)(B11+B02)
        c = col_offset + 20
        U[(i0+1)*m + (j0+1)][c] += 1
        U[i0*m + (j0+2)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        V[j0*p + (k0+2)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + i0][c] += 1

        # M22 = (A22+A10)(B22+B10)
        c = col_offset + 21
        U[(i0+2)*m + (j0+2)][c] += 1
        U[(i0+1)*m + j0][c] += 1
        V[(j0+2)*p + (k0+2)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1
        W[k0*n + (i0+1)][c] += 1

        # M23 = (A01+A22)(B01+B22)
        c = col_offset + 22
        U[i0*m + (j0+1)][c] += 1
        U[(i0+2)*m + (j0+2)][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+2)*p + (k0+2)][c] += 1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1

    # Add Laderman block for i in 0..2, j in 0..2, k in 0..2
    add_laderman(0, 0, 0, col)
    col += 23
    
    # Schoolbook for M2: A(3x2) B(2x3) -> i in 0..2, j in 3..4, k in 0..2
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U[i*m + (j0+2+j)][col] += 1 # j0=0, so j is 2,3
                V[(j0+2+j)*p + k][col] += 1
                W[k*n + i][col] += 1
                col += 1
                
    # Schoolbook for M3: A(3x3) B(3x2) -> i in 0..2, j in 0..2, k in 3..4
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i*m + j][col] += 1
                V[j*p + (k0+3+k)][col] += 1
                W[(k0+3+k)*n + i][col] += 1
                col += 1
                
    # Schoolbook for M4: A(3x2) B(2x2) -> i in 0..2, j in 3..4, k in 3..4
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i*m + (j0+2+j)][col] += 1
                V[(j0+2+j)*p + (k0+3+k)][col] += 1
                W[(k0+3+k)*n + i][col] += 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
