import time
import sys

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = 3, 3, 8
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2 A block and 2x2 B block
    # A block indices: (i1, i2) rows, (j1, j2) cols
    # B block indices: (j1, j2) rows, (k1, k2) cols
    # Output C indices: (i1, i2) rows, (k1, k2) cols
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A11 + A22)(B11 + B22)
        # A11 at (i1, j1), A22 at (i2, j2)
        # B11 at (j1, k1), B22 at (j2, k2)
        # C11 at (i1, k1), C22 at (i2, k2)
        U[i1 * m + j1][col] += 1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k1][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M2 = (A21 + A22)B11
        # A21 at (i2, j1), A22 at (i2, j2)
        # B11 at (j1, k1)
        # C21 at (i2, k1)
        U[i2 * m + j1][col] += 1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k1][col] += 1
        W[k1 * n + i2][col] += 1
        col += 1
        
        # M3 = A11(B12 - B22)
        # A11 at (i1, j1)
        # B12 at (j1, k2), B22 at (j2, k2)
        # C12 at (i1, k2)
        U[i1 * m + j1][col] += 1
        V[j1 * p + k2][col] += 1
        V[j2 * p + k2][col] -= 1
        W[k2 * n + i1][col] += 1
        col += 1
        
        # M4 = A22(B21 - B11)
        # A22 at (i2, j2)
        # B21 at (j2, k1), B11 at (j1, k1)
        # C21 at (i2, k1)
        U[i2 * m + j2][col] += 1
        V[j2 * p + k1][col] += 1
        V[j1 * p + k1][col] -= 1
        W[k1 * n + i2][col] += 1
        col += 1
        
        # M5 = (A11 + A12)B22
        # A11 at (i1, j1), A12 at (i1, j2)
        # B22 at (j2, k2)
        # C12 at (i1, k2)
        U[i1 * m + j1][col] += 1
        U[i1 * m + j2][col] += 1
        V[j2 * p + k2][col] += 1
        W[k2 * n + i1][col] += 1
        col += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        # A21 at (i2, j1), A11 at (i1, j1)
        # B11 at (j1, k1), B12 at (j1, k2)
        # C22 at (i2, k2)
        U[i2 * m + j1][col] += 1
        U[i1 * m + j1][col] -= 1
        V[j1 * p + k1][col] += 1
        V[j1 * p + k2][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        # A12 at (i1, j2), A22 at (i2, j2)
        # B21 at (j2, k1), B22 at (j2, k2)
        # C22 at (i2, k2)
        U[i1 * m + j2][col] += 1
        U[i2 * m + j2][col] -= 1
        V[j2 * p + k1][col] += 1
        V[j2 * p + k2][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1

    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:8]
    # B is split into 4 blocks of 2x2
    for t in range(4):
        k1 = 2 * t
        k2 = 2 * t + 1
        add_strassen(0, 1, 0, 1, k1, k2)
        
    # 2. A[0:2, 2] * B[2, 0:8]
    # A is 2x1, B is 1x8
    for k in range(8):
        # C[0, k] += A[0, 2] * B[2, k]
        U[0 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # C[1, k] += A[1, 2] * B[2, k]
        U[1 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1

    # 3. A[2, 0:2] * B[0:2, 0:8]
    # A is 1x2, B is 2x8
    for k in range(8):
        # C[2, k] += A[2, 0] * B[0, k]
        U[2 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
        
        # C[2, k] += A[2, 1] * B[1, k]
        U[2 * m + 1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1

    # 4. A[2, 2] * B[2, 0:8]
    # A is 1x1, B is 1x8
    for k in range(8):
        # C[2, k] += A[2, 2] * B[2, k]
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
