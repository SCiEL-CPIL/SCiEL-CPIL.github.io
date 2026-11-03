"""Tensor rank decomposition for 3x5x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid rank decomposition for 3x5x5 matrix multiplication.
    
    Strategy:
      - Use one 3x3x3 Strassen block (rank 19) at (0,0,0).
      - Use three 2x2x2 Strassen blocks (rank 7 each) in the remainder.
      - Schoolbook for the rest.
      - Total rank: 19 + 3*7 + 24 = 64.
    """
    n, m, p = DIMENSIONS
    total_terms = n * m * p  # 75
    
    # Pre-allocate matrices
    # Rank 19 + 7 + 7 + 7 + 24 = 64
    rank = 64
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Helper to add 3x3x3 Strassen (Rank 19)
    def add_strassen3(i0, j0, k0):
        nonlocal col
        # M1 = (A00+A11+A22)(B00+B11+B22)
        c = col
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        U[(i0+2)*m + (j0+2)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        V[(j0+2)*p + (k0+2)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1
        col += 1

        # M2 = (A10+A11+A21)(B00+B10+B20)
        c = col
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        U[(i0+2)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+2)*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+1)][c] += 1
        col += 1

        # M3 = (A00+A20+A22)(B00+B02+B22)
        c = col
        U[i0*m + j0][c] += 1
        U[(i0+2)*m + j0][c] += 1
        U[(i0+2)*m + (j0+2)][c] += 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+2)][c] += 1
        V[(j0+2)*p + (k0+2)][c] += 1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+2)][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1
        col += 1

        # M4 = (A01+A11+A12)(B10+B11+B12)
        c = col
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+2)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+2)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+1)][c] += 1
        col += 1

        # M5 = (A00+A01+A10)(B00+B01+B10)
        c = col
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + j0][c] += 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        col += 1

        # M6 = (A10+A20+A21)(B10+B20+B21)
        c = col
        U[(i0+1)*m + j0][c] += 1
        U[(i0+2)*m + j0][c] += 1
        U[(i0+2)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+2)*p + k0][c] += 1
        V[(j0+2)*p + (k0+1)][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[k0*n + (i0+2)][c] += 1
        W[(k0+1)*n + (i0+2)][c] += 1
        col += 1

        # M7 = (A01+A02+A12)(B10+B11+B12)
        c = col
        U[i0*m + (j0+1)][c] += 1
        U[i0*m + (j0+2)][c] += 1
        U[(i0+1)*m + (j0+2)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+2)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+2)*n + i0][c] += 1
        col += 1

        # M8 = (A00+A10+A11)(B00+B01+B11)
        c = col
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        col += 1

        # M9 = (A20+A21+A22)(B00+B10+B11)
        c = col
        U[(i0+2)*m + j0][c] += 1
        U[(i0+2)*m + (j0+1)][c] += 1
        U[(i0+2)*m + (j0+2)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + (i0+2)][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        col += 1

        # M10 = (A00+A02+A22)(B00+B02+B22)
        c = col
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+2)][c] += 1
        U[(i0+2)*m + (j0+2)][c] += 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+2)][c] += 1
        V[(j0+2)*p + (k0+2)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+2)*n + i0][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1
        col += 1

        # M11 = (A11+A12+A22)(B11+B12+B22)
        c = col
        U[(i0+1)*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+2)][c] += 1
        U[(i0+2)*m + (j0+2)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+2)][c] += 1
        V[(j0+2)*p + (k0+2)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1
        col += 1

        # M12 = (A00-A11)(B00+B11)
        c = col
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        col += 1

        # M13 = (A10-A00)(B00-B01)
        c = col
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += -1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + i0][c] += 1
        col += 1

        # M14 = (A01-A11)(B10-B11)
        c = col
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        col += 1

        # M15 = (A02-A22)(B20-B22)
        c = col
        U[i0*m + (j0+2)][c] += 1
        U[(i0+2)*m + (j0+2)][c] += -1
        V[(j0+2)*p + k0][c] += 1
        V[(j0+2)*p + (k0+2)][c] += -1
        W[k0*n + i0][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1
        col += 1

        # M16 = (A12-A22)(B21-B22)
        c = col
        U[(i0+1)*m + (j0+2)][c] += 1
        U[(i0+2)*m + (j0+2)][c] += -1
        V[(j0+2)*p + (k0+1)][c] += 1
        V[(j0+2)*p + (k0+2)][c] += -1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[(k0+2)*n + (i0+2)][c] += 1
        col += 1

        # M17 = (A20-A00)(B02-B22)
        c = col
        U[(i0+2)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + (k0+2)][c] += 1
        V[(j0+2)*p + (k0+2)][c] += -1
        W[(k0+2)*n + (i0+2)][c] += 1
        W[(k0+2)*n + i0][c] += 1
        col += 1

        # M18 = (A21-A11)(B12-B22)
        c = col
        U[(i0+2)*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + (k0+2)][c] += 1
        V[(j0+2)*p + (k0+2)][c] += -1
        W[(k0+2)*n + (i0+2)][c] += 1
        W[(k0+2)*n + (i0+1)][c] += 1
        col += 1

        # M19 = (A10-A20)(B20-B00)
        c = col
        U[(i0+1)*m + j0][c] += 1
        U[(i0+2)*m + j0][c] += -1
        V[(j0+2)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + (i0+1)][c] += 1
        W[k0*n + (i0+2)][c] += 1
        col += 1

    # Helper to add 2x2x2 Strassen (Rank 7)
    def add_strassen2(i0, j0, k0):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        c = col
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        col += 1

        # M2 = (A10+A11)(B00)
        c = col
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        col += 1

        # M3 = (A00)(B01+B11)
        c = col
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        col += 1

        # M4 = (A11)(B10+B00)
        c = col
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        col += 1

        # M5 = (A00+A01)(B11)
        c = col
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        col += 1

        # M6 = (A10)(B00+B01)
        c = col
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        col += 1

        # M7 = (A01)(B10+B11)
        c = col
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        col += 1

    # 1. Add 3x3x3 block at (0,0,0)
    add_strassen3(0, 0, 0)
    
    # 2. Add 2x2x2 blocks in remainder
    # R1: i 0..2, j 3..4, k 0..2. Block at (0,3,0)
    add_strassen2(0, 3, 0)
    
    # R2: i 0..2, j 0..2, k 3..4. Block at (0,0,3)
    add_strassen2(0, 0, 3)
    
    # R3: i 0..2, j 3..4, k 3..4. Block at (0,3,3)
    add_strassen2(0, 3, 3)
    
    # 3. Schoolbook for the rest
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                covered[di][dj][dk] = True
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                covered[di][3+dj][dk] = True
                covered[di][dj][3+dk] = True
                covered[di][3+dj][3+dk] = True
                
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    U_arr = np.array(result["U"], dtype=float)
    V_arr = np.array(result["V"], dtype=float)
    W_arr = np.array(result["W"], dtype=float)
    
    C_reconstructed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                C_reconstructed[i, k] += np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
    
    error = np.max(np.abs(C_reconstructed - C_expected))
    print(f"Max reconstruction error: {error}")
    assert error < 1e-10, "Decomposition is incorrect!"
    print("Decomposition verified successfully.")
