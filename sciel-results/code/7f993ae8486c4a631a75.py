import time
import sys

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 84
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # 7 multiplications
        # M1: (A11+A22)(B11+B22) -> C11(+), C22(+)
        # A indices: (i0, j0), (i0+1, j0+1)
        # B indices: (j0, k0), (j0+1, k0+1)
        # C indices: (i0, k0), (i0+1, k0+1)
        
        # M1
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # M2: (A21+A22)B11 -> C21(+)
        # A: (i0+1, j0), (i0+1, j0+1)
        # B: (j0, k0)
        # C: (i0+1, k0)
        U[(i0+1)*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # M3: A11(B12-B22) -> C12(+)
        # A: (i0, j0)
        # B: (j0, k0+1), (j0+1, k0+1)
        # C: (i0, k0+1)
        U[i0*m + j0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = -1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # M4: A22(B21-B11) -> C11(-), C21(+)
        # A: (i0+1, j0+1)
        # B: (j0+1, k0), (j0, k0)
        # C: (i0, k0), (i0+1, k0)
        U[(i0+1)*m + (j0+1)][col] = 1
        V[(j0+1)*p + k0][col] = 1
        V[j0*p + k0][col] = -1
        W[k0*n + i0][col] = -1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # M5: (A11+A12)B22 -> C11(-), C12(+)
        # A: (i0, j0), (i0, j0+1)
        # B: (j0+1, k0+1)
        # C: (i0, k0), (i0, k0+1)
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = -1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # M6: (A21-A11)(B11+B12) -> C22(+)
        # A: (i0+1, j0), (i0, j0)
        # B: (j0, k0), (j0, k0+1)
        # C: (i0+1, k0+1)
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # M7: (A12-A22)(B21+B22) -> C11(+), C22(+)
        # A: (i0, j0+1), (i0+1, j0+1)
        # B: (j0+1, k0), (j0+1, k0+1)
        # C: (i0, k0), (i0+1, k0+1)
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1

    def add_simple_2x1x6(i0, j0, k0):
        # i0 is start i (0), j0 is fixed j (4), k0 is start k (0)
        # A is 2x1 at (i0..i0+1, j0)
        # B is 1x6 at (j0, k0..k0+5)
        # C is 2x6 at (i0..i0+1, k0..k0+5)
        nonlocal col
        for k in range(6):
            for i in range(2):
                U[(i0+i)*m + j0][col] = 1
                V[j0*p + (k0+k)][col] = 1
                W[(k0+k)*n + (i0+i)][col] = 1
                col += 1

    def add_simple_1x5x6(i0, j0, k0):
        # i0 is fixed i (2), j0 is start j (0), k0 is start k (0)
        # A is 1x5 at (i0, j0..j0+4)
        # B is 5x6 at (j0..j0+4, k0..k0+5)
        # C is 1x6 at (i0, k0..k0+5)
        nonlocal col
        for j in range(5):
            for k in range(6):
                U[i0*m + (j0+j)][col] = 1
                V[(j0+j)*p + (k0+k)][col] = 1
                W[(k0+k)*n + i0][col] = 1
                col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:6]
    # Split B into 3x2
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 0, 4)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:6]
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 2, 4)
    
    # Block 3: A[0:2, 4] * B[4, 0:6]
    add_simple_2x1x6(0, 4, 0)
    
    # Block 4: A[2, 0:5] * B[0:5, 0:6]
    add_simple_1x5x6(2, 0, 0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
