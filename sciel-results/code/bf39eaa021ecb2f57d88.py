"""Matrix multiplication 2x4x7 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Correct rank-50 decomposition:
    # 2x4x7 = 2x4x4 + 2x4x2 + 2x4x1
    # 2x4x4: 4 Strassen blocks (2x2x2 each) = 28
    # 2x4x2: 2 Strassen blocks (2x2x2 each) = 14  
    # 2x4x1: 8 rank-1 terms = 8
    # Total = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for 2x2x2
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2: (A21+A22)B11 -> C21 += M2, C22 -= M2
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        # M3: A11(B12-B22) -> C12 += M3, C22 += M3
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4: A22(B21-B11) -> C11 += M4, C21 += M4
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5: (A11+A12)B22 -> C11 -= M5, C12 += M5
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12) -> C22 += M6
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22) -> C11 += M7
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Part 1: 2x4x4 (k=0..3) - 4 Strassen blocks
    add_strassen(0, 0, 0)   # i=0,1; j=0,1; k=0,1
    add_strassen(0, 0, 2)   # i=0,1; j=0,1; k=2,3
    add_strassen(0, 2, 0)   # i=0,1; j=2,3; k=0,1
    add_strassen(0, 2, 2)   # i=0,1; j=2,3; k=2,3
    
    # Part 2: 2x4x2 (k=4..5) - 2 Strassen blocks
    add_strassen(0, 0, 4)   # i=0,1; j=0,1; k=4,5
    add_strassen(0, 2, 4)   # i=0,1; j=2,3; k=4,5
    
    # Part 3: 2x4x1 (k=6) - 8 rank-1 terms
    for j in range(4):
        for i in range(2):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    # Time guard
    if time.time() - start_time > 100:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
