"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    # U shape: (n*m, rank) = (15, rank)
    # V shape: (m*p, rank) = (25, rank)
    # W shape: (p*n, rank) = (15, rank)
    # Convention: C[k*n+i, r] for W, so W index is k*n + i
    
    rank = 71
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen block helper:
    # For a 2x2x2 block with i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    # Strassen uses 7 multiplications:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # A indices: i0*m+j0, i0*m+j0+1, (i0+1)*m+j0, (i0+1)*m+j0+1
        # B indices: j0*p+k0, j0*p+k0+1, (j0+1)*p+k0, (j0+1)*p+k0+1
        # C indices (W): k*n+i, so (k0)*n+i0, (k0)*n+i0+1, (k0+1)*n+i0, (k0+1)*n+i0+1
        
        A00 = i0 * m + j0
        A01 = i0 * m + j0 + 1
        A10 = (i0 + 1) * m + j0
        A11 = (i0 + 1) * m + j0 + 1
        
        B00 = j0 * p + k0
        B01 = j0 * p + k0 + 1
        B10 = (j0 + 1) * p + k0
        B11 = (j0 + 1) * p + k0 + 1
        
        C00 = k0 * n + i0
        C01 = k0 * n + i0 + 1
        C10 = (k0 + 1) * n + i0
        C11 = (k0 + 1) * n + i0 + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[A00][col] = 1
        U[A11][col] = 1
        V[B00][col] = 1
        V[B11][col] = 1
        W[C00][col] = 1
        W[C11][col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[A10][col] = 1
        U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1
        W[C11][col] = -1
        col += 1
        
        # M3 = A00(B10 - B11)
        U[A00][col] = 1
        V[B10][col] = 1
        V[B11][col] = -1
        W[C01][col] = 1
        W[C11][col] = 1
        col += 1
        
        # M4 = A11(B01 - B00)
        U[A11][col] = 1
        V[B01][col] = 1
        V[B00][col] = -1
        W[C00][col] = 1
        W[C10][col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[A00][col] = 1
        U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1
        W[C01][col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B10)
        U[A10][col] = 1
        U[A00][col] = -1
        V[B00][col] = 1
        V[B10][col] = 1
        W[C11][col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B01 + B11)
        U[A01][col] = 1
        U[A11][col] = -1
        V[B01][col] = 1
        V[B11][col] = 1
        W[C00][col] = 1
        col += 1
    
    # Add 4 Strassen blocks
    add_strassen_block(0, 0, 0)  # i{0,1}, j{0,1}, k{0,1}
    add_strassen_block(0, 2, 0)  # i{0,1}, j{2,3}, k{0,1}
    add_strassen_block(0, 0, 2)  # i{0,1}, j{0,1}, k{2,3}
    add_strassen_block(0, 2, 2)  # i{0,1}, j{2,3}, k{2,3}
    
    # Schoolbook terms for remaining (i,j,k) triples
    # Covered by Strassen: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if covered by Strassen blocks
                if i in (0, 1) and j in (0, 1, 2, 3) and k in (0, 1, 2, 3):
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
