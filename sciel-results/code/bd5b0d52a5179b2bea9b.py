"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 4, 7
    # Rank 52 decomposition:
    # 2x4x4 part: Rank 28 (4 Strassen 2x2x2)
    # 2x4x3 part: Rank 24 (Schoolbook 2x2x1)
    
    rank = 52
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # A block at i0, j0 (2x2)
    # B block at j0, k0 (2x2)
    # C block at k0, i0 (2x2)
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Strassen mults for X=A[i0:i0+2, j0:j0+2], Y=B[j0:j0+2, k0:k0+2]
        # X00=A[i0,j0], X01=A[i0,j0+1], X10=A[i0+1,j0], X11=A[i0+1,j0+1]
        # Y00=B[j0,k0], Y01=B[j0,k0+1], Y10=B[j0+1,k0], Y11=B[j0+1,k0+1]
        
        # M1 = (X00+X11)(Y00+Y11)
        U[i0*m + j0, col] = 1
        U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + k0, col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        # W contributions:
        # Z00 = M1 + ... -> W[k0*n + i0, col] = 1
        # Z01 = ...
        # Z10 = ...
        # Z11 = M1 - ... -> W[(k0+1)*n + (i0+1), col] = 1
        W[k0*n + i0, col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # M2 = (X10+X11)Y00
        U[(i0+1)*m + j0, col] = 1
        U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + k0, col] = 1
        # Z01 = M2 + M3 -> W[k0*n + (i0+1), col] = 1 ? No.
        # Z01 = M2 + M3. Z10 = M2 + M4.
        # So W[k0*n + (i0+1), col] = 1 (for Z01)
        # W[(k0+1)*n + i0, col] = 1 (for Z10)
        W[k0*n + (i0+1), col] = 1
        W[(k0+1)*n + i0, col] = 1
        col += 1
        
        # M3 = X00(Y10-Y11)
        U[i0*m + j0, col] = 1
        V[(j0+1)*p + k0, col] = 1
        V[(j0+1)*p + (k0+1), col] = -1
        # Z01 = M2 + M3 -> W[k0*n + (i0+1), col] = 1
        # Z11 = M1 - M3 + ... -> W[(k0+1)*n + (i0+1), col] = -1
        W[k0*n + (i0+1), col] = 1
        W[(k0+1)*n + (i0+1), col] = -1
        col += 1
        
        # M4 = X11(Y01-Y00)
        U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + (k0+1), col] = 1
        V[j0*p + k0, col] = -1
        # Z00 = M1 + M4 - ... -> W[k0*n + i0, col] = 1
        # Z10 = M2 + M4 -> W[(k0+1)*n + i0, col] = 1
        W[k0*n + i0, col] = 1
        W[(k0+1)*n + i0, col] = 1
        col += 1
        
        # M5 = (X00+X10)Y11
        U[i0*m + j0, col] = 1
        U[(i0+1)*m + j0, col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        # Z00 = M1 + M4 - M5 - ... -> W[k0*n + i0, col] = -1
        # Z11 = M1 - M3 + M6 - ... -> W[(k0+1)*n + (i0+1), col] = -1
        W[k0*n + i0, col] = -1
        W[(k0+1)*n + (i0+1), col] = -1
        col += 1
        
        # M6 = (X11-X00)(Y00+Y01)
        U[(i0+1)*m + (j0+1), col] = 1
        U[i0*m + j0, col] = -1
        V[j0*p + k0, col] = 1
        V[j0*p + (k0+1), col] = 1
        # Z11 = M1 - M3 + M6 - ... -> W[(k0+1)*n + (i0+1), col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # M7 = (X01-X11)(Y10+Y11)
        U[i0*m + (j0+1), col] = 1
        U[(i0+1)*m + (j0+1), col] = -1
        V[(j0+1)*p + k0, col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        # Z00 = M1 + M4 - M5 - M7 -> W[k0*n + i0, col] = -1
        # Z11 = M1 - M3 + M6 - M7 -> W[(k0+1)*n + (i0+1), col] = -1
        W[k0*n + i0, col] = -1
        W[(k0+1)*n + (i0+1), col] = -1
        col += 1

    # Part 1: 2x4x4
    # A is 2x4. B1 is 4x4.
    # Split A into A0 (cols 0-1), A1 (cols 2-3).
    # Split B1 into B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3).
    # C1_0 (cols 0-1) = A0 B00 + A1 B10
    # C1_1 (cols 2-3) = A0 B01 + A1 B11
    
    # We need to compute A0 B00, A1 B10, A0 B01, A1 B11.
    # Each is 2x2x2.
    
    # A0 B00: i=0,1; j=0,1; k=0,1
    add_strassen(0, 0, 0)
    
    # A1 B10: i=0,1; j=2,3; k=0,1
    # Note: A1 is cols 2-3 of A. B10 is rows 2-3, cols 0-1 of B1.
    add_strassen(0, 2, 0)
    
    # A0 B01: i=0,1; j=0,1; k=2,3
    add_strassen(0, 0, 2)
    
    # A1 B11: i=0,1; j=2,3; k=2,3
    add_strassen(0, 2, 2)
    
    # Part 2: 2x4x3
    # A is 2x4. B2 is 4x3.
    # Split A into A0 (cols 0-1), A1 (cols 2-3).
    # Split B2 into 3 blocks of 4x1? Or 2x1?
    # Schoolbook for 2x2x1.
    # B2 cols 0, 1, 2.
    # For each col k in 0..2:
    #   For each j_block in 0, 1 (rows 0-1, 2-3):
    #     Compute A_j_block * B2[j_block, k]
    #     This is 2x2x1.
    #     Rank 4.
    
    for k in range(3):
        for j_block in range(2):
            j0 = j_block * 2
            # A0 or A1 (2x2) * B2[j0:j0+2, k] (2x1)
            # 4 mults:
            # A00 * B00
            # A01 * B10
            # A10 * B00
            # A11 * B10
            # Result C[k, 0] += A00 B00 + A01 B10
            # Result C[k, 1] += A10 B00 + A11 B10
            
            # Term 1: A[i0, j0] * B[j0, k] -> C[k, i0]
            U[0*m + j0, col] = 1
            V[j0*p + k, col] = 1
            W[k*n + 0, col] = 1
            col += 1
            
            # Term 2: A[i0, j0+1] * B[j0+1, k] -> C[k, i0]
            U[0*m + (j0+1), col] = 1
            V[(j0+1)*p + k, col] = 1
            W[k*n + 0, col] = 1
            col += 1
            
            # Term 3: A[i0+1, j0] * B[j0, k] -> C[k, i0+1]
            U[1*m + j0, col] = 1
            V[j0*p + k, col] = 1
            W[k*n + 1, col] = 1
            col += 1
            
            # Term 4: A[i0+1, j0+1] * B[j0+1, k] -> C[k, i0+1]
            U[1*m + (j0+1), col] = 1
            V[(j0+1)*p + k, col] = 1
            W[k*n + 1, col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
