import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition to reduce rank from 120 to 112.
    # A (4x5) split into A1 (4x2) and A2 (4x3)
    # B (5x6) split into B1 (2x6) and B2 (3x6)
    # C = A1 B1 + A2 B2
    
    # A1 B1 (4x2x6):
    # A1 split into A1a (2x2), A1b (2x2)
    # B1 split into B1a (2x3), B1b (2x3)
    # 4 blocks of (2x2x3), each rank 11. Total 44.
    
    # A2 B2 (4x3x6):
    # A2 split into A2a (2x3), A2b (2x3)
    # B2 split into B2a (3x3), B2b (3x3)
    # 4 blocks of (2x3x3), each rank 17. Total 68.
    
    # Total Rank = 44 + 68 = 112.
    
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x3 block
    # A is 2x2, B is 2x3, C is 2x3
    # A indices: rows i0, i1; cols j0, j1
    # B indices: rows j0, j1; cols k0, k1, k2
    # C indices: rows i0, i1; cols k0, k1, k2
    def add_2x2x3_block(i0, j0, k0):
        nonlocal col
        # A elements
        # A[0,0] -> U[i0*m + j0]
        # A[0,1] -> U[i0*m + j0+1]
        # A[1,0] -> U[(i0+1)*m + j0]
        # A[1,1] -> U[(i0+1)*m + j0+1]
        
        # B elements
        # B[0,0] -> V[j0*p + k0]
        # B[0,1] -> V[j0*p + k0+1]
        # B[0,2] -> V[j0*p + k0+2]
        # B[1,0] -> V[(j0+1)*p + k0]
        # B[1,1] -> V[(j0+1)*p + k0+1]
        # B[1,2] -> V[(j0+1)*p + k0+2]
        
        # Strassen 2x2x2 part (cols k0, k0+1)
        # M1 = a11 b11
        U[i0*m + j0][col] = 1
        V[j0*p + k0][col] = 1
        # W contributions:
        # c11 += M1
        W[k0*n + i0][col] = 1
        # c22 -= M1
        W[(k0+1)*n + (i0+1)][col] = -1
        col += 1
        
        # M2 = a11 (b12 + b22)
        U[i0*m + j0][col] = 1
        V[j0*p + k0+1][col] = 1
        V[(j0+1)*p + k0+1][col] = 1
        # c12 += M2
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # M3 = (a21 + a22) b11
        U[(i0+1)*m + j0][col] = 1
        U[(i0+1)*m + j0+1][col] = 1
        V[j0*p + k0][col] = 1
        # c21 += M3
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # M4 = a22 b22
        U[(i0+1)*m + j0+1][col] = 1
        V[(j0+1)*p + k0+1][col] = 1
        # c11 -= M4
        W[k0*n + i0][col] = -1
        # c22 += M4
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # M5 = (a11 + a22) (b12 - b22)
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + j0+1][col] = 1
        V[j0*p + k0+1][col] = 1
        V[(j0+1)*p + k0+1][col] = -1
        # c11 -= M5
        W[k0*n + i0][col] = -1
        # c12 += M5
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # M6 = (a21 - a11) (b11 + b21)
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + k0][col] = 1
        # c21 += M6
        W[k0*n + (i0+1)][col] = 1
        # c22 += M6
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # M7 = (a12 + a22) (b21 - b11)
        U[i0*m + j0+1][col] = 1
        U[(i0+1)*m + j0+1][col] = 1
        V[(j0+1)*p + k0][col] = 1
        V[j0*p + k0][col] = -1
        # c11 += M7
        W[k0*n + i0][col] = 1
        # c22 += M7
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # Schoolbook 2x2x1 part (col k0+2)
        # N1 = a11 b13
        U[i0*m + j0][col] = 1
        V[j0*p + k0+2][col] = 1
        # c13 += N1
        W[(k0+2)*n + i0][col] = 1
        col += 1
        
        # N2 = a12 b23
        U[i0*m + j0+1][col] = 1
        V[(j0+1)*p + k0+2][col] = 1
        # c13 += N2
        W[(k0+2)*n + i0][col] = 1
        col += 1
        
        # N3 = a21 b13
        U[(i0+1)*m + j0][col] = 1
        V[j0*p + k0+2][col] = 1
        # c23 += N3
        W[(k0+2)*n + (i0+1)][col] = 1
        col += 1
        
        # N4 = a22 b23
        U[(i0+1)*m + j0+1][col] = 1
        V[(j0+1)*p + k0+2][col] = 1
        # c23 += N4
        W[(k0+2)*n + (i0+1)][col] = 1
        col += 1

    # Helper to add a 2x3x3 block
    # A is 2x3, B is 3x3, C is 2x3
    # A indices: rows i0, i1; cols j0, j0+1, j0+2
    # B indices: rows j0, j0+1, j0+2; cols k0, k0+1, k0+2
    # C indices: rows i0, i1; cols k0, k0+1, k0+2
    def add_2x3x3_block(i0, j0, k0):
        nonlocal col
        # Split A into A1 (2x2) and A2 (2x1)
        # A1 cols: j0, j0+1
        # A2 col: j0+2
        # Split B into B1 (2x3) and B2 (1x3)
        # B1 rows: j0, j0+1
        # B2 row: j0+2
        
        # Part 1: A1 B1 (2x2x3) -> Rank 11
        # Call add_2x2x3_block for the top-left 2x2x3 part
        # But we need to handle the full 2x3x3.
        # Let's just call the 2x2x3 logic for A1 and B1.
        # A1 is A[i0:i0+2, j0:j0+2]
        # B1 is B[j0:j0+2, k0:k0+3]
        
        # We can reuse the 2x2x3 logic if we map indices correctly.
        # The 2x2x3 block assumes A is 2x2, B is 2x3.
        # Here A1 is 2x2, B1 is 2x3.
        # So we can just call add_2x2x3_block(i0, j0, k0) ?
        # NO. add_2x2x3_block writes to W for C indices.
        # C for A1 B1 is 2x3. Indices i0, i0+1 and k0, k0+1, k0+2.
        # This matches.
        add_2x2x3_block(i0, j0, k0)
        
        # Part 2: A2 B2 (2x1x3) -> Rank 6
        # A2 is 2x1: rows i0, i0+1; col j0+2
        # B2 is 1x3: row j0+2; cols k0, k0+1, k0+2
        
        # P1 = a13 b31
        U[i0*m + j0+2][col] = 1
        V[(j0+2)*p + k0][col] = 1
        W[k0*n + i0][col] = 1
        col += 1
        
        # P2 = a13 b32
        U[i0*m + j0+2][col] = 1
        V[(j0+2)*p + k0+1][col] = 1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # P3 = a13 b33
        U[i0*m + j0+2][col] = 1
        V[(j0+2)*p + k0+2][col] = 1
        W[(k0+2)*n + i0][col] = 1
        col += 1
        
        # P4 = a23 b31
        U[(i0+1)*m + j0+2][col] = 1
        V[(j0+2)*p + k0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # P5 = a23 b32
        U[(i0+1)*m + j0+2][col] = 1
        V[(j0+2)*p + k0+1][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # P6 = a23 b33
        U[(i0+1)*m + j0+2][col] = 1
        V[(j0+2)*p + k0+2][col] = 1
        W[(k0+2)*n + (i0+1)][col] = 1
        col += 1

    # Now assemble the full 4x5x6
    # A1 (4x2): rows 0-3, cols 0-1
    # B1 (2x6): rows 0-1, cols 0-5
    # A1 B1:
    # Split A1 into A1a (2x2, rows 0-1, cols 0-1) and A1b (2x2, rows 2-3, cols 0-1)
    # Split B1 into B1a (2x3, rows 0-1, cols 0-2) and B1b (2x3, rows 0-1, cols 3-5)
    
    # Block 1: A1a B1a (2x2x3)
    add_2x2x3_block(0, 0, 0)
    # Block 2: A1a B1b (2x2x3)
    add_2x2x3_block(0, 0, 3)
    # Block 3: A1b B1a (2x2x3)
    add_2x2x3_block(2, 0, 0)
    # Block 4: A1b B1b (2x2x3)
    add_2x2x3_block(2, 0, 3)
    
    # A2 (4x3): rows 0-3, cols 2-4
    # B2 (3x6): rows 2-4, cols 0-5
    # A2 B2:
    # Split A2 into A2a (2x3, rows 0-1, cols 2-4) and A2b (2x3, rows 2-3, cols 2-4)
    # Split B2 into B2a (3x3, rows 2-4, cols 0-2) and B2b (3x3, rows 2-4, cols 3-5)
    
    # Block 5: A2a B2a (2x3x3)
    add_2x3x3_block(0, 2, 0)
    # Block 6: A2a B2b (2x3x3)
    add_2x3x3_block(0, 2, 3)
    # Block 7: A2b B2a (2x3x3)
    add_2x3x3_block(2, 2, 0)
    # Block 8: A2b B2b (2x3x3)
    add_2x3x3_block(2, 2, 3)
    
    return {"U": U, "V": V, "W": W}

