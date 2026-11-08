"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank estimated: 88
    # Part 1: 2x2x8 using Strassen (4 * 7 = 28)
    # Part 2: 2x2x8 using Strassen (4 * 7 = 28)
    # Part 3: 1x4x8 using Schoolbook (32)
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2x2 block
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Strassen 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices in local 2x2x2
        # A: (0,0), (0,1), (1,0), (1,1) -> global i_off+di, j_off+dj
        # B: (0,0), (0,1), (1,0), (1,1) -> global j_off+dj, k_off+dk
        # C: (0,0), (0,1), (1,0), (1,1) -> global i_off+di, k_off+dk
        
        # We define the 7 terms as (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: dict mapping (di, dj) to coeff
        # V_coeffs: dict mapping (dj, dk) to coeff
        # W_coeffs: dict mapping (di, dk) to coeff
        
        terms = []
        
        # M1
        terms.append((
            {(0,0):1, (1,1):1}, 
            {(0,0):1, (1,1):1}, 
            {(0,0):1, (1,1):1} # C11+=M1, C22+=M1
        ))
        # Actually, we need to add to specific C positions.
        # It's easier to just list the contributions to U, V, W for each of the 7 columns.
        
        # Let's just hardcode the 7 columns for a 2x2x2 Strassen block.
        # Each column c in U, V, W corresponds to one multiplication.
        
        # Col 0: M1
        # U: A11, A22
        # V: B11, B22
        # W: C11, C22
        U[i_off*4 + j_off][col] = 1
        U[(i_off+1)*4 + (j_off+1)][col] = 1
        V[(j_off)*8 + k_off][col] = 1
        V[(j_off+1)*8 + (k_off+1)][col] = 1
        W[k_off*3 + i_off][col] = 1
        W[(k_off+1)*3 + (i_off+1)][col] = 1
        col += 1
        
        # Col 1: M2 = (A21+A22)B11
        U[(i_off+1)*4 + j_off][col] = 1
        U[(i_off+1)*4 + (j_off+1)][col] = 1
        V[(j_off)*8 + k_off][col] = 1
        W[(k_off)*3 + (i_off+1)][col] = 1 # C21 += M2
        col += 1
        
        # Col 2: M3 = A11(B12-B22)
        U[i_off*4 + j_off][col] = 1
        V[(j_off)*8 + (k_off+1)][col] = 1
        V[(j_off+1)*8 + (k_off+1)][col] = -1
        W[(k_off+1)*3 + i_off][col] = 1 # C12 += M3
        col += 1
        
        # Col 3: M4 = A22(B21-B11)
        U[(i_off+1)*4 + (j_off+1)][col] = 1
        V[(j_off+1)*8 + k_off][col] = 1
        V[(j_off)*8 + k_off][col] = -1
        W[k_off*3 + (i_off+1)][col] = 1 # C21 += M4
        W[(k_off+1)*3 + (i_off+1)][col] = 1 # C22 += M4
        col += 1
        
        # Col 4: M5 = (A11+A12)B22
        U[i_off*4 + j_off][col] = 1
        U[i_off*4 + (j_off+1)][col] = 1
        V[(j_off+1)*8 + (k_off+1)][col] = 1
        W[k_off*3 + i_off][col] = -1 # C11 -= M5
        W[(k_off+1)*3 + i_off][col] = 1 # C12 += M5
        col += 1
        
        # Col 5: M6 = (A21-A11)(B11+B12)
        U[(i_off+1)*4 + j_off][col] = 1
        U[i_off*4 + j_off][col] = -1
        V[(j_off)*8 + k_off][col] = 1
        V[(j_off)*8 + (k_off+1)][col] = 1
        W[(k_off+1)*3 + (i_off+1)][col] = 1 # C22 += M6
        col += 1
        
        # Col 6: M7 = (A12-A22)(B21+B22)
        U[i_off*4 + (j_off+1)][col] = 1
        U[(i_off+1)*4 + (j_off+1)][col] = -1
        V[(j_off+1)*8 + k_off][col] = 1
        V[(j_off+1)*8 + (k_off+1)][col] = 1
        W[k_off*3 + i_off][col] = 1 # C11 += M7
        col += 1
        
        # Check W contributions for C11, C12, C21, C22
        # C11 = M1 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's re-verify the W assignments above.
        # Col 0 (M1): W[C11]=1, W[C22]=1. Correct.
        # Col 1 (M2): W[C21]=1. Correct.
        # Col 2 (M3): W[C12]=1. Correct.
        # Col 3 (M4): W[C21]=1, W[C22]=1. Correct.
        # Col 4 (M5): W[C11]=-1, W[C12]=1. Correct.
        # Col 5 (M6): W[C22]=1. Correct.
        # Col 6 (M7): W[C11]=1. Correct.
        
        # Wait, C22 formula: M1 - M2 + M3 + M6.
        # My assignments:
        # M1: +1
        # M2: 0 (I only assigned C21)
        # M3: 0 (I only assigned C12)
        # M6: +1
        # Missing -M2 and +M3 for C22?
        
        # Let's re-read Strassen formulas.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # My previous assignments:
        # Col 0 (M1): C11=1, C22=1.
        # Col 1 (M2): C21=1. (Need C22=-1)
        # Col 2 (M3): C12=1. (Need C22=1)
        # Col 3 (M4): C21=1, C22=1. (Need C11=1)
        # Col 4 (M5): C11=-1, C12=1.
        # Col 5 (M6): C22=1.
        # Col 6 (M7): C11=1.
        
        # Corrections needed:
        # Col 1 (M2): Add W[(k_off+1)*3 + (i_off+1)][col] = -1
        # Col 2 (M3): Add W[(k_off+1)*3 + (i_off+1)][col] = 1
        # Col 3 (M4): Add W[k_off*3 + i_off][col] = 1
        
        # I will implement the corrected version in the code below.
        
    # Re-implementing add_strassen with corrections
    
    def add_strassen_corrected(i_off, j_off, k_off):
        nonlocal col
        
        # M1
        U[i_off*4 + j_off][col] = 1
        U[(i_off+1)*4 + (j_off+1)][col] = 1
        V[(j_off)*8 + k_off][col] = 1
        V[(j_off+1)*8 + (k_off+1)][col] = 1
        W[k_off*3 + i_off][col] = 1       # C11
        W[(k_off+1)*3 + (i_off+1)][col] = 1 # C22
        col += 1
        
        # M2
        U[(i_off+1)*4 + j_off][col] = 1
        U[(i_off+1)*4 + (j_off+1)][col] = 1
        V[(j_off)*8 + k_off][col] = 1
        W[k_off*3 + (i_off+1)][col] = 1       # C21
        W[(k_off+1)*3 + (i_off+1)][col] = -1  # C22
        col += 1
        
        # M3
        U[i_off*4 + j_off][col] = 1
        V[(j_off)*8 + (k_off+1)][col] = 1
        V[(j_off+1)*8 + (k_off+1)][col] = -1
        W[(k_off+1)*3 + i_off][col] = 1       # C12
        W[(k_off+1)*3 + (i_off+1)][col] = 1   # C22
        col += 1
        
        # M4
        U[(i_off+1)*4 + (j_off+1)][col] = 1
        V[(j_off+1)*8 + k_off][col] = 1
        V[(j_off)*8 + k_off][col] = -1
        W[k_off*3 + (i_off+1)][col] = 1       # C21
        W[(k_off+1)*3 + (i_off+1)][col] = 1   # C22
        W[k_off*3 + i_off][col] = 1           # C11
        col += 1
        
        # M5
        U[i_off*4 + j_off][col] = 1
        U[i_off*4 + (j_off+1)][col] = 1
        V[(j_off+1)*8 + (k_off+1)][col] = 1
        W[k_off*3 + i_off][col] = -1          # C11
        W[(k_off+1)*3 + i_off][col] = 1       # C12
        col += 1
        
        # M6
        U[(i_off+1)*4 + j_off][col] = 1
        U[i_off*4 + j_off][col] = -1
        V[(j_off)*8 + k_off][col] = 1
        V[(j_off)*8 + (k_off+1)][col] = 1
        W[(k_off+1)*3 + (i_off+1)][col] = 1   # C22
        col += 1
        
        # M7
        U[i_off*4 + (j_off+1)][col] = 1
        U[(i_off+1)*4 + (j_off+1)][col] = -1
        V[(j_off+1)*8 + k_off][col] = 1
        V[(j_off+1)*8 + (k_off+1)][col] = 1
        W[k_off*3 + i_off][col] = 1           # C11
        col += 1

    # Part 1: A[0:2, 0:2] * B[0:2, 0:8]
    # Split B[0:2, 0:8] into 4 blocks of 2x2
    for k_block in range(4):
        k_off = k_block * 2
        add_strassen_corrected(0, 0, k_off)
        
    # Part 2: A[0:2, 2:4] * B[2:4, 0:8]
    # Split B[2:4, 0:8] into 4 blocks of 2x2
    for k_block in range(4):
        k_off = k_block * 2
        add_strassen_corrected(0, 2, k_off)
        
    # Part 3: A[2:3, 0:4] * B[0:4, 0:8]
    # Schoolbook
    for j in range(4):
        for k in range(8):
            U[2*4 + j][col] = 1
            V[j*8 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
