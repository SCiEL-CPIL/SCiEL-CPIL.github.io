"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 52 decomposition
    # Split p=7 into 4 and 3.
    # Part 1: 2x4 * 4x4 (Rank 28)
    # Part 2: 2x4 * 4x3 (Rank 24)
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper: Add Strassen block for 2x2 * 2x2
    # i_base, j_base, k_base are offsets in the global matrices
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        # M1: (A00+A11)(B00+B11)
        U[i_base*m + j_base][col] = 1
        U[(i_base+1)*m + (j_base+1)][col] = 1
        V[j_base*p + k_base][col] = 1
        V[(j_base+1)*p + (k_base+1)][col] = 1
        W[k_base*n + i_base][col] = 1
        W[(k_base+1)*n + (i_base+1)][col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        U[(i_base+1)*m + j_base][col] = 1
        U[(i_base+1)*m + (j_base+1)][col] = 1
        V[j_base*p + k_base][col] = 1
        W[k_base*n + (i_base+1)][col] = 1
        col += 1
        
        # M3: A00(B01-B11)
        U[i_base*m + j_base][col] = 1
        V[j_base*p + (k_base+1)][col] = 1
        V[(j_base+1)*p + (k_base+1)][col] = -1
        W[(k_base+1)*n + i_base][col] = 1
        col += 1
        
        # M4: A11(B10-B00)
        U[(i_base+1)*m + (j_base+1)][col] = 1
        V[(j_base+1)*p + k_base][col] = 1
        V[j_base*p + k_base][col] = -1
        W[k_base*n + i_base][col] = -1
        W[k_base*n + (i_base+1)][col] = 1
        col += 1
        
        # M5: (A00+A01)B11
        U[i_base*m + j_base][col] = 1
        U[i_base*m + (j_base+1)][col] = 1
        V[(j_base+1)*p + (k_base+1)][col] = 1
        W[(k_base+1)*n + i_base][col] = 1
        W[(k_base+1)*n + (i_base+1)][col] = -1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[(i_base+1)*m + j_base][col] = 1
        U[i_base*m + j_base][col] = -1
        V[j_base*p + k_base][col] = 1
        V[j_base*p + (k_base+1)][col] = 1
        W[k_base*n + i_base][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[i_base*m + (j_base+1)][col] = 1
        U[(i_base+1)*m + (j_base+1)][col] = -1
        V[(j_base+1)*p + k_base][col] = 1
        V[(j_base+1)*p + (k_base+1)][col] = 1
        W[k_base*n + i_base][col] = -1
        col += 1

    # Helper: Add schoolbook block for 2x2 * 2x3
    def add_schoolbook_2x2x3(i_base, j_base, k_base):
        nonlocal col
        # A is 2x2, B is 2x3
        # C is 2x3
        # 12 terms
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[(i_base+i)*m + (j_base+j)][col] = 1
                    V[(j_base+j)*p + (k_base+k)][col] = 1
                    W[(k_base+k)*n + (i_base+i)][col] = 1
                    col += 1

    # Part 1: 2x4 * 4x4
    # Split A (2x4) into A1 (2x2, j=0,1) and A2 (2x2, j=2,3)
    # Split B1 (4x4) into B11 (2x4, j=0,1) and B12 (2x4, j=2,3)
    # C1 = A1 B11 + A2 B12
    
    # A1 B11:
    # B11 is 2x4. Split into B11a (2x2, k=0,1) and B11b (2x2, k=2,3)
    # A1 B11a: Strassen
    add_strassen(0, 0, 0)
    # A1 B11b: Strassen
    add_strassen(0, 0, 2)
    
    # A2 B12:
    # B12 is 2x4. Split into B12a (2x2, k=0,1) and B12b (2x2, k=2,3)
    # A2 B12a: Strassen
    add_strassen(0, 2, 0)
    # A2 B12b: Strassen
    add_strassen(0, 2, 2)
    
    # Part 2: 2x4 * 4x3
    # Split A into A1 (j=0,1) and A2 (j=2,3)
    # Split B2 (4x3) into B21 (2x3, j=0,1) and B22 (2x3, j=2,3)
    # C2 = A1 B21 + A2 B22
    
    # A1 B21: Schoolbook 2x2 * 2x3
    add_schoolbook_2x2x3(0, 0, 4)
    
    # A2 B22: Schoolbook 2x2 * 2x3
    add_schoolbook_2x2x3(0, 2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
