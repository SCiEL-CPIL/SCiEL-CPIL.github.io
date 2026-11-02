"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will use a hybrid decomposition:
    # Split indices i in {0,1} vs {2}
    # Split indices j in {0,1} vs {2,3,4}
    # Split indices k in {0,1} vs {2,3,4,5}
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 using Strassen (Rank 7)
    # Other blocks: Schoolbook
    
    # Total Rank Calculation:
    # Block 1 (Strassen): 7
    # Block 2: i<2, j<2, k>=2 -> 2x2x4 -> 16
    # Block 3: i<2, j>=2, k<2 -> 2x3x2 -> 12
    # Block 4: i<2, j>=2, k>=2 -> 2x3x4 -> 24
    # Block 5: i=2, j<2, k<2 -> 1x2x2 -> 4
    # Block 6: i=2, j<2, k>=2 -> 1x2x4 -> 8
    # Block 7: i=2, j>=2, k<2 -> 1x3x2 -> 6
    # Block 8: i=2, j>=2, k>=2 -> 1x3x4 -> 12
    
    rank = 7 + 16 + 12 + 24 + 4 + 8 + 6 + 12
    # rank = 89
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add schoolbook block
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # Helper to add Strassen block for 2x2x2
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    def add_strassen_2x2x2():
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U, V, W contributions for each M
        
        # M1
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + 0][col] = 1
        V[1 * p + 1][col] = 1
        W[0 * n + 0][col] = 1
        W[1 * n + 1][col] = 1
        col += 1
        
        # M2
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + 0][col] = 1
        W[1 * n + 0][col] = 1
        W[1 * n + 1][col] = 1
        col += 1
        
        # M3
        U[0 * m + 0][col] = 1
        V[1 * p + 0][col] = 1
        V[1 * p + 1][col] = -1
        W[0 * n + 1][col] = 1
        W[1 * n + 0][col] = 1
        col += 1
        
        # M4
        U[1 * m + 1][col] = 1
        V[0 * p + 1][col] = 1
        V[0 * p + 0][col] = -1
        W[0 * n + 0][col] = 1
        W[1 * n + 1][col] = 1
        col += 1
        
        # M5
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + 1][col] = 1
        W[0 * n + 0][col] = -1
        W[1 * n + 1][col] = 1
        col += 1
        
        # M6
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + 0][col] = 1
        V[0 * p + 1][col] = 1
        W[1 * n + 0][col] = 1
        W[1 * n + 1][col] = 1
        col += 1
        
        # M7
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + 0][col] = 1
        V[1 * p + 1][col] = 1
        W[0 * n + 0][col] = 1
        W[1 * n + 1][col] = -1
        col += 1

    # 1. Strassen for i<2, j<2, k<2
    add_strassen_2x2x2()
    
    # 2. Schoolbook for i<2, j<2, k>=2
    add_schoolbook(0, 2, 0, 2, 2, 6)
    
    # 3. Schoolbook for i<2, j>=2, k<2
    add_schoolbook(0, 2, 2, 5, 0, 2)
    
    # 4. Schoolbook for i<2, j>=2, k>=2
    add_schoolbook(0, 2, 2, 5, 2, 6)
    
    # 5. Schoolbook for i=2, j<2, k<2
    add_schoolbook(2, 3, 0, 2, 0, 2)
    
    # 6. Schoolbook for i=2, j<2, k>=2
    add_schoolbook(2, 3, 0, 2, 2, 6)
    
    # 7. Schoolbook for i=2, j>=2, k<2
    add_schoolbook(2, 3, 2, 5, 0, 2)
    
    # 8. Schoolbook for i=2, j>=2, k>=2
    add_schoolbook(2, 3, 2, 5, 2, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
