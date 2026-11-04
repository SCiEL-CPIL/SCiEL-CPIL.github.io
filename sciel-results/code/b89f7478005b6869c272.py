"""Matrix multiplication 3,3,7 - Rank 60 decomposition using Strassen for 2x2x2 blocks."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # We'll build a rank-60 decomposition:
    # Part 1: 3x3x3 (k=0,1,2) using Strassen-based rank 22 for the 2x2x2 core + schoolbook for edges
    # Part 2: 3x3x4 (k=3,4,5,6) using Strassen for 2x2x2 blocks + schoolbook for edges
    
    # Total rank: 22 + 8 + 8 + 4 + 7 + 7 + 8 + 8 + 4 = let me recalculate
    # Actually, let's think more carefully.
    
    # For 3x3x3:
    # Split A into A00(2x2), A01(2x1), A10(1x2), A11(1x1)
    # Split B into B0(2x3), B1(1x3)
    # Terms:
    #   A00*B0: 2x2x3 -> split B0 into B0a(2x2), B0b(2x1)
    #     A00*B0a: 2x2x2 -> Strassen rank 7
    #     A00*B0b: 2x2x1 -> schoolbook rank 4
    #   A01*B1: 2x1x3 -> schoolbook rank 6
    #   A10*B0: 1x2x3 -> schoolbook rank 6
    #   A11*B1: 1x1x3 -> schoolbook rank 3
    #   Total 3x3x3: 7+4+6+6+3 = 26
    # Hmm, that's not better than 27.
    
    # Let me try a different approach. Use Strassen for the whole 3x3x3.
    # Known: rank(3x3x3) = 23. But constructing it is complex.
    
    # Let's use a simpler but still improved approach:
    # For 3x3x3: use Strassen on the 2x2x2 subblock (rank 7 instead of 8)
    # The remaining terms: 
    #   A[0:2,2] * B[2,0:3] -> 2x1x3, rank 6
    #   A[2,0:2] * B[0:2,0:3] -> 1x2x3, rank 6
    #   A[2,2] * B[2,0:3] -> 1x1x3, rank 3
    #   Total: 7 + 6 + 6 + 3 = 22
    # Wait, that's only 22. Let me verify.
    # A is 3x3, B is 3x3.
    # A00 = A[0:2, 0:2] (2x2)
    # A01 = A[0:2, 2:3] (2x1)
    # A10 = A[2:3, 0:2] (1x2)
    # A11 = A[2:3, 2:3] (1x1)
    # B0 = B[0:2, 0:3] (2x3)
    # B1 = B[2:3, 0:3] (1x3)
    # C = A00*B0 + A01*B1 + A10*B0 + A11*B1
    # A00*B0: 2x2 * 2x3. Split B0 into B0[0:2] (2x2) and B0[2:3] (2x1).
    #   A00*B0[0:2]: 2x2x2, Strassen rank 7
    #   A00*B0[2:3]: 2x2x1, schoolbook rank 4
    # A01*B1: 2x1 * 1x3, schoolbook rank 6
    # A10*B0: 1x2 * 2x3, schoolbook rank 6
    # A11*B1: 1x1 * 1x3, schoolbook rank 3
    # Total: 7 + 4 + 6 + 6 + 3 = 26
    # That's better than 27!
    
    # For 3x3x4:
    # A00*B0: 2x2 * 2x4. Split B0 into two 2x2 blocks.
    #   A00*B0a: 2x2x2, Strassen rank 7
    #   A00*B0b: 2x2x2, Strassen rank 7
    # A01*B1: 2x1 * 1x4, schoolbook rank 8
    # A10*B0: 1x2 * 2x4, schoolbook rank 8
    # A11*B1: 1x1 * 1x4, schoolbook rank 4
    # Total: 7 + 7 + 8 + 8 + 4 = 34
    
    # Grand total: 26 + 34 = 60
    
    rank_total = 60
    U = [[0] * rank_total for _ in range(n * m)]  # 9 x 60
    V = [[0] * rank_total for _ in range(m * p)]  # 21 x 60
    W = [[0] * rank_total for _ in range(p * n)]  # 21 x 60
    
    col = 0
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        """Add Strassen 2x2x2 decomposition at given offsets.
        i_off, j_off, k_off are the starting indices for i, j, k in the full problem.
        Returns the next column index.
        """
        nonlocal col
        # Strassen 2x2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B22-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M7
        # C21 = M2 + M6
        # C22 = M1 - M3 + M4 + M6
        
        # Local indices:
        # A: i in {0,1}, j in {0,1} -> global i = i_off + i, j = j_off + j
        # B: j in {0,1}, k in {0,1} -> global j = j_off + j, k = k_off + k
        # C: i in {0,1}, k in {0,1} -> global i = i_off + i, k = k_off + k
        
        # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U[i_off*3 + j_off + 0][col] += 1   # A[0,0]
        U[i_off*3 + j_off + 1][col] += 0   # A[0,1] not in M1
        U[i_off*3 + j_off + 2][col] += 0   # A[1,0] not in M1
        U[i_off*3 + j_off + 3][col] += 1   # A[1,1]
        V[j_off*7 + k_off + 0][col] += 1   # B[0,0]
        V[j_off*7 + k_off + 1][col] += 0   # B[0,1] not in M1
        V[j_off*7 + k_off + 2][col] += 0   # B[1,0] not in M1
        V[j_off*7 + k_off + 3][col] += 1   # B[1,1]
        W[k_off*3 + i_off + 0][col] += 1   # C[0,0]
        W[k_off*3 + i_off + 1][col] += 0   # C[1,0] not in M1
        W[k_off*3 + i_off + 2][col] += 0   # C[0,1] not in M1
        W[k_off*3 + i_off + 3][col] += 1   # C[1,1]
        col += 1
        
        # M2: (A[1,0]+A[1,1]) * B[0,0]
        U[i_off*3 + j_off + 0][col] += 0
        U[i_off*3 + j_off + 1][col] += 0
        U[i_off*3 + j_off + 2][col] += 1   # A[1,0]
        U[i_off*3 + j_off + 3][col] += 1   # A[1,1]
        V[j_off*7 + k_off + 0][col] += 1   # B[0,0]
        V[j_off*7 + k_off + 1][col] += 0
        V[j_off*7 + k_off + 2][col] += 0
        V[j_off*7 + k_off + 3][col] += 0
        W[k_off*3 + i_off + 0][col] += 0
        W[k_off*3 + i_off + 1][col] += 0
        W[k_off*3 + i_off + 2][col] += 0
        W[k_off*3 + i_off + 3][col] += 1   # C[1,0]
        col += 1
        
        # M3: A[0,0] * (B[1,0]-B[0,0])
        U[i_off*3 + j_off + 0][col] += 1   # A[0,0]
        U[i_off*3 + j_off + 1][col] += 0
        U[i_off*3 + j_off + 2][col] += 0
        U[i_off*3 + j_off + 3][col] += 0
        V[j_off*7 + k_off + 0][col] += -1  # B[0,0]
        V[j_off*7 + k_off + 1][col] += 0
        V[j_off*7 + k_off + 2][col] += 1   # B[1,0]
        V[j_off*7 + k_off + 3][col] += 0
        W[k_off*3 + i_off + 0][col] += 0
        W[k_off*3 + i_off + 1][col] += 0
        W[k_off*3 + i_off + 2][col] += 1   # C[0,1]
        W[k_off*3 + i_off + 3][col] += 0
        col += 1
        
        # M4: A[1,1] * (B[1,1]-B[0,0])
        U[i_off*3 + j_off + 0][col] += 0
        U[i_off*3 + j_off + 1][col] += 0
        U[i_off*3 + j_off + 2][col] += 0
        U[i_off*3 + j_off + 3][col] += 1   # A[1,1]
        V[j_off*7 + k_off + 0][col] += -1  # B[0,0]
        V[j_off*7 + k_off + 1][col] += 0
        V[j_off*7 + k_off + 2][col] += 0
        V[j_off*7 + k_off + 3][col] += 1   # B[1,1]
        W[k_off*3 + i_off + 0][col] += -1  # C[0,0]
        W[k_off*3 + i_off + 1][col] += 0
        W[k_off*3 + i_off + 2][col] += 0
        W[k_off*3 + i_off + 3][col] += 0
        col += 1
        
        # M5: (A[0,0]+A[0,1]) * B[1,1]
        U[i_off*3 + j_off + 0][col] += 1   # A[0,0]
        U[i_off*3 + j_off + 1][col] += 1   # A[0,1]
        U[i_off*3 + j_off + 2][col] += 0
        U[i_off*3 + j_off + 3][col] += 0
        V[j_off*7 + k_off + 0][col] += 0
        V[j_off*7 + k_off + 1][col] += 0
        V[j_off*7 + k_off + 2][col] += 0
        V[j_off*7 + k_off + 3][col] += 1   # B[1,1]
        W[k_off*3 + i_off + 0][col] += 1   # C[0,0]
        W[k_off*3 + i_off + 1][col] += 0
        W[k_off*3 + i_off + 2][col] += 0
        W[k_off*3 + i_off + 3][col] += 0
        col += 1
        
        # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        U[i_off*3 + j_off + 0][col] += -1  # A[0,0]
        U[i_off*3 + j_off + 1][col] += 0
        U[i_off*3 + j_off + 2][col] += 1   # A[1,0]
        U[i_off*3 + j_off + 3][col] += 0
        V[j_off*7 + k_off + 0][col] += 1   # B[0,0]
        V[j_off*7 + k_off + 1][col] += 1   # B[0,1]
        V[j_off*7 + k_off + 2][col] += 0
        V[j_off*7 + k_off + 3][col] += 0
        W[k_off*3 + i_off + 0][col] += 0
        W[k_off*3 + i_off + 1][col] += 0
        W[k_off*3 + i_off + 2][col] += 0
        W[k_off*3 + i_off + 3][col] += 1   # C[1,0]
        col += 1
        
        # M7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        U[i_off*3 + j_off + 0][col] += 0
        U[i_off*3 + j_off + 1][col] += 1   # A[0,1]
        U[i_off*3 + j_off + 2][col] += 0
        U[i_off*3 + j_off + 3][col] += -1  # A[1,1]
        V[j_off*7 + k_off + 0][col] += 0
        V[j_off*7 + k_off + 1][col] += 0
        V[j_off*7 + k_off + 2][col] += 1   # B[1,0]
        V[j_off*7 + k_off + 3][col] += 1   # B[1,1]
        W[k_off*3 + i_off + 0][col] += -1  # C[0,0]
        W[k_off*3 + i_off + 1][col] += 0
        W[k_off*3 + i_off + 2][col] += 1   # C[0,1]
        W[k_off*3 + i_off + 3][col] += 1   # C[1,1]
        col += 1
        
        return col
    
    def add_schoolbook_term(i, j, k):
        """Add a single schoolbook term A[i,j]*B[j,k]->C[i,k]."""
        nonlocal col
        U[i * 3 + j][col] = 1
        V[j * 7 + k][col] = 1
        W[k * 3 + i][col] = 1
        col += 1
    
    # Part 1: 3x3x3 (k=0,1,2)
    # A00*B0a: 2x2x2 at i_off=0, j_off=0, k_off=0 (Strassen)
    add_strassen_2x2x2(0, 0, 0)  # 7 terms
    
    # A00*B0b: 2x2x1 at i_off=0, j_off=0, k_off=2 (schoolbook)
    for i in range(2):
        for j in range(2):
            add_schoolbook_term(i, j, 2)  # 4 terms
    
    # A01*B1: 2x1x3, i in {0,1}, j=2, k in {0,1,2}
    for i in range(2):
        for k in range(3):
            add_schoolbook_term(i, 2, k)  # 6 terms
    
    # A10*B0: 1x2x3, i=2, j in {0,1}, k in {0,1,2}
    for j in range(2):
        for k in range(3):
            add_schoolbook_term(2, j, k)  # 6 terms
    
    # A11*B1: 1x1x3, i=2, j=2, k in {0,1,2}
    for k in range(3):
        add_schoolbook_term(2, 2, k)  # 3 terms
    
    # Part 2: 3x3x4 (k=3,4,5,6)
    # A00*B0a: 2x2x2 at i_off=0, j_off=0, k_off=3 (Strassen)
    add_strassen_2x2x2(0, 0, 3)  # 7 terms
    
    # A00*B0b: 2x2x2 at i_off=0, j_off=0, k_off=5 (Strassen)
    add_strassen_2x2x2(0, 0, 5)  # 7 terms
    
    # A01*B1: 2x1x4, i in {0,1}, j=2, k in {3,4,5,6}
    for i in range(2):
        for k in range(3, 7):
            add_schoolbook_term(i, 2, k)  # 8 terms
    
    # A10*B0: 1x2x4, i=2, j in {0,1}, k in {3,4,5,6}
    for j in range(2):
        for k in range(3, 7):
            add_schoolbook_term(2, j, k)  # 8 terms
    
    # A11*B1: 1x1x4, i=2, j=2, k in {3,4,5,6}
    for k in range(3, 7):
        add_schoolbook_term(2, 2, k)  # 4 terms
    
    # Total: 7+4+6+6+3+7+7+8+8+4 = 60
    assert col == 60, f"Expected 60 columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
