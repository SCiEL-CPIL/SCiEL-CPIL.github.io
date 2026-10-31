"""4x4 matrix multiplication tensor decomposition using Recursive Strassen (Rank 49)."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add an entry to a matrix
    def add_entry(M, i, j, val):
        M[i][j] += val

    # Define the 7 block products for 4x4 Strassen
    # Each is a pair (X, Y) of 2x2 matrices
    # X and Y entries are linear combinations of A and B entries
    # A indices: A[r][c] -> index r*m+c
    # B indices: B[r][c] -> index r*p+c
    
    # We represent linear combinations as lists of (coeff, index)
    
    # Block A00: A[0][0], A[0][1], A[1][0], A[1][1]
    # Block A11: A[2][2], A[2][3], A[3][2], A[3][3]
    # etc.
    
    def block_A(r, c):
        # Returns 2x2 matrix of linear combos of A
        # r, c are 0 or 1
        base_r = r * 2
        base_c = c * 2
        return [
            [(1, base_r * m + base_c), (1, base_r * m + base_c + 1)], # Row 0: A[br][bc], A[br][bc+1]
            [(1, (base_r + 1) * m + base_c), (1, (base_r + 1) * m + base_c + 1)] # Row 1
        ]
        # This representation is messy for Strassen which needs specific entries.
        # Let's just define the 7 products explicitly using indices.
        
    # Let's use a different approach.
    # We define the 7 block products M1..M7.
    # For each, we define the 2x2 matrix X (from A) and Y (from B).
    # X and Y are 2x2 arrays of linear combinations.
    
    # Indices for A:
    # A00: (0,0), (0,1), (1,0), (1,1)
    # A01: (0,2), (0,3), (1,2), (1,3)
    # A10: (2,0), (2,1), (3,0), (3,1)
    # A11: (2,2), (2,3), (3,2), (3,3)
    
    # Indices for B:
    # B00: (0,0), (0,1), (1,0), (1,1)
    # B01: (0,2), (0,3), (1,2), (1,3)
    # B10: (2,0), (2,1), (3,0), (3,1)
    # B11: (2,2), (2,3), (3,2), (3,3)
    
    def A_idx(r, c): return r * m + c
    def B_idx(r, c): return r * p + c
    
    # Define X and Y for each of the 7 block products
    # X is 2x2, Y is 2x2
    # Each cell is a list of (coeff, index)
    
    def make_A_block(r, c):
        # r, c in {0,1}
        br = r * 2
        bc = c * 2
        return [
            [(1, A_idx(br, bc)), (1, A_idx(br, bc+1))], # This is wrong, Strassen needs individual elements
            # Actually, Strassen on 2x2 blocks treats the block as a matrix.
            # So X[0][0] is a scalar from A.
        ]
        
    # Let's define the 7 products explicitly.
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # We need to extract the 4 scalar entries of X and Y for each product.
    
    products = []
    
    # M1
    X1 = [
        [(1, A_idx(0,0)), (1, A_idx(2,2))], # X1[0][0] = A00[0][0] + A11[0][0]
        [(1, A_idx(0,1)), (1, A_idx(2,3))], # X1[0][1]
        [(1, A_idx(1,0)), (1, A_idx(3,2))], # X1[1][0]
        [(1, A_idx(1,1)), (1, A_idx(3,3))]  # X1[1][1]
    ]
    Y1 = [
        [(1, B_idx(0,0)), (1, B_idx(2,2))],
        [(1, B_idx(0,1)), (1, B_idx(2,3))],
        [(1, B_idx(1,0)), (1, B_idx(3,2))],
        [(1, B_idx(1,1)), (1, B_idx(3,3))]
    ]
    products.append((X1, Y1))
    
    # M2
    X2 = [
        [(1, A_idx(2,0)), (1, A_idx(2,2))], # A10[0][0] + A11[0][0]
        [(1, A_idx(2,1)), (1, A_idx(2,3))],
        [(1, A_idx(3,0)), (1, A_idx(3,2))],
        [(1, A_idx(3,1)), (1, A_idx(3,3))]
    ]
    Y2 = [
        [(1, B_idx(0,0))],
        [(1, B_idx(0,1))],
        [(1, B_idx(1,0))],
        [(1, B_idx(1,1))]
    ]
    products.append((X2, Y2))
    
    # M3
    X3 = [
        [(1, A_idx(0,0))],
        [(1, A_idx(0,1))],
        [(1, A_idx(1,0))],
        [(1, A_idx(1,1))]
    ]
    Y3 = [
        [(1, B_idx(0,2)), (-1, B_idx(2,2))], # B01[0][0] - B11[0][0]
        [(1, B_idx(0,3)), (-1, B_idx(2,3))],
        [(1, B_idx(1,2)), (-1, B_idx(3,2))],
        [(1, B_idx(1,3)), (-1, B_idx(3,3))]
    ]
    products.append((X3, Y3))
    
    # M4
    X4 = [
        [(1, A_idx(2,2))],
        [(1, A_idx(2,3))],
        [(1, A_idx(3,2))],
        [(1, A_idx(3,3))]
    ]
    Y4 = [
        [(1, B_idx(2,0)), (-1, B_idx(0,0))], # B10[0][0] - B00[0][0]
        [(1, B_idx(2,1)), (-1, B_idx(0,1))],
        [(1, B_idx(3,0)), (-1, B_idx(1,0))],
        [(1, B_idx(3,1)), (-1, B_idx(1,1))]
    ]
    products.append((X4, Y4))
    
    # M5
    X5 = [
        [(1, A_idx(0,0)), (1, A_idx(0,2))], # A00[0][0] + A01[0][0]
        [(1, A_idx(0,1)), (1, A_idx(0,3))],
        [(1, A_idx(1,0)), (1, A_idx(1,2))],
        [(1, A_idx(1,1)), (1, A_idx(1,3))]
    ]
    Y5 = [
        [(1, B_idx(2,2))],
        [(1, B_idx(2,3))],
        [(1, B_idx(3,2))],
        [(1, B_idx(3,3))]
    ]
    products.append((X5, Y5))
    
    # M6
    X6 = [
        [(1, A_idx(2,0)), (-1, A_idx(0,0))], # A10[0][0] - A00[0][0]
        [(1, A_idx(2,1)), (-1, A_idx(0,1))],
        [(1, A_idx(3,0)), (-1, A_idx(1,0))],
        [(1, A_idx(3,1)), (-1, A_idx(1,1))]
    ]
    Y6 = [
        [(1, B_idx(0,0)), (1, B_idx(0,2))], # B00[0][0] + B01[0][0]
        [(1, B_idx(0,1)), (1, B_idx(0,3))],
        [(1, B_idx(1,0)), (1, B_idx(1,2))],
        [(1, B_idx(1,1)), (1, B_idx(1,3))]
    ]
    products.append((X6, Y6))
    
    # M7
    X7 = [
        [(1, A_idx(0,2)), (-1, A_idx(2,2))], # A01[0][0] - A11[0][0]
        [(1, A_idx(0,3)), (-1, A_idx(2,3))],
        [(1, A_idx(1,2)), (-1, A_idx(3,2))],
        [(1, A_idx(1,3)), (-1, A_idx(3,3))]
    ]
    Y7 = [
        [(1, B_idx(2,0)), (1, B_idx(2,2))], # B10[0][0] + B11[0][0]
        [(1, B_idx(2,1)), (1, B_idx(2,3))],
        [(1, B_idx(3,0)), (1, B_idx(3,2))],
        [(1, B_idx(3,1)), (1, B_idx(3,3))]
    ]
    products.append((X7, Y7))
    
    # Now, for each of the 7 products, apply Strassen 2x2
    # Strassen 2x2 steps:
    # s1 = (x00 + x11)(y00 + y11)
    # s2 = (x10 + x11)y00
    # s3 = x00(y01 - y11)
    # s4 = x11(y10 - y00)
    # s5 = (x00 + x01)y11
    # s6 = (x10 - x00)(y00 + y01)
    # s7 = (x01 - x11)(y10 + y11)
    
    # Result Z:
    # z00 = s1 + s4 - s5 + s7
    # z01 = s3 + s5
    # z10 = s2 + s4
    # z11 = s1 - s2 + s3 + s6
    
    # We need to map these to U, V, W.
    # For each scalar product s_k, we add a column to U and V.
    # We also need to track which C entries this s_k contributes to.
    
    # Let's store the contributions to W later.
    # For now, build U and V.
    
    col = 0
    # w_contributions: list of (col, coeff, c_index)
    w_contributions = []
    
    for i in range(7):
        X, Y = products[i]
        # X is 2x2, Y is 2x2
        x00, x01, x10, x11 = X[0], X[1], X[2], X[3]
        y00, y01, y10, y11 = Y[0], Y[1], Y[2], Y[3]
        
        # s1
        u1 = [(c, idx) for c, idx in x00] + [(c, idx) for c, idx in x11]
        v1 = [(c, idx) for c, idx in y00] + [(c, idx) for c, idx in y11]
        for c, idx in u1: add_entry(U, idx, col, c)
        for c, idx in v1: add_entry(V, idx, col, c)
        # W contributions: s1 contributes to z00 (+1) and z11 (+1) of block i
        # Block i contributes to C blocks based on Strassen 4x4 formulas
        # We need to know which C blocks block i affects.
        # M1 -> C00 (+1), C11 (+1)
        # M2 -> C10 (+1), C11 (-1)
        # M3 -> C01 (+1), C11 (+1)
        # M4 -> C00 (+1), C10 (+1)
        # M5 -> C00 (-1), C01 (+1)
        # M6 -> C11 (+1)
        # M7 -> C00 (+1)
        
        # Let's define the mapping of block index i to C block contributions
        # C blocks are 2x2.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will handle W mapping in a separate pass or by storing info.
        # It's easier to store (col, coeff, block_idx, z_pos) and then expand.
        # z_pos: 0=z00, 1=z01, 2=z10, 3=z11
        
        # s1 -> z00 (+1), z11 (+1)
        w_contributions.append((col, 1, i, 0))
        w_contributions.append((col, 1, i, 3))
        col += 1
        
        # s2
        u2 = [(c, idx) for c, idx in x10] + [(c, idx) for c, idx in x11]
        v2 = [(c, idx) for c, idx in y00]
        for c, idx in u2: add_entry(U, idx, col, c)
        for c, idx in v2: add_entry(V, idx, col, c)
        # s2 -> z10 (+1), z11 (-1)
        w_contributions.append((col, 1, i, 2))
        w_contributions.append((col, -1, i, 3))
        col += 1
        
        # s3
        u3 = [(c, idx) for c, idx in x00]
        v3 = [(c, idx) for c, idx in y01] + [(-c, idx) for c, idx in y11]
        for c, idx in u3: add_entry(U, idx, col, c)
        for c, idx in v3: add_entry(V, idx, col, c)
        # s3 -> z01 (+1), z11 (+1)
        w_contributions.append((col, 1, i, 1))
        w_contributions.append((col, 1, i, 3))
        col += 1
        
        # s4
        u4 = [(c, idx) for c, idx in x11]
        v4 = [(c, idx) for c, idx in y10] + [(-c, idx) for c, idx in y00]
        for c, idx in u4: add_entry(U, idx, col, c)
        for c, idx in v4: add_entry(V, idx, col, c)
        # s4 -> z00 (+1), z10 (+1)
        w_contributions.append((col, 1, i, 0))
        w_contributions.append((col, 1, i, 2))
        col += 1
        
        # s5
        u5 = [(c, idx) for c, idx in x00] + [(c, idx) for c, idx in x01]
        v5 = [(c, idx) for c, idx in y11]
        for c, idx in u5: add_entry(U, idx, col, c)
        for c, idx in v5: add_entry(V, idx, col, c)
        # s5 -> z00 (-1), z01 (+1)
        w_contributions.append((col, -1, i, 0))
        w_contributions.append((col, 1, i, 1))
        col += 1
        
        # s6
        u6 = [(c, idx) for c, idx in x10] + [(-c, idx) for c, idx in x00]
        v6 = [(c, idx) for c, idx in y00] + [(c, idx) for c, idx in y01]
        for c, idx in u6: add_entry(U, idx, col, c)
        for c, idx in v6: add_entry(V, idx, col, c)
        # s6 -> z11 (+1)
        w_contributions.append((col, 1, i, 3))
        col += 1
        
        # s7
        u7 = [(c, idx) for c, idx in x01] + [(-c, idx) for c, idx in x11]
        v7 = [(c, idx) for c, idx in y10] + [(c, idx) for c, idx in y11]
        for c, idx in u7: add_entry(U, idx, col, c)
        for c, idx in v7: add_entry(V, idx, col, c)
        # s7 -> z00 (+1)
        w_contributions.append((col, 1, i, 0))
        col += 1
        
    # Now build W
    # Block i contributes to C blocks:
    # i=0 (M1): C00 (+1), C11 (+1)
    # i=1 (M2): C10 (+1), C11 (-1)
    # i=2 (M3): C01 (+1), C11 (+1)
    # i=3 (M4): C00 (+1), C10 (+1)
    # i=4 (M5): C00 (-1), C01 (+1)
    # i=5 (M6): C11 (+1)
    # i=6 (M7): C00 (+1)
    
    block_to_c = {
        0: [(0, 1), (3, 1)], # (c_block_idx, coeff)
        1: [(2, 1), (3, -1)],
        2: [(1, 1), (3, 1)],
        3: [(0, 1), (2, 1)],
        4: [(0, -1), (1, 1)],
        5: [(3, 1)],
        6: [(0, 1)]
    }
    
    # c_block_idx: 0=C00, 1=C01, 2=C10, 3=C11
    # z_pos: 0=z00, 1=z01, 2=z10, 3=z11
    # C entry index: C_block_idx * 4 + z_pos?
    # C00 is rows 0-1, cols 0-1.
    # z00 -> C[0][0] (idx 0)
    # z01 -> C[0][1] (idx 1)
    # z10 -> C[1][0] (idx 2)
    # z11 -> C[1][1] (idx 3)
    
    # C01 is rows 0-1, cols 2-3.
    # z00 -> C[0][2] (idx 2)
    # z01 -> C[0][3] (idx 3)
    # z10 -> C[1][2] (idx 6)
    # z11 -> C[1][3] (idx 7)
    
    # C10 is rows 2-3, cols 0-1.
    # z00 -> C[2][0] (idx 8)
    # z01 -> C[2][1] (idx 9)
    # z10 -> C[3][0] (idx 12)
    # z11 -> C[3][1] (idx 13)
    
    # C11 is rows 2-3, cols 2-3.
    # z00 -> C[2][2] (idx 10)
    # z01 -> C[2][3] (idx 11)
    # z10 -> C[3][2] (idx 14)
    # z11 -> C[3][3] (idx 15)
    
    # Wait, the trace convention is W[k*n+i, r].
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This usually means W is indexed by (output_index, rank).
    # Output index for C[r][c] is r*p+c? Or r*n+c?
    # n=4, m=4, p=4.
    # U is (n*m, rank). V is (m*p, rank). W is (p*n, rank).
    # C is p x n.
    # So C index is r*p + c?
    # Standard matrix multiplication C = A * B.
    # A is n x m. B is m x p. C is n x p.
    # Here n=4, m=4, p=4.
    # U is (4*4, rank). V is (4*4, rank). W is (4*4, rank).
    # C index is r*p + c.
    
    # Let's map C block indices to C entry indices.
    # C00 (block 0): C[0][0], C[0][1], C[1][0], C[1][1] -> idx 0, 1, 4, 5
    # C01 (block 1): C[0][2], C[0][3], C[1][2], C[1][3] -> idx 2, 3, 6, 7
    # C10 (block 2): C[2][0], C[2][1], C[3][0], C[3][1] -> idx 8, 9, 12, 13
    # C11 (block 3): C[2][2], C[2][3], C[3][2], C[3][3] -> idx 10, 11, 14, 15
    
    c_block_to_entries = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    for col_idx, coeff, block_idx, z_pos in w_contributions:
        c_blocks = block_to_c[block_idx]
        for c_block, c_coeff in c_blocks:
            entry_idx = c_block_to_entries[c_block][z_pos]
            add_entry(W, entry_idx, col_idx, coeff * c_coeff)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
