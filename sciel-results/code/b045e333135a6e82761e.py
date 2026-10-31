import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block indices for A and B
    # A blocks: 00, 01, 10, 11
    # Indices in flat array (row-major 4x4)
    # A[0:2, 0:2] -> 0, 1, 4, 5
    # A[0:2, 2:4] -> 2, 3, 6, 7
    # A[2:4, 0:2] -> 8, 9, 12, 13
    # A[2:4, 2:4] -> 10, 11, 14, 15
    
    def get_block_indices(base_block):
        # base_block is (row, col) in 0..1
        r, c = base_block
        start = (2 * r) * 4 + (2 * c)
        return [start, start + 1, start + 4, start + 5]

    # Define the 7 macro products
    # Each is defined by X (2x2 matrix of A-blocks) and Y (2x2 matrix of B-blocks)
    # X and Y are 2x2 grids. Each cell is a linear combination of blocks.
    # We represent a linear combination as a list of (block, coeff)
    
    # M1: X = A00 + A11, Y = B00 + B11
    # M2: X = A10 + A11, Y = B01
    # M3: X = A00, Y = B01 - B11
    # M4: X = A11, Y = B10 - B00
    # M5: X = A00 + A01, Y = B11
    # M6: X = A10 - A00, Y = B00 + B10
    # M7: X = A01 - A11, Y = B11 + B01
    
    # We need to handle the 2x2 structure of X and Y for Strassen.
    # For M1, X is a single 2x2 matrix (sum of two blocks).
    # So X[0,0] = A00[0,0] + A11[0,0], etc.
    
    # Let's define the 7 Strassen products for a generic 2x2 product Z = X Y.
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # P1 = (x00 + x11)(y00 + y11)
    # P2 = (x10 + x11)y01
    # P3 = x00(y01 - y11)
    # P4 = x11(y10 - y00)
    # P5 = (x00 + x01)y11
    # P6 = (x10 - x00)(y00 + y10)
    # P7 = (x01 - x11)(y11 + y01)
    
    # Contributions to Z:
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # For each macro product k, we have X_k and Y_k.
    # X_k is a 2x2 matrix of "A-terms".
    # Each A-term is a linear combination of A's entries.
    
    # Let's define the A-terms for each cell of X_k and Y_k.
    
    # Helper to add a block to a vector
    def add_block(vec, block, coeff):
        idxs = get_block_indices(block)
        for idx in idxs:
            vec[idx] += coeff

    # Define X and Y for each M_k
    # X_k is 2x2, Y_k is 2x2.
    # Each cell is a list of (block, coeff)
    
    macros = []
    
    # M1
    X1 = [
        [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], # Wait, X is 2x2 matrix of values.
        # A00 is 2x2. A11 is 2x2.
        # X1[0,0] = A00[0,0] + A11[0,0]
        # This means the "term" at X1[0,0] is A00[0,0] + A11[0,0].
        # My "add_block" adds the whole block.
        # But Strassen operates on the scalar entries of the 2x2 matrices X and Y.
        # So X[0,0] is a scalar (linear combo of A entries).
        # I need to represent X[0,0] as a vector of coeffs for A.
        
        # Let's re-structure.
        # For each M_k, X_k is a 2x2 matrix.
        # Each entry X_k[i,j] is a linear combination of A's entries.
        # We can represent this as a dict or list of (index, coeff).
        
        # Let's build the vectors for X_k[i,j] and Y_k[i,j].
    ]
    
    # Let's just hardcode the 49 terms generation.
    
    # Strassen formulas for Z = X Y
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    
    # P1: (x00 + x11) * (y00 + y11)
    # P2: (x10 + x11) * y01
    # P3: x00 * (y01 - y11)
    # P4: x11 * (y10 - y00)
    # P5: (x00 + x01) * y11
    # P6: (x10 - x00) * (y00 + y10)
    # P7: (x01 - x11) * (y11 + y01)
    
    # Contributions to Z[i,j]
    # Z00: P1(1), P4(1), P5(-1), P7(1)
    # Z01: P3(1), P5(1)
    # Z10: P2(1), P4(1)
    # Z11: P1(1), P2(-1), P3(1), P6(1)
    
    # Map M_k to C blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # C blocks are 2x2 matrices.
    # C00 is C[0:2, 0:2] -> flat 0,1,4,5
    # C01 is C[0:2, 2:4] -> flat 2,3,6,7
    # C10 is C[2:4, 0:2] -> flat 8,9,12,13
    # C11 is C[2:4, 2:4] -> flat 10,11,14,15
    
    # For each M_k, we have a 2x2 matrix Z_k.
    # Z_k[i,j] contributes to C_block[row, col][i, j].
    # The coefficient is determined by the formula above.
    
    # Let's define the 7 macros with their X and Y terms.
    # X and Y are 2x2 matrices of "A-terms" and "B-terms".
    # An "A-term" is a vector of length 16.
    
    def make_term(blocks):
        # blocks is list of (block, coeff)
        vec = [0] * 16
        for block, coeff in blocks:
            idxs = get_block_indices(block)
            for idx in idxs:
                vec[idx] += coeff
        return vec

    # Define X and Y for each M
    # M1: X = A00 + A11, Y = B00 + B11
    # X is 2x2 matrix. X[i,j] = A00[i,j] + A11[i,j]
    # So X[0,0] = A00[0,0] + A11[0,0]
    # This is not a single block, but specific entries.
    # Wait, A00 is a 2x2 block.
    # A00[0,0] is A[0,0]. A11[0,0] is A[2,2].
    # So X1[0,0] corresponds to A[0,0] + A[2,2].
    
    # Let's define the 16 entries of X and Y for each M.
    
    # M1
    X1 = [
        [make_term([((0,0), 1), ((1,1), 1)]), make_term([((0,0), 1), ((1,1), 1)])], # Wrong, make_term takes blocks, not entries.
    ]
    
    # Correction: make_term should take specific entries or I need a helper for entries.
    # Let's use a helper to get the vector for a single entry A[r,c]
    def entry_vec_A(r, c):
        vec = [0] * 16
        vec[r * 4 + c] = 1
        return vec
        
    def entry_vec_B(r, c):
        vec = [0] * 16
        vec[r * 4 + c] = 1
        return vec

    # Now define X and Y for M1..M7
    # X_k is 2x2 matrix of vectors (length 16)
    
    # M1: X = A00 + A11
    # X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2]
    # X[1,1] = A[1,1] + A[3,3]
    
    X1 = [
        [ [1 if i in [0, 10] else 0 for i in range(16)], [1 if i in [1, 11] else 0 for i in range(16)] ],
        [ [1 if i in [4, 14] else 0 for i in range(16)], [1 if i in [5, 15] else 0 for i in range(16)] ]
    ]
    # This is getting messy. Let's just build the 49 terms explicitly.
    
    # I will write a function to generate the U, V, W vectors for a specific Strassen product of two 2x2 matrices X and Y.
    
    def generate_strassen_terms(X, Y, M_idx, U, V, W, rank_offset):
        # X, Y are 2x2 lists of lists of vectors (length 16)
        # M_idx is 0..6 for M1..M7
        
        # P1 = (X[0][0] + X[1][1]) * (Y[0][0] + Y[1][1])
        u1 = [x + y for x, y in zip(X[0][0], X[1][1])]
        v1 = [x + y for x, y in zip(Y[0][0], Y[1][1])]
        
        # P2 = (X[1][0] + X[1][1]) * Y[0][1]
        u2 = [x + y for x, y in zip(X[1][0], X[1][1])]
        v2 = Y[0][1]
        
        # P3 = X[0][0] * (Y[0][1] - Y[1][1])
        u3 = X[0][0]
        v3 = [x - y for x, y in zip(Y[0][1], Y[1][1])]
        
        # P4 = X[1][1] * (Y[1][0] - Y[0][0])
        u4 = X[1][1]
        v4 = [x - y for x, y in zip(Y[1][0], Y[0][0])]
        
        # P5 = (X[0][0] + X[0][1]) * Y[1][1]
        u5 = [x + y for x, y in zip(X[0][0], X[0][1])]
        v5 = Y[1][1]
        
        # P6 = (X[1][0] - X[0][0]) * (Y[0][0] + Y[1][0])
        u6 = [x - y for x, y in zip(X[1][0], X[0][0])]
        v6 = [x + y for x, y in zip(Y[0][0], Y[1][0])]
        
        # P7 = (X[0][1] - X[1][1]) * (Y[1][1] + Y[0][1])
        u7 = [x - y for x, y in zip(X[0][1], X[1][1])]
        v7 = [x + y for x, y in zip(Y[1][1], Y[0][1])]
        
        terms = [
            (u1, v1, 1, 1, 1, 1), # P1: Z00(1), Z11(1) -- Wait, Z00 coeff 1, Z11 coeff 1.
            (u2, v2, 1, 1, 0, 0), # P2: Z10(1), Z11(-1)
            (u3, v3, 0, 0, 1, 1), # P3: Z01(1), Z11(1)
            (u4, v4, 1, 0, 1, 0), # P4: Z00(1), Z10(1)
            (u5, v5, 0, 0, 1, -1), # P5: Z01(1), Z00(-1)
            (u6, v6, 0, 0, 0, 1), # P6: Z11(1)
            (u7, v7, 1, 0, 0, 1)  # P7: Z00(1), Z11(1) -- Wait, Z00 coeff 1?
        ]
        
        # Let's re-verify Strassen contributions
        # Z00 = P1 + P4 - P5 + P7
        # Z01 = P3 + P5
        # Z10 = P2 + P4
        # Z11 = P1 - P2 + P3 + P6
        
        # My tuple format: (u, v, coeff_Z00, coeff_Z10, coeff_Z01, coeff_Z11)
        # P1: Z00(1), Z11(1) -> (1, 0, 0, 1)
        # P2: Z10(1), Z11(-1) -> (0, 1, 0, -1)
        # P3: Z01(1), Z11(1) -> (0, 0, 1, 1)
        # P4: Z00(1), Z10(1) -> (1, 1, 0, 0)
        # P5: Z01(1), Z00(-1) -> (-1, 0, 1, 0)
        # P6: Z11(1) -> (0, 0, 0, 1)
        # P7: Z00(1), Z11(1) -> (1, 0, 0, 1)
        
        coeffs = [
            (1, 0, 0, 1),
            (0, 1, 0, -1),
            (0, 0, 1, 1),
            (1, 1, 0, 0),
            (-1, 0, 1, 0),
            (0, 0, 0, 1),
            (1, 0, 0, 1)
        ]
        
        uv_list = [
            (u1, v1), (u2, v2), (u3, v3), (u4, v4), (u5, v5), (u6, v6), (u7, v7)
        ]
        
        for j in range(7):
            u, v = uv_list[j]
            c00, c10, c01, c11 = coeffs[j]
            
            r = rank_offset + j
            
            # Fill U and V
            for i in range(16):
                U[i][r] = u[i]
                V[i][r] = v[i]
                
            # Fill W
            # W index for C[i,j] is j*4 + i (from seed: W[k*n+i])
            # C00 is block (0,0) -> C[0,0], C[0,1], C[1,0], C[1,1]
            # Flat indices: 0, 1, 4, 5
            # C01 is block (0,1) -> C[0,2], C[0,3], C[1,2], C[1,3]
            # Flat indices: 2, 3, 6, 7
            # C10 is block (1,0) -> C[2,0], C[2,1], C[3,0], C[3,1]
            # Flat indices: 8, 9, 12, 13
            # C11 is block (1,1) -> C[2,2], C[2,3], C[3,2], C[3,3]
            # Flat indices: 10, 11, 14, 15
            
            # Z00 contributes to C00
            if c00 != 0:
                for i, j in [(0,0), (0,1), (1,0), (1,1)]:
                    # Z[i,j] corresponds to C[0+i, 0+j]
                    # W index = (0+j)*4 + (0+i)
                    w_idx = (0 + j) * 4 + (0 + i)
                    W[w_idx][r] += c00
            if c10 != 0:
                for i, j in [(0,0), (0,1), (1,0), (1,1)]:
                    # Z[i,j] corresponds to C[1+i, 0+j]
                    # W index = (0+j)*4 + (1+i)
                    w_idx = (0 + j) * 4 + (1 + i)
                    W[w_idx][r] += c10
            if c01 != 0:
                for i, j in [(0,0), (0,1), (1,0), (1,1)]:
                    # Z[i,j] corresponds to C[0+i, 1+j]
                    # W index = (1+j)*4 + (0+i)
                    w_idx = (1 + j) * 4 + (0 + i)
                    W[w_idx][r] += c01
            if c11 != 0:
                for i, j in [(0,0), (0,1), (1,0), (1,1)]:
                    # Z[i,j] corresponds to C[1+i, 1+j]
                    # W index = (1+j)*4 + (1+i)
                    w_idx = (1 + j) * 4 + (1 + i)
                    W[w_idx][r] += c11

    # Now define X and Y for M1..M7
    # X_k[i,j] is a vector of length 16
    
    def vec_A(r, c):
        v = [0]*16
        v[r*4+c] = 1
        return v
        
    def vec_B(r, c):
        v = [0]*16
        v[r*4+c] = 1
        return v
        
    def add_vec(v1, v2, sign=1):
        return [a + sign*b for a, b in zip(v1, v2)]
        
    def sub_vec(v1, v2):
        return [a - b for a, b in zip(v1, v2)]
        
    # M1: X = A00 + A11, Y = B00 + B11
    X1 = [
        [add_vec(vec_A(0,0), vec_A(2,2)), add_vec(vec_A(0,1), vec_A(2,3))],
        [add_vec(vec_A(1,0), vec_A(3,2)), add_vec(vec_A(1,1), vec_A(3,3))]
    ]
    Y1 = [
        [add_vec(vec_B(0,0), vec_B(2,2)), add_vec(vec_B(0,1), vec_B(2,3))],
        [add_vec(vec_B(1,0), vec_B(3,2)), add_vec(vec_B(1,1), vec_B(3,3))]
    ]
    
    # M2: X = A10 + A11, Y = B01
    X2 = [
        [add_vec(vec_A(2,0), vec_A(2,2)), add_vec(vec_A(2,1), vec_A(2,3))], # Wait, A10 is A[2:4, 0:2], A11 is A[2:4, 2:4]
        # A10[0,0] = A[2,0]. A11[0,0] = A[2,2].
        # A10[0,1] = A[2,1]. A11[0,1] = A[2,3].
        # A10[1,0] = A[3,0]. A11[1,0] = A[3,2].
        # A10[1,1] = A[3,1]. A11[1,1] = A[3,3].
        [add_vec(vec_A(3,0), vec_A(3,2)), add_vec(vec_A(3,1), vec_A(3,3))]
    ]
    Y2 = [
        [vec_B(0,2), vec_B(0,3)], # B01 is B[0:2, 2:4]
        [vec_B(1,2), vec_B(1,3)]
    ]
    
    # M3: X = A00, Y = B01 - B11
    X3 = [
        [vec_A(0,0), vec_A(0,1)],
        [vec_A(1,0), vec_A(1,1)]
    ]
    Y3 = [
        [sub_vec(vec_B(0,2), vec_B(2,2)), sub_vec(vec_B(0,3), vec_B(2,3))],
        [sub_vec(vec_B(1,2), vec_B(3,2)), sub_vec(vec_B(1,3), vec_B(3,3))]
    ]
    
    # M4: X = A11, Y = B10 - B00
    X4 = [
        [vec_A(2,2), vec_A(2,3)],
        [vec_A(3,2), vec_A(3,3)]
    ]
    Y4 = [
        [sub_vec(vec_B(2,0), vec_B(0,0)), sub_vec(vec_B(2,1), vec_B(0,1))],
        [sub_vec(vec_B(3,0), vec_B(1,0)), sub_vec(vec_B(3,1), vec_B(1,1))]
    ]
    
    # M5: X = A00 + A01, Y = B11
    X5 = [
        [add_vec(vec_A(0,0), vec_A(0,2)), add_vec(vec_A(0,1), vec_A(0,3))],
        [add_vec(vec_A(1,0), vec_A(1,2)), add_vec(vec_A(1,1), vec_A(1,3))]
    ]
    Y5 = [
        [vec_B(2,2), vec_B(2,3)],
        [vec_B(3,2), vec_B(3,3)]
    ]
    
    # M6: X = A10 - A00, Y = B00 + B10
    X6 = [
        [sub_vec(vec_A(2,0), vec_A(0,0)), sub_vec(vec_A(2,1), vec_A(0,1))],
        [sub_vec(vec_A(3,0), vec_A(1,0)), sub_vec(vec_A(3,1), vec_A(1,1))]
    ]
    Y6 = [
        [add_vec(vec_B(0,0), vec_B(2,0)), add_vec(vec_B(0,1), vec_B(2,1))],
        [add_vec(vec_B(1,0), vec_B(3,0)), add_vec(vec_B(1,1), vec_B(3,1))]
    ]
    
    # M7: X = A01 - A11, Y = B11 + B01
    X7 = [
        [sub_vec(vec_A(0,2), vec_A(2,2)), sub_vec(vec_A(0,3), vec_A(2,3))],
        [sub_vec(vec_A(1,2), vec_A(3,2)), sub_vec(vec_A(1,3), vec_A(3,3))]
    ]
    Y7 = [
        [add_vec(vec_B(2,2), vec_B(0,2)), add_vec(vec_B(2,3), vec_B(0,3))],
        [add_vec(vec_B(3,2), vec_B(1,2)), add_vec(vec_B(3,3), vec_B(1,3))]
    ]
    
    generate_strassen_terms(X1, Y1, 0, U, V, W, 0)
    generate_strassen_terms(X2, Y2, 1, U, V, W, 7)
    generate_strassen_terms(X3, Y3, 2, U, V, W, 14)
    generate_strassen_terms(X4, Y4, 3, U, V, W, 21)
    generate_strassen_terms(X5, Y5, 4, U, V, W, 28)
    generate_strassen_terms(X6, Y6, 5, U, V, W, 35)
    generate_strassen_terms(X7, Y7, 6, U, V, W, 42)
    
    return {"U": U, "V": V, "W": W}
