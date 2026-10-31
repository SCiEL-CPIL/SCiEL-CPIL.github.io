"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen algorithm.
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    _t0 = time.time()
    _DEADLINE = 120.0
    _MARGIN = 10.0
    
    rank = 49
    # Use float64 for intermediate calculations to avoid overflow issues if any, 
    # then cast to int. The coefficients are small integers.
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Indices for A (n x m), B (m x p), C (p x n)
    # A is 4x4, B is 4x4, C is 4x4.
    # Index in U: row * m + col
    # Index in V: row * p + col
    # Index in W: col * n + row (trace convention W[k*n+i, r] where k is col, i is row)
    
    # Strassen 2x2 decomposition definitions
    # Outer level: 4x4 as 2x2 blocks of 2x2
    # Inner level: 2x2 as 2x2 blocks of 1x1 (scalars)
    
    # We represent linear combinations as lists of (index, coeff)
    # A index: i * 4 + j
    # B index: i * 4 + j
    
    def a_idx(i, j):
        return i * 4 + j
        
    def b_idx(i, j):
        return i * 4 + j
        
    def w_idx(r, c):
        return c * 4 + r
        
    # Outer Strassen terms (7 terms)
    # Each term is (X_blocks, Y_blocks)
    # X_blocks is a 2x2 matrix of linear combos of A entries
    # Y_blocks is a 2x2 matrix of linear combos of B entries
    
    # Define blocks
    # A00: (0,0), A01: (0,1), A10: (1,0), A11: (1,1)
    
    def block_A(br, bc):
        return [[(a_idx(br*2+i, bc*2+j), 1) for j in range(2)] for i in range(2)]
        
    def block_B(br, bc):
        return [[(b_idx(br*2+i, bc*2+j), 1) for j in range(2)] for i in range(2)]

    def add_matrix(M1, M2):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                combo = M1[i][j] + M2[i][j]
                # Combine duplicates
                d = {}
                for idx, c in combo:
                    d[idx] = d.get(idx, 0) + c
                res[i][j] = [(idx, c) for idx, c in d.items() if c != 0]
        return res

    def sub_matrix(M1, M2):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                combo = M1[i][j] + [ (idx, -c) for idx, c in M2[i][j] ]
                d = {}
                for idx, c in combo:
                    d[idx] = d.get(idx, 0) + c
                res[i][j] = [(idx, c) for idx, c in d.items() if c != 0]
        return res

    A00 = block_A(0,0)
    A01 = block_A(0,1)
    A10 = block_A(1,0)
    A11 = block_A(1,1)
    
    B00 = block_B(0,0)
    B01 = block_B(0,1)
    B10 = block_B(1,0)
    B11 = block_B(1,1)
    
    # Outer Terms
    # M1 = (A00+A11)(B00+B11)
    X1 = add_matrix(A00, A11)
    Y1 = add_matrix(B00, B11)
    
    # M2 = (A10+A11)B00
    X2 = add_matrix(A10, A11)
    Y2 = B00
    
    # M3 = A00(B01-B11)
    X3 = A00
    Y3 = sub_matrix(B01, B11)
    
    # M4 = A11(B10-B00)
    X4 = A11
    Y4 = sub_matrix(B10, B00)
    
    # M5 = (A00+A01)B11
    X5 = add_matrix(A00, A01)
    Y5 = B11
    
    # M6 = (A10-A00)(B00+B01)
    X6 = sub_matrix(A10, A00)
    Y6 = add_matrix(B00, B01)
    
    # M7 = (A01-A11)(B10+B11)
    X7 = sub_matrix(A01, A11)
    Y7 = add_matrix(B10, B11)
    
    outer_terms = [
        (X1, Y1),
        (X2, Y2),
        (X3, Y3),
        (X4, Y4),
        (X5, Y5),
        (X6, Y6),
        (X7, Y7)
    ]
    
    # C targets for each outer term
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # c_targets[t] is a list of (block_row, block_col, coeff)
    c_targets = [
        [(0,0,1), (1,1,1)],       # M1
        [(1,0,1), (1,1,-1)],      # M2
        [(0,1,1), (1,1,1)],       # M3
        [(0,0,1), (1,0,1)],       # M4
        [(0,0,-1), (0,1,1)],      # M5
        [(1,1,1)],                 # M6
        [(0,0,1)],                 # M7
    ]
    
    col = 0
    for t in range(7):
        X = outer_terms[t][0]
        Y = outer_terms[t][1]
        
        # Inner Strassen on X and Y
        # X and Y are 2x2 matrices of linear combos
        # We need to compute 7 scalar products (which are actually 2x2 matrix products of the linear combos)
        
        # Helper to get a 2x2 matrix of linear combos for a specific Strassen term
        # Inner M1 = (X00+X11)(Y00+Y11)
        X_m1 = add_matrix(X, [[X[1][1], X[1][0]], [X[0][0], X[0][1]]]) # No, simply X00+X11 etc
        # Actually, let's just use the matrix add/sub helpers on the 2x2 structures
        
        # Inner Term 1: (X00 + X11) * (Y00 + Y11)
        X_in_1 = [[add_matrix([X[0][0], X[0][1]], [X[1][0], X[1][1]])[0][0], None], [None, None]] # This is getting messy.
        
        # Let's define a helper to extract the 2x2 matrix of combos for a specific local Strassen step
        def get_inner_term(X_mat, Y_mat, term_id):
            # term_id 0..6
            if term_id == 0: # (X00+X11)(Y00+Y11)
                X_x = [[add_matrix([X_mat[0][0], X_mat[0][1]], [X_mat[1][0], X_mat[1][1]])[0][0], None], [None, None]]
                # Simpler:
                X_combos = [[[] for _ in range(2)] for _ in range(2)]
                Y_combos = [[[] for _ in range(2)] for _ in range(2)]
                # X00+X11 is a 2x2 matrix? No, X00 is a 2x2 matrix of combos.
                # The "scalar" in the inner Strassen is the 2x2 block of A.
                # So X00 is a 2x2 matrix of (idx, coeff).
                # Inner Strassen operates on these 2x2 matrices.
                
                # Let's restart the inner loop logic clearly.
                pass

        # Correct Inner Strassen Implementation
        # X and Y are 2x2 matrices where each entry is a list of (idx, coeff)
        
        # Inner M1: (X00 + X11) * (Y00 + Y11)
        # X00, X11 are 2x2 matrices of combos.
        # We need to add them element-wise? No.
        # In Strassen, X00 is a scalar. Here, X00 is a 2x2 matrix.
        # So "X00 + X11" means the 2x2 matrix sum.
        
        def mat_add(M1, M2):
            return [[add_matrix([M1[i][j], M2[i][j]])[0][0] for j in range(2)] for i in range(2)]
            
        def mat_sub(M1, M2):
            return [[sub_matrix([M1[i][j], M2[i][j]])[0][0] for j in range(2)] for i in range(2)]
            
        # This helper is wrong. add_matrix expects 2x2 matrices of lists.
        # M1[i][j] is a list.
        
        def add_2x2(M1, M2):
            res = [[[] for _ in range(2)] for _ in range(2)]
            for i in range(2):
                for j in range(2):
                    combo = M1[i][j] + M2[i][j]
                    d = {}
                    for idx, c in combo:
                        d[idx] = d.get(idx, 0) + c
                    res[i][j] = [(idx, c) for idx, c in d.items() if c != 0]
            return res
            
        def sub_2x2(M1, M2):
            res = [[[] for _ in range(2)] for _ in range(2)]
            for i in range(2):
                for j in range(2):
                    combo = M1[i][j] + [(idx, -c) for idx, c in M2[i][j]]
                    d = {}
                    for idx, c in combo:
                        d[idx] = d.get(idx, 0) + c
                    res[i][j] = [(idx, c) for idx, c in d.items() if c != 0]
            return res

        # Inner Strassen Steps
        # 1. (X00+X11)(Y00+Y11)
        X_in = add_2x2(X, [[X[1][1], X[1][0]], [X[0][0], X[0][1]]]) # No, X00 is X[0][0]. X11 is X[1][1].
        # Wait, X is a 2x2 matrix. X[0][0] is the top-left block.
        # So X00 = X[0][0], X11 = X[1][1].
        # Sum = X[0][0] + X[1][1] (element-wise addition of the 2x2 combo matrices)
        
        X_m1 = add_2x2(X, [[X[1][1], X[1][0]], [X[0][0], X[0][1]]]) # This is not right.
        # X_m1 should be a 2x2 matrix of combos.
        # The "scalar" X00 is X[0][0].
        # The sum X00+X11 is a 2x2 matrix of combos.
        # We add X[0][0] and X[1][1] element-wise.
        
        X_sum_00_11 = add_2x2(X, [[X[1][1], X[1][0]], [X[0][0], X[0][1]]]) # Still confusing.
        
        # Let's just do it explicitly for the 7 inner terms.
        
        # Inner 1: (X00+X11)(Y00+Y11)
        X_c1 = add_2x2(X, [[X[1][1], X[1][0]], [X[0][0], X[0][1]]]) # NO.
        # X is 2x2. X[0][0] is a 2x2 matrix of combos.
        # X00 + X11 means adding the 2x2 matrix X[0][0] and X[1][1].
        X_c1 = add_2x2(X, [[X[1][1], X[1][0]], [X[0][0], X[0][1]]]) # This creates a 2x2 matrix where [0][0] is X[0][0]+X[1][1]? No.
        
        # add_2x2(M1, M2) adds M1[i][j] and M2[i][j].
        # I want to add X[0][0] and X[1][1].
        # So I need a matrix where [0][0] is X[0][0]+X[1][1], and others are 0?
        # No, the result of the inner Strassen term is a 2x2 matrix Z.
        # Z is the product of two 2x2 matrices of combos.
        # The product is computed by 7 scalar multiplications (which are 2x2 matrix products).
        
        # This is getting too complex to refactor on the fly without breaking it.
        # I will revert to the original loop logic which was working, but keep it in the function.
        
        pass

    # Fallback: Use the original working logic from the current program to ensure validity.
    # The original program was valid. I will just copy it.
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    def a_idx(block_r, block_c, local_r, local_c):
        return (block_r * 2 + local_r) * m + (block_c * 2 + local_c)
        
    def b_idx(block_r, block_c, local_r, local_c):
        return (block_r * 2 + local_r) * p + (block_c * 2 + local_c)
        
    def w_idx(block_r, block_c, local_r, local_c):
        row = block_r * 2 + local_r
        col = block_c * 2 + local_c
        return col * n + row
        
    outer_terms = []
    
    # M1
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1), (2+i, 2+j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1), (2+i, 2+j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M2
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, j, 1), (2+i, 2+j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M3
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, 2+j, 1), (2+i, 2+j, -1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M4
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, 2+j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, j, 1), (i, j, -1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M5
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1), (i, 2+j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, 2+j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M6
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, j, 1), (i, j, -1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1), (i, 2+j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M7
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, 2+j, 1), (2+i, 2+j, -1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, j, 1), (2+i, 2+j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    c_targets = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]
    
    col = 0
    for t in range(7):
        X = outer_terms[t][0]
        Y = outer_terms[t][1]
        
        def add_combos(c1, c2):
            result = {}
            for (r, c, v) in c1:
                result[(r,c)] = result.get((r,c), 0) + v
            for (r, c, v) in c2:
                result[(r,c)] = result.get((r,c), 0) + v
            return [(r, c, v) for (r,c), v in result.items() if v != 0]
            
        def sub_combos(c1, c2):
            return add_combos(c1, [(r, c, -v) for (r,c,v) in c2])
            
        # Inner 1
        X_in = add_combos(X[0][0], X[1][1])
        Y_in = add_combos(Y[0][0], Y[1][1])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c, col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c, col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0), col] += coeff
            W[w_idx(br, bc, 1, 1), col] += coeff
        col += 1
        
        # Inner 2
        X_in = add_combos(X[1][0], X[1][1])
        Y_in = Y[0][0]
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c, col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c, col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 1, 0), col] += coeff
            W[w_idx(br, bc, 1, 1), col] -= coeff
        col += 1
        
        # Inner 3
        X_in = X[0][0]
        Y_in = sub_combos(Y[0][1], Y[1][1])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c, col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c, col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 1), col] += coeff
            W[w_idx(br, bc, 1, 1), col] += coeff
        col += 1
        
        # Inner 4
        X_in = X[1][1]
        Y_in = sub_combos(Y[1][0], Y[0][0])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c, col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c, col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0), col] += coeff
            W[w_idx(br, bc, 1, 0), col] += coeff
        col += 1
        
        # Inner 5
        X_in = add_combos(X[0][0], X[0][1])
        Y_in = Y[1][1]
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c, col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c, col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0), col] -= coeff
            W[w_idx(br, bc, 0, 1), col] += coeff
        col += 1
        
        # Inner 6
        X_in = sub_combos(X[1][0], X[0][0])
        Y_in = add_combos(Y[0][0], Y[0][1])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c, col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c, col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 1, 1), col] += coeff
        col += 1
        
        # Inner 7
        X_in = sub_combos(X[0][1], X[1][1])
        Y_in = add_combos(Y[1][0], Y[1][1])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c, col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c, col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0), col] += coeff
        col += 1
        
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if time.time() > _DEADLINE - _MARGIN:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
