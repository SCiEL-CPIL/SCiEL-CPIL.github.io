import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook baseline (rank 48)
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W, "rank": rank}
    
    best_result = make_schoolbook()
    
    # Strassen 2x2 blocking approach (target rank 42)
    # A is 2x4, B is 4x6
    # Partition A into 2x2 blocks of size 1x2:
    #   A11 = A[0:1, 0:2], A12 = A[0:1, 2:4]
    #   A21 = A[1:2, 0:2], A22 = A[1:2, 2:4]
    # Partition B into 2x2 blocks of size 2x3:
    #   B11 = B[0:2, 0:3], B12 = B[0:2, 3:6]
    #   B21 = B[2:4, 0:3], B22 = B[2:4, 3:6]
    # C is 2x6, partition into 2x2 blocks of size 1x3:
    #   C11 = C[0:1, 0:3], C12 = C[0:1, 3:6]
    #   C21 = C[1:2, 0:3], C22 = C[1:2, 3:6]
    
    # Strassen 7 steps:
    # M1 = (A11 + A22) * (B11 + B22) -> C11 += M1, C12 -= M1, C21 += M1, C22 -= M1
    # M2 = A22 * (B12 - B22)         -> C12 += M2, C22 += M2
    # M3 = (A11 - A21) * B21         -> C11 += M3, C21 += M3
    # M4 = A11 * (B12 - B22)         -> C12 += M4
    # M5 = (A11 + A12) * B22         -> C22 += M5
    # M6 = (A21 - A11) * (B11 + B12) -> C21 += M6
    # M7 = (A12 - A22) * (B21 + B22) -> C11 += M7, C22 += M7
    
    # Wait, let me re-derive Strassen correctly for 2x2 blocks.
    # Standard Strassen for 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each multiplication is of a 1x2 matrix by a 2x3 matrix, which has rank 6.
    # Total rank = 7 * 6 = 42.
    
    # Decomposition of 1x2 * 2x3 multiplication (rank 6):
    # X is 1x2: [x0, x1]
    # Y is 2x3: [[y00, y01, y02], [y10, y11, y12]]
    # Z = X * Y is 1x3: [z0, z1, z2]
    # z0 = x0*y00 + x1*y10
    # z1 = x0*y01 + x1*y11
    # z2 = x0*y02 + x1*y12
    
    # Rank-6 decomposition:
    # T1: u=[1,0], v=[1,0], w=[1,0,0]  -> x0*y00 -> z0
    # T2: u=[1,0], v=[0,1], w=[0,1,0]  -> x0*y01 -> z1
    # T3: u=[1,0], v=[0,0], w=[0,0,1]  -> x0*y02 -> z2
    # T4: u=[0,1], v=[1,0], w=[0,0,0]  -> x1*y10 -> z0
    # T5: u=[0,1], v=[0,1], w=[0,0,0]  -> x1*y11 -> z1
    # T6: u=[0,1], v=[0,0], w=[0,0,0]  -> x1*y12 -> z2
    
    # For each Strassen step, we need to express the 1x2 and 2x3 inputs as
    # linear combinations of the global A and B entries.
    
    # A indices: A[i,j] for i in {0,1}, j in {0,1,2,3}
    #   A11 = [[A[0,0], A[0,1]]]
    #   A12 = [[A[0,2], A[0,3]]]
    #   A21 = [[A[1,0], A[1,1]]]
    #   A22 = [[A[1,2], A[1,3]]]
    
    # B indices: B[j,k] for j in {0,1,2,3}, k in {0,1,2,3,4,5}
    #   B11 = [[B[0,0], B[0,1], B[0,2]], [B[1,0], B[1,1], B[1,2]]]
    #   B12 = [[B[0,3], B[0,4], B[0,5]], [B[1,3], B[1,4], B[1,5]]]
    #   B21 = [[B[2,0], B[2,1], B[2,2]], [B[3,0], B[3,1], B[3,2]]]
    #   B22 = [[B[2,3], B[2,4], B[2,5]], [B[3,3], B[3,4], B[3,5]]]
    
    # C indices: C[i,k] for i in {0,1}, k in {0,1,2,3,4,5}
    #   C11 = [[C[0,0], C[0,1], C[0,2]]]
    #   C12 = [[C[0,3], C[0,4], C[0,5]]]
    #   C21 = [[C[1,0], C[1,1], C[1,2]]]
    #   C22 = [[C[1,3], C[1,4], C[1,5]]]
    
    # Define the 7 Strassen steps as (X_coeffs, Y_coeffs, C_contributions)
    # X_coeffs: list of 2 tuples (a_idx, coeff) where a_idx is (i,j) in A
    # Y_coeffs: list of 6 tuples (b_idx, coeff) where b_idx is (j,k) in B
    # C_contributions: list of (c_idx, coeff) where c_idx is (i,k) in C
    
    # Helper to get A index from block
    def a_idx(block, r, c):
        # block: 11, 12, 21, 22
        # r: 0 or 1 (row within 1x2 block, always 0)
        # c: 0 or 1 (col within 1x2 block)
        if block == 11:
            return (0, c)
        elif block == 12:
            return (0, 2 + c)
        elif block == 21:
            return (1, c)
        elif block == 22:
            return (1, 2 + c)
    
    def b_idx(block, r, c):
        # block: 11, 12, 21, 22
        # r: 0 or 1 (row within 2x3 block)
        # c: 0, 1, or 2 (col within 2x3 block)
        if block == 11:
            return (r, c)
        elif block == 12:
            return (r, 3 + c)
        elif block == 21:
            return (2 + r, c)
        elif block == 22:
            return (2 + r, 3 + c)
    
    def c_idx(block, c):
        # block: 11, 12, 21, 22
        # c: 0, 1, or 2 (col within 1x3 block)
        if block == 11:
            return (0, c)
        elif block == 12:
            return (0, 3 + c)
        elif block == 21:
            return (1, c)
        elif block == 22:
            return (1, 3 + c)
    
    # Strassen steps:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For each step, define X as a linear combination of A blocks,
    # Y as a linear combination of B blocks, and C as contributions to C blocks.
    
    # X is 1x2. Each block is 1x2. So X = sum of coeff * block.
    # Represent X as dict: {(i,j): coeff}
    
    def make_x(blocks_coeffs):
        """blocks_coeffs: list of (block, coeff)"""
        x = {}
        for block, coeff in blocks_coeffs:
            for c in range(2):
                idx = a_idx(block, 0, c)
                x[idx] = x.get(idx, 0) + coeff
        return x
    
    def make_y(blocks_coeffs):
        """blocks_coeffs: list of (block, coeff)"""
        y = {}
        for block, coeff in blocks_coeffs:
            for r in range(2):
                for c in range(3):
                    idx = b_idx(block, r, c)
                    y[idx] = y.get(idx, 0) + coeff
        return y
    
    def make_c(blocks_coeffs):
        """blocks_coeffs: list of (block, coeff)"""
        c = {}
        for block, coeff in blocks_coeffs:
            for c_col in range(3):
                idx = c_idx(block, c_col)
                c[idx] = c.get(idx, 0) + coeff
        return c
    
    # Define the 7 Strassen steps
    steps = [
        # M1 = (A11 + A22)(B11 + B22) -> C11 += M1, C12 -= M1, C21 += M1, C22 -= M1
        # Wait, standard Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let me re-verify. Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Step 1: M1 = (A11 + A22)(B11 + B22)
        # X = A11 + A22, Y = B11 + B22
        # C: C11 += M1, C22 += M1
        {
            "x": [(11, 1), (22, 1)],
            "y": [(11, 1), (22, 1)],
            "c": [(11, 1), (22, 1)]
        },
        # Step 2: M2 = (A21 + A22)B11
        # X = A21 + A22, Y = B11
        # C: C21 += M2, C22 -= M2
        {
            "x": [(21, 1), (22, 1)],
            "y": [(11, 1)],
            "c": [(21, 1), (22, -1)]
        },
        # Step 3: M3 = A11(B12 - B22)
        # X = A11, Y = B12 - B22
        # C: C12 += M3, C22 += M3
        {
            "x": [(11, 1)],
            "y": [(12, 1), (22, -1)],
            "c": [(12, 1), (22, 1)]
        },
        # Step 4: M4 = A22(B21 - B11)
        # X = A22, Y = B21 - B11
        # C: C11 += M4, C21 += M4
        {
            "x": [(22, 1)],
            "y": [(21, 1), (11, -1)],
            "c": [(11, 1), (21, 1)]
        },
        # Step 5: M5 = (A11 + A12)B22
        # X = A11 + A12, Y = B22
        # C: C12 += M5, C22 += M5
        # Wait, C11 = M1 + M4 - M5 + M7, so C11 -= M5
        # C12 = M3 + M5, so C12 += M5
        {
            "x": [(11, 1), (12, 1)],
            "y": [(22, 1)],
            "c": [(11, -1), (12, 1)]
        },
        # Step 6: M6 = (A21 - A11)(B11 + B12)
        # X = A21 - A11, Y = B11 + B12
        # C: C22 += M6
        {
            "x": [(21, 1), (11, -1)],
            "y": [(11, 1), (12, 1)],
            "c": [(22, 1)]
        },
        # Step 7: M7 = (A12 - A22)(B21 + B22)
        # X = A12 - A22, Y = B21 + B22
        # C: C11 += M7, C22 += M7
        {
            "x": [(12, 1), (22, -1)],
            "y": [(21, 1), (22, 1)],
            "c": [(11, 1), (22, 1)]
        },
    ]
    
    # Rank-6 decomposition of 1x2 * 2x3
    # T1: u=[1,0], v=[1,0], w=[1,0,0]
    # T2: u=[1,0], v=[0,1], w=[0,1,0]
    # T3: u=[1,0], v=[0,0], w=[0,0,1]
    # T4: u=[0,1], v=[1,0], w=[0,0,0]
    # T5: u=[0,1], v=[0,1], w=[0,0,0]
    # T6: u=[0,1], v=[0,0], w=[0,0,0]
    
    # u is 1x2: [u0, u1] -> multiplies the two columns of X
    # v is 2x1: [v0; v1] -> multiplies the two rows of Y
    # w is 1x3: [w0, w1, w2] -> contributes to the three columns of Z
    
    decomp_123 = [
        ([1, 0], [1, 0], [1, 0, 0]),
        ([1, 0], [0, 1], [0, 1, 0]),
        ([1, 0], [0, 0], [0, 0, 1]),
        ([0, 1], [1, 0], [0, 0, 0]),
        ([0, 1], [0, 1], [0, 0, 0]),
        ([0, 1], [0, 0], [0, 0, 0]),
    ]
    
    # Build the decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for step in steps:
        x_coeffs = make_x(step["x"])
        y_coeffs = make_y(step["y"])
        c_coeffs = make_c(step["c"])
        
        for u, v, w in decomp_123:
            # u = [u0, u1]: coefficients for the two columns of X
            # v = [v0, v1]: coefficients for the two rows of Y
            # w = [w0, w1, w2]: coefficients for the three columns of Z
            
            # X is 1x2. The two "components" are the two columns.
            # Column 0 of X: sum over (i,j) where j is the col index within the 1x2 block
            # But our x_coeffs is a dict from (i,j) global A index to coeff.
            # The two columns of X correspond to the two columns within each 1x2 block.
            # For block 11: col 0 is A[0,0], col 1 is A[0,1]
            # For block 12: col 0 is A[0,2], col 1 is A[0,3]
            # etc.
            
            # So u[0] multiplies all entries in column 0 of the 1x2 blocks,
            # u[1] multiplies all entries in column 1 of the 1x2 blocks.
            
            # Similarly for Y: v[0] multiplies row 0 of the 2x3 blocks,
            # v[1] multiplies row 1 of the 2x3 blocks.
            
            # And w[0], w[1], w[2] multiply the three columns of the 1x3 output blocks.
            
            # U contribution: for each (i,j) in x_coeffs,
            # if j is col 0 of its block (j % 2 == 0 for blocks 11,21; j % 2 == 0 for 12,22? No.)
            # Actually, within a 1x2 block, col 0 is the first col, col 1 is the second.
            # For block 11: A[0,0] is col 0, A[0,1] is col 1.
            # For block 12: A[0,2] is col 0, A[0,3] is col 1.
            # So col index within block = j % 2 for blocks in row 0 (11,12), and j % 2 for blocks in row 1 (21,22).
            # Wait, j is the global column index (0-3).
            # Block 11: j=0,1. Col within block: j%2.
            # Block 12: j=2,3. Col within block: (j-2)%2 = j%2.
            # So col within block = j % 2 always.
            
            # Row within block for A: always 0 (since blocks are 1x2).
            
            # For U: U[i*m+j, col] = x_coeffs[(i,j)] * u[col_within_block]
            for (i, j), coeff in x_coeffs.items():
                col_in_block = j % 2
                U[i * m + j][col] = coeff * u[col_in_block]
            
            # For V: V[j*p+k, col] = y_coeffs[(j,k)] * v[row_within_block]
            # Row within block for B: for block 11,21: row = j (0 or 1). For block 12,22: row = j-2 (0 or 1).
            # So row within block = j % 2 for j in {0,1} (blocks 11,12) and (j-2) % 2 for j in {2,3} (blocks 21,22).
            # Actually: block 11: j=0,1. Row within block: j.
            # Block 12: j=0,1. Row within block: j.
            # Block 21: j=2,3. Row within block: j-2.
            # Block 22: j=2,3. Row within block: j-2.
            # So row within block = j % 2 for j in {0,1}, and (j-2) % 2 for j in {2,3}.
            # Which is just j % 2 in all cases? j=0:0, j=1:1, j=2:0, j=3:1. Yes, j%2.
            
            for (j, k), coeff in y_coeffs.items():
                row_in_block = j % 2
                V[j * p + k][col] = coeff * v[row_in_block]
            
            # For W: W[k*n+i, col] = c_coeffs[(i,k)] * w[col_within_block]
            # Col within block for C: for block 11,21: col = k (0,1,2). For block 12,22: col = k-3 (0,1,2).
            # So col within block = k % 3 for k in {0,1,2}, and (k-3) % 3 for k in {3,4,5}.
            # Which is just k % 3 in all cases? k=0:0, k=1:1, k=2:2, k=3:0, k=4:1, k=5:2. Yes, k%3.
            
            for (i, k), coeff in c_coeffs.items():
                col_in_block = k % 3
                W[k * n + i][col] = coeff * w[col_in_block]
            
            col += 1
    
    # Verify the decomposition
    def verify(U, V, W, rank):
        # Check that sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1 for all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W, rank):
        best_result = {"U": U, "V": V, "W": W, "rank": rank}
    else:
        # Fall back to schoolbook
        best_result = make_schoolbook()
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > 120:
        return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
