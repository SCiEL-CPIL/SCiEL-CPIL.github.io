import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify the decomposition is correct."""
    import numpy as np
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                max_err = max(max_err, abs(val - 1))
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Split B into B1 (4x4) and B2 (4x3)
    # C = A*B1 + A*B2
    # Use Strassen for A*B1 (rank 49), schoolbook for A*B2 (rank 48)
    # Total rank = 49 + 48 = 97
    
    rank = 93
    U = [[0]*rank for _ in range(n*m)]  # 16 x 93
    V = [[0]*rank for _ in range(m*p)]  # 28 x 93
    W = [[0]*rank for _ in range(p*n)]  # 28 x 93
    
    # Strassen 2x2 base tensors (rank 7)
    # For multiplying 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    
    # Strassen 4x4 (two levels of 2x2, rank 49)
    # A split into 4 blocks of 2x2: A00, A01, A10, A11
    # B1 split into 4 blocks of 2x2: B00, B01, B10, B11
    
    # Level 1: compute 7 macro products, each is 2x2 * 2x2
    # X1 = A00+A11, Y1 = B00+B11 -> M1
    # X2 = A10+A11, Y2 = B00 -> M2
    # X3 = A00, Y3 = B10-B11 -> M3
    # X4 = A11, Y4 = B01-B00 -> M4
    # X5 = A00+A01, Y5 = B11 -> M5
    # X6 = A10-A00, Y6 = B00+B10 -> M6
    # X7 = A01-A11, Y7 = B01+B11 -> M7
    
    # Level 2: each M_k is product of two 2x2 matrices, use 7 base tensors
    # Base Strassen tensors for 2x2:
    # u vectors (4 entries: A00,A01,A10,A11)
    # v vectors (4 entries: B00,B01,B10,B11)
    # w vectors (4 entries: C00,C01,C10,C11)
    
    base_u = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    base_v = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 0, 1, -1],  # M3: B10-B11
        [-1, 1, 0, 0],  # M4: B01-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 0, 1, 0],   # M6: B00+B10
        [0, 1, 0, 1],   # M7: B01+B11
    ]
    base_w = [
        [1, 0, 0, 1],   # M1 -> C00+C11
        [0, 0, 1, -1],  # M2 -> C10-C11
        [0, 1, 0, 1],   # M3 -> C01+C11
        [1, 0, 1, 0],   # M4 -> C00+C10
        [-1, 1, 0, 0],  # M5 -> -C00+C01
        [0, 0, 0, 1],   # M6 -> C11
        [1, 0, 0, 0],   # M7 -> C00
    ]
    
    # Actually let me be more careful. The base Strassen for 2x2*2x2->2x2:
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # So for each M_k, its contribution to each C_ij:
    # M1: C00+=1, C11+=1
    # M2: C10+=1, C11-=1
    # M3: C01+=1, C11+=1
    # M4: C00+=1, C10+=1
    # M5: C00-=1, C01+=1
    # M6: C11+=1
    # M7: C00+=1
    
    # For the 4x4 Strassen, we have 7 macro products M_k, each computed via 7 base tensors.
    # Each macro product M_k = X_k * Y_k where X_k and Y_k are 2x2 matrices.
    # The result C has 4 blocks of 2x2: C00, C01, C10, C11.
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M_k is a 2x2 matrix. So M_k fills a 2x2 block in each of C00, C01, C10, C11.
    
    # Let's define the mapping more carefully.
    # A is 4x4, split into 2x2 blocks:
    # A = [A00 A01; A10 A11] where each is 2x2
    # B1 is 4x4, split into 2x2 blocks:
    # B1 = [B00 B01; B10 B11]
    # C1 = A*B1 is 4x4, split into 2x2 blocks:
    # C1 = [C00 C01; C10 C11]
    
    # For macro product M_k = X_k * Y_k:
    # X_k is a 2x2 matrix formed from A blocks
    # Y_k is a 2x2 matrix formed from B1 blocks
    
    # The 2x2 result M_k contributes to C blocks with coefficients.
    
    # Let me define for each macro product k (1-7):
    # - Which A blocks are involved and with what signs
    # - Which B1 blocks are involved and with what signs
    # - Which C blocks receive the result and with what coefficients
    
    macro_A = {
        1: [(0,0,1), (1,1,1)],   # A00+A11
        2: [(1,0,1), (1,1,1)],   # A10+A11
        3: [(0,0,1)],            # A00
        4: [(1,1,1)],            # A11
        5: [(0,0,1), (0,1,1)],   # A00+A01
        6: [(1,0,1), (0,0,-1)],  # A10-A00
        7: [(0,1,1), (1,1,-1)],  # A01-A11
    }
    macro_B = {
        1: [(0,0,1), (1,1,1)],   # B00+B11
        2: [(0,0,1)],            # B00
        3: [(1,0,1), (1,1,-1)],  # B10-B11
        4: [(0,1,1), (0,0,-1)],  # B01-B00
        5: [(1,1,1)],            # B11
        6: [(0,0,1), (1,0,1)],   # B00+B10
        7: [(0,1,1), (1,1,1)],   # B01+B11
    }
    macro_C = {
        1: [(0,0,1), (1,1,1)],   # C00+=M1, C11+=M1
        2: [(1,0,1), (1,1,-1)],  # C10+=M2, C11-=M2
        3: [(0,1,1), (1,1,1)],   # C01+=M3, C11+=M3
        4: [(0,0,1), (1,0,1)],   # C00+=M4, C10+=M4
        5: [(0,0,-1), (0,1,1)],  # C00-=M5, C01+=M5
        6: [(1,1,1)],            # C11+=M6
        7: [(0,0,1)],            # C00+=M7
    }
    
    # For each macro product k, we use 7 base tensors.
    # Base tensor t has u_t (4 entries for 2x2 A-side), v_t (4 entries for 2x2 B-side), w_t (4 entries for 2x2 C-side).
    
    # The 2x2 A-side entries map to specific positions in the 4x4 A matrix.
    # For a 2x2 block at position (bi, bj) in the 4x4 matrix, the 2x2 entries are:
    # (0,0) -> row bi*2, col bj*2
    # (0,1) -> row bi*2, col bj*2+1
    # (1,0) -> row bi*2+1, col bj*2
    # (1,1) -> row bi*2+1, col bj*2+1
    
    # U index: i*m+j where i is row in A (0-3), j is col in A (0-3)
    # V index: j*p+k where j is row in B (0-3), k is col in B (0-6)
    # W index: k*n+i where k is col in B (0-6), i is row in A (0-3)
    
    # For B1 (cols 0-3), V index is j*7+k for k in 0-3
    # For B2 (cols 4-6), V index is j*7+k for k in 4-6
    
    # For W, k is col in B (0-6), i is row in A (0-3)
    # W index: k*4+i
    
    c = 0
    for k in range(1, 8):  # macro products 1-7
        for t in range(7):  # base tensors 0-6
            u_t = base_u[t]
            v_t = base_v[t]
            w_t = base_w[t]
            
            # For each A block involved in X_k
            # For each B block involved in Y_k
            # For each C block that receives the result
            
            # Actually, the way Strassen works:
            # M_k = X_k * Y_k where X_k and Y_k are 2x2 matrices
            # We decompose this 2x2*2x2 product using 7 base tensors
            # Each base tensor t contributes u_t[i]*v_t[j]*w_t[ell] to the 2x2 result
            
            # The 2x2 result M_k then contributes to C blocks with coefficients from macro_C[k]
            
            # So for base tensor t:
            # u_t has 4 entries corresponding to the 2x2 A-side of X_k
            # v_t has 4 entries corresponding to the 2x2 B-side of Y_k
            # w_t has 4 entries corresponding to the 2x2 C-side of M_k
            
            # The 2x2 A-side of X_k maps to specific positions in the 4x4 A matrix
            # The 2x2 B-side of Y_k maps to specific positions in the 4x4 B1 matrix
            # The 2x2 C-side of M_k maps to specific positions in each 2x2 block of C1
            
            # Let's figure out the mapping.
            # X_k is formed from A blocks. The 2x2 entries of X_k correspond to:
            # X_k[0,0], X_k[0,1], X_k[1,0], X_k[1,1]
            # These are linear combinations of A block entries.
            
            # Actually, I think I'm overcomplicating this. Let me use a different approach.
            
            # The full 4x4 Strassen decomposition can be viewed as follows:
            # There are 7 macro products, each of which is a 2x2*2x2 product.
            # Each 2x2*2x2 product has rank 7 (Strassen).
            # So total rank for the 4x4 part is 7*7 = 49.
            
            # For each macro product k and base tensor t:
            # We need to determine:
            # - Which entries of U (4x4 A matrix) are nonzero and their values
            # - Which entries of V (4x4 B1 part of B) are nonzero and their values
            # - Which entries of W (4x4 C1 part of C) are nonzero and their values
            
            # The 2x2 A-side of X_k: X_k is a 2x2 matrix where each entry is a linear combination of A block entries.
            # For example, X_1 = A00+A11, so X_1[0,0] = A00[0,0]+A11[0,0], etc.
            
            # Let me define the A-block to 4x4 position mapping:
            # A00: rows 0-1, cols 0-1
            # A01: rows 0-1, cols 2-3
            # A10: rows 2-3, cols 0-1
            # A11: rows 2-3, cols 2-3
            
            # Similarly for B1 and C1.
            
            # For macro product k, X_k is a 2x2 matrix.
            # X_k[r,c] = sum of (coeff * A_block[br,bc]) for each (br,bc,coeff) in macro_A[k]
            # where A_block is the 2x2 block at position (br,bc) in the 2x2 block grid.
            
            # The 2x2 A-side of the base tensor t has entries u_t[0], u_t[1], u_t[2], u_t[3]
            # corresponding to X_k[0,0], X_k[0,1], X_k[1,0], X_k[1,1]
            
            # So the contribution to U from base tensor t is:
            # For each (r,c) in 0-1:
            #   val = u_t[r*2+c]
            #   For each (br,bc,coeff) in macro_A[k]:
            #     A_row = br*2 + r
            #     A_col = bc*2 + c
            #     U[A_row*4 + A_col][c_idx] += val * coeff
            
            # Similarly for V and W.
            
            c_idx = c  # current column index in the rank dimension
            
            # U contributions
            for r in range(2):
                for cc in range(2):
                    val = u_t[r*2+cc]
                    if val == 0:
                        continue
                    for (br, bc, coeff) in macro_A[k]:
                        A_row = br*2 + r
                        A_col = bc*2 + cc
                        U[A_row*4 + A_col][c_idx] += val * coeff
            
            # V contributions (B1 part, cols 0-3)
            for r in range(2):
                for cc in range(2):
                    val = v_t[r*2+cc]
                    if val == 0:
                        continue
                    for (br, bc, coeff) in macro_B[k]:
                        B_row = br*2 + r
                        B_col = bc*2 + cc
                        V[B_row*7 + B_col][c_idx] += val * coeff
            
            # W contributions
            for r in range(2):
                for cc in range(2):
                    val = w_t[r*2+cc]
                    if val == 0:
                        continue
                    for (br, bc, coeff) in macro_C[k]:
                        C_row = br*2 + r
                        C_col = bc*2 + cc
                        # W index: C_col*4 + C_row (since W[k*n+i] where k is col, i is row)
                        W[C_col*4 + C_row][c_idx] += val * coeff
            
            c += 1
    
    # Add optimized A*B2 (4x4 * 4x2, rank 28)
    # B2 has cols 4-5
    # Split into Top (rows 0-1 of A) and Bottom (rows 2-3 of A)
    # Top: C_top = A_top * B2 = [A00 A01] * [B_top; B_bot] = A00*B_top + A01*B_bot
    # Bottom: C_bot = A_bot * B2 = [A10 A11] * [B_top; B_bot] = A10*B_top + A11*B_bot
    
    # Helper to add 2x2x2 Strassen terms
    def add_strassen_2x2x2(u_blocks, v_blocks, w_blocks, c_start):
        # u_blocks, v_blocks, w_blocks are lists of 4 (row, col) pairs for 2x2 blocks
        # Actually, let's just iterate over the 7 base tensors
        for t in range(7):
            u_t = base_u[t]
            v_t = base_v[t]
            w_t = base_w[t]
            c_idx = c_start + t
            
            # U contributions
            for r in range(2):
                for cc in range(2):
                    val = u_t[r*2+cc]
                    if val == 0: continue
                    # Map 2x2 local (r,cc) to global A position
                    # u_blocks is a list of (global_row, global_col) for the 2x2 block
                    # Wait, u_blocks is for the A-side.
                    # The 2x2 block is at some position.
                    # Let's pass the offset.
                    pass
            
            # This helper is getting complicated with closures. Let's inline it.
            pass
        return c_start + 7

    # Inline logic for A00 * B_top
    # A00 is at rows 0-1, cols 0-1
    # B_top is at rows 0-1, cols 4-5
    # C_top is at rows 0-1, cols 4-5 (in C matrix, which is 4x7? No, C is 4x7? No, C is 4x7? 
    # Wait, C is 4x7? No, A is 4x4, B is 4x7, C is 4x7.
    # W index is k*n+i. k is col of B (0-6), i is row of A (0-3).
    # So for C_top (rows 0-1 of A, cols 4-5 of B), W indices are:
    # k=4, i=0 -> 4*4+0 = 16
    # k=4, i=1 -> 4*4+1 = 17
    # k=5, i=0 -> 5*4+0 = 20
    # k=5, i=1 -> 5*4+1 = 21
    
    # Let's define a function to add a 2x2x2 Strassen product
    # A_block: (row_offset, col_offset)
    # B_block: (row_offset, col_offset)
    # C_block: (row_offset, col_offset) -> maps to W indices
    def add_2x2x2_strassen(a_ro, a_co, b_ro, b_co, c_ro, c_co, c_start):
        for t in range(7):
            u_t = base_u[t]
            v_t = base_v[t]
            w_t = base_w[t]
            c_idx = c_start + t
            
            # U
            for r in range(2):
                for cc in range(2):
                    val = u_t[r*2+cc]
                    if val == 0: continue
                    U[(a_ro+r)*4 + (a_co+cc)][c_idx] += val
            
            # V
            for r in range(2):
                for cc in range(2):
                    val = v_t[r*2+cc]
                    if val == 0: continue
                    V[(b_ro+r)*7 + (b_co+cc)][c_idx] += val
            
            # W
            for r in range(2):
                for cc in range(2):
                    val = w_t[r*2+cc]
                    if val == 0: continue
                    # C_block is (row_offset, col_offset) in C matrix (which is indexed by B_col, A_row)
                    # W index: B_col * 4 + A_row
                    # C_col = c_co + cc
                    # C_row = c_ro + r
                    W[(c_co+cc)*4 + (c_ro+r)][c_idx] += val
        
        return c_start + 7

    # 1. A00 * B_top -> C_top
    # A00: (0,0), B_top: (0,4), C_top: (0,4)
    c = add_2x2x2_strassen(0, 0, 0, 4, 0, 4, c)
    
    # 2. A01 * B_bot -> C_top
    # A01: (0,2), B_bot: (2,4), C_top: (0,4)
    c = add_2x2x2_strassen(0, 2, 2, 4, 0, 4, c)
    
    # 3. A10 * B_top -> C_bot
    # A10: (2,0), B_top: (0,4), C_bot: (2,4)
    c = add_2x2x2_strassen(2, 0, 0, 4, 2, 4, c)
    
    # 4. A11 * B_bot -> C_bot
    # A11: (2,2), B_bot: (2,4), C_bot: (2,4)
    c = add_2x2x2_strassen(2, 2, 2, 4, 2, 4, c)
    
    # Add schoolbook for A*B3 (4x4 * 4x1, rank 16)
    # B3 has col 6
    for i in range(4):
        for j in range(4):
            B_col = 6
            U[i*4 + j][c] = 1
            V[j*7 + B_col][c] = 1
            W[B_col*4 + i][c] = 1
            c += 1
    
    # Verify
    err = _verify(U, V, W)
    if err > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
