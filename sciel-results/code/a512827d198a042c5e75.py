import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Generate factors for 2x2 Strassen
    # u, v, w are length 4 vectors corresponding to A, B, C entries
    # Order: A00, A01, A10, A11
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # 2. Define outer Strassen block combinations for 4x4
    # We need to express X_t and Y_t (2x2 matrices) in terms of A and B blocks (2x2)
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # Helper to get coefficient vector for a block combination
    # Block index: 0=A00, 1=A01, 2=A10, 3=A11
    # Returns a vector of 16 coefficients for the 16 entries of A
    
    def get_block_coeffs(block_indices_and_coeffs):
        # block_indices_and_coeffs: list of (block_idx, coeff)
        coeffs = [0.0] * 16
        for b_idx, coeff in block_indices_and_coeffs:
            # Block b_idx covers entries:
            # b_idx 0: 0,1,4,5 (rows 0,1 cols 0,1)
            # b_idx 1: 2,3,6,7 (rows 0,1 cols 2,3)
            # b_idx 2: 8,9,12,13 (rows 2,3 cols 0,1)
            # b_idx 3: 10,11,14,15 (rows 2,3 cols 2,3)
            # Entry index in 16-vector: row*4 + col
            # Block 0: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
            # Block 1: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
            # Block 2: (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13
            # Block 3: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15
            
            if b_idx == 0:
                coeffs[0] += coeff
                coeffs[1] += coeff
                coeffs[4] += coeff
                coeffs[5] += coeff
            elif b_idx == 1:
                coeffs[2] += coeff
                coeffs[3] += coeff
                coeffs[6] += coeff
                coeffs[7] += coeff
            elif b_idx == 2:
                coeffs[8] += coeff
                coeffs[9] += coeff
                coeffs[12] += coeff
                coeffs[13] += coeff
            elif b_idx == 3:
                coeffs[10] += coeff
                coeffs[11] += coeff
                coeffs[14] += coeff
                coeffs[15] += coeff
        return coeffs

    # Outer Strassen terms M1..M7
    # X_t is 2x2 matrix of block coeffs
    # Y_t is 2x2 matrix of block coeffs
    
    outer_terms = []
    
    # M1 = (A00+A11)(B00+B11)
    X1 = [
        [get_block_coeffs([(0,1), (3,1)])], # X1[0][0] is A00+A11? No.
        # Wait. X_t is a 2x2 matrix of *blocks*.
        # In Strassen, X1 is a 2x2 matrix where the entries are the blocks themselves?
        # No. Strassen computes M1 = (A00+A11)(B00+B11).
        # Here A00+A11 is a 2x2 matrix (sum of two 2x2 blocks).
        # So X1 is the 2x2 matrix A00+A11.
        # Y1 is the 2x2 matrix B00+B11.
        
        # So X1 is a 2x2 matrix.
        # X1[0,0] is the (0,0) entry of the matrix (A00+A11).
        # This entry is A00[0,0] + A11[0,0].
        # We need to express X1[r,c] as a linear combination of A's 16 entries.
    ]
    
    # Let's redefine.
    # For each outer term t, we have a product X_t * Y_t.
    # X_t is a 2x2 matrix. Y_t is a 2x2 matrix.
    # We apply inner Strassen to X_t, Y_t.
    # Inner Strassen works on 2x2 matrices of "scalars".
    # Here "scalars" are the entries of X_t and Y_t.
    # So we need to express each entry of X_t and Y_t as a vector of 16 coeffs.
    
    # Let's build the 7 outer terms' X and Y matrices of coeff-vectors.
    
    def make_outer_term(x_blocks, y_blocks):
        # x_blocks: 2x2 list of (block_idx, coeff) lists
        # y_blocks: 2x2 list of (block_idx, coeff) lists
        X = [[None for _ in range(2)] for _ in range(2)]
        Y = [[None for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                X[r][c] = get_block_coeffs(x_blocks[r][c])
                Y[r][c] = get_block_coeffs(y_blocks[r][c])
        return X, Y

    # M1: X = A00+A11, Y = B00+B11
    # X is 2x2 matrix.
    # X[0,0] = A00[0,0] + A11[0,0]
    # This is tricky. The "block" A00 is a 2x2 matrix.
    # In Strassen, we treat the blocks as scalars.
    # But here the blocks are 2x2 matrices.
    # So X_t is a 2x2 matrix of 2x2 matrices?
    # No.
    # Standard Strassen on 4x4 using 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # A00+A11 is a 2x2 matrix.
    # B00+B11 is a 2x2 matrix.
    # So M1 is a 2x2 matrix product of two 2x2 matrices.
    # To compute this product, we use Strassen again on these 2x2 matrices.
    # The "entries" of these 2x2 matrices are the original A,B entries.
    
    # So, for M1:
    # X1 is the 2x2 matrix A00+A11.
    # X1[0,0] = A[0,0] + A[2,2]
    # X1[0,1] = A[0,1] + A[2,3]
    # X1[1,0] = A[1,0] + A[3,2]
    # X1[1,1] = A[1,1] + A[3,3]
    
    # We can generate these coefficient vectors directly.
    
    def get_entry_coeffs_A(r, c, blocks):
        # blocks: list of (block_idx, coeff)
        # block_idx 0: A00, 1: A01, 2: A10, 3: A11
        # Entry (r,c) in the 2x2 block matrix corresponds to:
        # If block is A00: A[2*0+r, 2*0+c]
        # If block is A01: A[2*0+r, 2*1+c]
        # If block is A10: A[2*1+r, 2*0+c]
        # If block is A11: A[2*1+r, 2*1+c]
        coeffs = [0.0] * 16
        for b_idx, coeff in blocks:
            br = b_idx // 2
            bc = b_idx % 2
            global_r = 2*br + r
            global_c = 2*bc + c
            idx = global_r * 4 + global_c
            coeffs[idx] += coeff
        return coeffs

    def get_entry_coeffs_B(r, c, blocks):
        coeffs = [0.0] * 16
        for b_idx, coeff in blocks:
            br = b_idx // 2
            bc = b_idx % 2
            global_r = 2*br + r
            global_c = 2*bc + c
            idx = global_r * 4 + global_c
            coeffs[idx] += coeff
        return coeffs

    outer_X = []
    outer_Y = []
    
    # M1
    X1 = [[get_entry_coeffs_A(r, c, [(0,1), (3,1)]) for c in range(2)] for r in range(2)]
    Y1 = [[get_entry_coeffs_B(r, c, [(0,1), (3,1)]) for c in range(2)] for r in range(2)]
    outer_X.append(X1); outer_Y.append(Y1)
    
    # M2: (A10+A11)B00
    X2 = [[get_entry_coeffs_A(r, c, [(2,1), (3,1)]) for c in range(2)] for r in range(2)]
    Y2 = [[get_entry_coeffs_B(r, c, [(0,1)]) for c in range(2)] for r in range(2)]
    outer_X.append(X2); outer_Y.append(Y2)
    
    # M3: A00(B01-B11)
    X3 = [[get_entry_coeffs_A(r, c, [(0,1)]) for c in range(2)] for r in range(2)]
    Y3 = [[get_entry_coeffs_B(r, c, [(1,1), (3,-1)]) for c in range(2)] for r in range(2)]
    outer_X.append(X3); outer_Y.append(Y3)
    
    # M4: A11(B10-B00)
    X4 = [[get_entry_coeffs_A(r, c, [(3,1)]) for c in range(2)] for r in range(2)]
    Y4 = [[get_entry_coeffs_B(r, c, [(2,1), (0,-1)]) for c in range(2)] for r in range(2)]
    outer_X.append(X4); outer_Y.append(Y4)
    
    # M5: (A00+A01)B11
    X5 = [[get_entry_coeffs_A(r, c, [(0,1), (1,1)]) for c in range(2)] for r in range(2)]
    Y5 = [[get_entry_coeffs_B(r, c, [(3,1)]) for c in range(2)] for r in range(2)]
    outer_X.append(X5); outer_Y.append(Y5)
    
    # M6: (A10-A00)(B00+B01)
    X6 = [[get_entry_coeffs_A(r, c, [(2,1), (0,-1)]) for c in range(2)] for r in range(2)]
    Y6 = [[get_entry_coeffs_B(r, c, [(0,1), (1,1)]) for c in range(2)] for r in range(2)]
    outer_X.append(X6); outer_Y.append(Y6)
    
    # M7: (A01-A11)(B10+B11)
    X7 = [[get_entry_coeffs_A(r, c, [(1,1), (3,-1)]) for c in range(2)] for r in range(2)]
    Y7 = [[get_entry_coeffs_B(r, c, [(2,1), (3,1)]) for c in range(2)] for r in range(2)]
    outer_X.append(X7); outer_Y.append(Y7)
    
    # C reconstruction weights for M1..M7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # C blocks are 2x2.
    # C00 is block (0,0). C01 is (0,1). C10 is (1,0). C11 is (1,1).
    
    c_weights = [
        [(0, 1), (1, 1)],   # M1: C00(+1), C11(+1)
        [(2, 1), (3, -1)],  # M2: C10(+1), C11(-1)
        [(1, 1)],           # M3: C01(+1)
        [(0, 1), (2, 1)],   # M4: C00(+1), C10(+1)
        [(0, -1), (1, 1)],  # M5: C00(-1), C01(+1)
        [(3, 1)],           # M6: C11(+1)
        [(0, 1)]            # M7: C00(+1)
    ]
    
    factors_444 = []
    
    for t in range(7):
        if time.time() > deadline: break
        X_t = outer_X[t]
        Y_t = outer_Y[t]
        w_t = c_weights[t]
        
        for s in range(7):
            if time.time() > deadline: break
            u_in, v_in, w_in = strassen_2x2[s]
            
            # Global U: 16 coeffs
            U_g = [0.0] * 16
            for r in range(2):
                for c in range(2):
                    coeff = u_in[r*2+c]
                    if coeff != 0:
                        for i in range(16):
                            U_g[i] += coeff * X_t[r][c][i]
            
            # Global V: 16 coeffs
            V_g = [0.0] * 16
            for r in range(2):
                for c in range(2):
                    coeff = v_in[r*2+c]
                    if coeff != 0:
                        for i in range(16):
                            V_g[i] += coeff * Y_t[r][c][i]
            
            # Global W: 16 coeffs
            W_g = [0.0] * 16
            # w_in is for C entries of the 2x2 result Z_t
            # Z_t[r,c] contributes to C blocks specified in w_t
            for r in range(2):
                for c in range(2):
                    coeff = w_in[r*2+c]
                    if coeff != 0:
                        # Z_t[r,c] is the (r,c) entry of the 2x2 product
                        # This entry corresponds to the (r,c) entry of each C block in w_t
                        for block_idx, block_weight in w_t:
                            br = block_idx // 2
                            bc = block_idx % 2
                            global_r = 2*br + r
                            global_c = 2*bc + c
                            idx = global_r * 4 + global_c
                            W_g[idx] += coeff * block_weight
            
            factors_444.append((U_g, V_g, W_g))
            
    if time.time() > deadline:
        # Return schoolbook if timeout
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
        return {"U": U, "V": V, "W": W}

    # Build 4x4x8 decomposition
    rank_444 = len(factors_444)
    total_rank = rank_444 * 2
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(32)]
    W = [[0] * total_rank for _ in range(32)]
    
    for c in range(rank_444):
        U_g, V_g, W_g = factors_444[c]
        
        # Left half (cols 0-3 of B, 0-3 of C)
        for i in range(16):
            U[i][c] = U_g[i]
            U[i][c + rank_444] = U_g[i]
            
        for i in range(16):
            V[i][c] = V_g[i]
            V[i + 16][c] = 0
            V[i][c + rank_444] = 0
            V[i + 16][c + rank_444] = V_g[i]
            
        for i in range(16):
            W[i][c] = W_g[i]
            W[i + 16][c] = 0
            W[i][c + rank_444] = 0
            W[i + 16][c + rank_444] = W_g[i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
