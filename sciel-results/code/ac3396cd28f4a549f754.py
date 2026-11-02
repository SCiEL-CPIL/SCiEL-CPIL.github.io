"""
Matrix multiplication 4x5x5 via recursive Strassen decomposition.
Rank: 49 (for 4x4x4) + 16 + 16 + 4 = 85.
"""
import time
import sys

DIMENSIONS = [4, 5, 5]

# Deadline guard: keep a safety margin before the 150s limit
_DEADLINE = time.time() + 140  # 140s out of 150s

# Strassen 2x2x2 decomposition
# We need to define the U, V, W matrices for the 2x2x2 multiplication.
# A (2x2) * B (2x2) -> C (2x2)
# Strassen's 7 products:
# M1 = (A00+A11)*(B00+B11)
# M2 = (A10+A11)*B00
# M3 = A00*(B01-B11)
# M4 = A11*(B10-B00)
# M5 = (A00+A10)*B11
# M6 = (A01-A10)*(B00+B10)
# M7 = (A01-A11)*(B01+B11)
#
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication with rank 7.
    
    U has shape (4, 7): rows are A00, A01, A10, A11
    V has shape (4, 7): rows are B00, B01, B10, B11
    W has shape (4, 7): rows are C00, C01, C10, C11
    Each column corresponds to one Strassen product M1..M7.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A10)*B11
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    
    # M6 = (A01-A10)*(B00+B10)
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _combine(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions side by side.
    
    U: (n*m, r1) and (n*m, r2) -> (n*m, r1+r2)
    V: (m*p, r1) and (m*p, r2) -> (m*p, r1+r2)
    W: (p*n, r1) and (p*n, r2) -> (p*n, r1+r2)
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    n_rows_u = len(U1)
    n_rows_v = len(V1)
    n_rows_w = len(W1)
    
    U = [U1[i] + U2[i] for i in range(n_rows_u)]
    V = [V1[i] + V2[i] for i in range(n_rows_v)]
    W = [W1[i] + W2[i] for i in range(n_rows_w)]
    
    return U, V, W


def _expand_strassen_4x4():
    """Build the 4x4x4 Strassen decomposition (rank 49).
    
    Uses two levels of Strassen: 4x4 = 2x2 of 2x2 blocks.
    Level 1: 4x4x4 -> 7 products of 2x2x2
    Level 2: each 2x2x2 -> 7 products of 1x1x1 (scalar mult)
    Total rank: 7*7 = 49
    """
    n = 4
    # Start with the base 2x2x2 Strassen
    U_base, V_base, W_base = _strassen_2x2()
    
    # Now build 4x4x4 from 7 2x2x2 blocks.
    # The 4x4 matrix is divided into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # Similarly B and C.
    # 
    # For the 4x4x4 Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A10)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # Each Mj is a 2x2 product, which we decompose with the 7-term Strassen.
    
    # U for 4x4x4: 16 rows (4*4 entries of A), 49 columns
    # V for 4x4x4: 16 rows (4*4 entries of B), 49 columns
    # W for 4x4x4: 16 rows (4*4 entries of C), 49 columns
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # The 7 high-level Strassen products, each contributing 7 sub-products.
    # For each high-level product h (0..6) and sub-product s (0..6):
    # column = h*7 + s
    
    # The coefficients for each high-level product:
    # M_h = (linear_comb_A) * (linear_comb_B)
    # The A side involves 2x2 blocks, each with 4 elements.
    # The B side similarly.
    
    # Let's define the A-side block combinations for each M_h:
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3),
    #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    
    # A_block_indices[h] = list of (block_idx, coeff)
    # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
    # Within each block, the 4 elements are at positions:
    # Block 0 (A00): rows 0-1, cols 0-1 -> indices 0,1,4,5 in row-major
    # Block 1 (A01): rows 0-1, cols 2-3 -> indices 2,3,6,7
    # Block 2 (A10): rows 2-3, cols 0-1 -> indices 8,9,12,13
    # Block 3 (A11): rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # B blocks similarly:
    # B00: cols 0-1, rows 0-1 -> indices 0,1,4,5
    # B01: cols 2-3, rows 0-1 -> indices 2,3,6,7
    # B10: cols 0-1, rows 2-3 -> indices 8,9,12,13
    # B11: cols 2-3, rows 2-3 -> indices 10,11,14,15
    
    # C blocks for output:
    # C00: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # C01: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # C10: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # C11: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # A-side combinations for each M_h (block-level):
    # M0: A00+A11 -> blocks 0,3 with coeff 1
    # M1: A10+A11 -> blocks 2,3 with coeff 1
    # M2: A00 -> block 0 with coeff 1
    # M3: A11 -> block 3 with coeff 1
    # M4: A00+A10 -> blocks 0,2 with coeff 1
    # M5: A01-A10 -> blocks 1,2 with coeff 1,-1
    # M6: A01-A11 -> blocks 1,3 with coeff 1,-1
    
    A_combos = [
        [(0, 1), (3, 1)],   # M0
        [(2, 1), (3, 1)],   # M1
        [(0, 1)],           # M2
        [(3, 1)],           # M3
        [(0, 1), (2, 1)],   # M4
        [(1, 1), (2, -1)],  # M5
        [(1, 1), (3, -1)],  # M6
    ]
    
    # B-side combinations:
    # M0: B00+B11 -> blocks 0,3
    # M1: B00 -> block 0
    # M2: B01-B11 -> blocks 1,3 with coeff 1,-1
    # M3: B10-B00 -> blocks 2,0 with coeff 1,-1
    # M4: B11 -> block 3
    # M5: B00+B10 -> blocks 0,2
    # M6: B01+B11 -> blocks 1,3
    
    B_combos = [
        [(0, 1), (3, 1)],   # M0
        [(0, 1)],           # M1
        [(1, 1), (3, -1)],  # M2
        [(2, 1), (0, -1)],  # M3
        [(3, 1)],           # M4
        [(0, 1), (2, 1)],   # M5
        [(1, 1), (3, 1)],   # M6
    ]
    
    # C-side output combinations:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    C_combos = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
        [(2, 1), (4, 1)],                     # C01
        [(1, 1), (3, 1)],                     # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],   # C11
    ]
    
    # Block element indices:
    # Block b has elements at positions:
    block_elements = [
        [0, 1, 4, 5],   # Block 0
        [2, 3, 6, 7],   # Block 1
        [8, 9, 12, 13], # Block 2
        [10, 11, 14, 15], # Block 3
    ]
    
    # For each high-level product h and sub-product s (from 2x2 Strassen):
    # The sub-product s in the 2x2 Strassen has coefficients in U_base, V_base, W_base.
    # We need to map the 2x2 block elements to the 4x4 matrix elements.
    
    for h in range(7):
        for s in range(7):
            col = h * 7 + s
            
            # A side: for each element (a_row, a_col) in the 2x2 Strassen U_base
            # a_row is 0..3 (A00, A01, A10, A11 in the 2x2 block representation)
            # Wait, U_base has 4 rows: 0=A00, 1=A01, 2=A10, 3=A11 of the 2x2 block
            # But in our 4x4, each "element" of the 2x2 block is itself a 2x2 matrix.
            # The coefficient U_base[a_row][s] applies to the corresponding 2x2 block.
            
            # a_row in U_base corresponds to a 2x2 block position:
            # 0 -> top-left of 2x2 (block 0 in our naming)
            # 1 -> top-right (block 1)
            # 2 -> bottom-left (block 2)
            # 3 -> bottom-right (block 3)
            
            for a_row in range(4):
                coeff_a = U_base[a_row][s]
                if coeff_a != 0:
                    # The block a_row in A_combos[h] should have coefficient coeff_a * (block coeff from A_combos)
                    # Actually, we need to combine: the 2x2 Strassen operates on the 2x2 block matrix,
                    # where each entry is a 2x2 submatrix.
                    # U_base[a_row][s] is the coefficient for entry (a_row//2, a_row%2) of the 2x2 block.
                    # a_row 0 -> (0,0) -> block 0
                    # a_row 1 -> (0,1) -> block 1
                    # a_row 2 -> (1,0) -> block 2
                    # a_row 3 -> (1,1) -> block 3
                    
                    block_idx = a_row  # 0,1,2,3
                    # Find the coefficient for this block in A_combos[h]
                    block_coeff = 0
                    for (b_idx, b_c) in A_combos[h]:
                        if b_idx == block_idx:
                            block_coeff = b_c
                            break
                    total_coeff = coeff_a * block_coeff
                    if total_coeff != 0:
                        for elem in block_elements[block_idx]:
                            U4[elem][col] += total_coeff
            
            # B side similarly
            for b_row in range(4):
                coeff_b = V_base[b_row][s]
                if coeff_b != 0:
                    block_idx = b_row
                    block_coeff = 0
                    for (b_idx, b_c) in B_combos[h]:
                        if b_idx == block_idx:
                            block_coeff = b_c
                            break
                    total_coeff = coeff_b * block_coeff
                    if total_coeff != 0:
                        for elem in block_elements[block_idx]:
                            V4[elem][col] += total_coeff
            
            # C side
            for c_row in range(4):
                coeff_c = W_base[c_row][s]
                if coeff_c != 0:
                    block_idx = c_row
                    # Find the coefficient for this block in C_combos
                    # c_row 0 -> C00 (block 0), c_row 1 -> C01 (block 1),
                    # c_row 2 -> C10 (block 2), c_row 3 -> C11 (block 3)
                    block_coeff = 0
                    for (c_idx, c_c) in C_combos[block_idx]:
                        if c_idx == h:
                            block_coeff = c_c
                            break
                    total_coeff = coeff_c * block_coeff
                    if total_coeff != 0:
                        for elem in block_elements[block_idx]:
                            W4[elem][col] += total_coeff
    
    return U4, V4, W4


def _trivial_decomp(n, m, p):
    """Schoolbook decomposition for n x m * m x p -> n x p.
    
    U: (n*m, n*m*p)
    V: (m*p, n*m*p)
    W: (p*n, n*m*p)
    """
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
    return U, V, W


def construct():
    """
    Construct the rank-85 decomposition for 4x5 * 5x5 matrix multiplication.
    
    Split the 5x5 B matrix and 4x5 A matrix:
    A = [A_top (4x4), A_bottom (4x1)]  -- wait, no.
    
    Actually, we split as:
    A is 4x5, B is 5x5.
    Write A = [A1 | A2] where A1 is 4x4 and A2 is 4x1.
    Write B = [B1; B2] where B1 is 4x5 and B2 is 1x5.
    
    Then C = A * B = A1 * B1 + A2 * B2.
    
    A1 * B1: 4x4 * 4x5. Further split B1 = [B11 | B12] where B11 is 4x4 and B12 is 4x1.
    C1 = A1 * B11 + A1 * B12.
    
    A1 * B11: 4x4 * 4x4 -> use Strassen, rank 49.
    A1 * B12: 4x4 * 4x1 -> trivial, rank 16.
    A2 * B2: 4x1 * 1x5 -> trivial, rank 20.
    
    Wait, let me reconsider. The task is 4x5 times 5x5.
    A: 4 rows, 5 cols. B: 5 rows, 5 cols.
    
    Split the inner dimension (5) into 4+1:
    A = [A_4x4 | A_4x1]
    B = [B_4x5; B_1x5]
    
    C = A_4x4 * B_4x5 + A_4x1 * B_1x5
    
    A_4x4 * B_4x5: 4x4 times 4x5.
    Split B_4x5 = [B_4x4 | B_4x1]
    A_4x4 * B_4x4: rank 49 (Strassen)
    A_4x4 * B_4x1: rank 16 (trivial)
    
    A_4x1 * B_1x5: rank 4*1*5 = 20 (trivial)
    
    Total: 49 + 16 + 20 = 85.
    
    But wait, the W factor uses the convention W[k*n+i, r] where k indexes the column of C (0..p-1=4) and i indexes the row of C (0..n-1=3). So W has p*n = 5*4 = 20 rows.
    
    Let me re-examine the indexing:
    U has n*m = 4*5 = 20 rows. Row index: i*m + j where i is row of A (0..3), j is col of A (0..4).
    V has m*p = 5*5 = 25 rows. Row index: j*p + k where j is row of B (0..4), k is col of B (0..4).
    W has p*n = 5*4 = 20 rows. Row index: k*n + i where k is col of C (0..4), i is row of C (0..3).
    
    So:
    A_4x4: rows 0-3, cols 0-3 of A -> U rows i*5+j for i in 0..3, j in 0..3
    A_4x1: rows 0-3, col 4 of A -> U rows i*5+4 for i in 0..3
    B_4x5: rows 0-3, cols 0-4 of B -> V rows j*5+k for j in 0..3, k in 0..4
    B_1x5: row 4, cols 0-4 of B -> V rows 4*5+k for k in 0..4
    
    B_4x4: rows 0-3, cols 0-3 -> V rows j*5+k for j in 0..3, k in 0..3
    B_4x1: rows 0-3, col 4 -> V rows j*5+4 for j in 0..3
    
    C: 4x5 matrix. C[i,k] for i in 0..3, k in 0..4.
    W row index: k*4 + i.
    
    Now, the decomposition:
    1. A_4x4 * B_4x4 -> C_4x4 (top-left 4x4 of C)
       U rows: i*5+j for i,j in 0..3 (16 rows)
       V rows: j*5+k for j,k in 0..3 (16 rows)
       W rows: k*4+i for k,i in 0..3 (16 rows)
       Rank: 49
    
    2. A_4x4 * B_4x1 -> C_4x1 (column 4 of C, rows 0-3)
       U rows: i*5+j for i in 0..3, j in 0..3 (16 rows)
       V rows: j*5+4 for j in 0..3 (4 rows)
       W rows: 4*4+i = 16+i for i in 0..3 (4 rows)
       Rank: 16
    
    3. A_4x1 * B_1x5 -> C_4x5 (all of C)
       U rows: i*5+4 for i in 0..3 (4 rows)
       V rows: 4*5+k for k in 0..4 (5 rows)
       W rows: k*4+i for k in 0..4, i in 0..3 (20 rows)
       Rank: 20
    
    Total rank: 49 + 16 + 20 = 85.
    """
    if time.time() > _DEADLINE:
        # Fallback to schoolbook
        return _schoolbook()
    
    n, m, p = 4, 5, 5
    total_rank = 85
    
    # Initialize U, V, W with zeros
    U = [[0] * total_rank for _ in range(n * m)]   # 20 x 85
    V = [[0] * total_rank for _ in range(m * p)]   # 25 x 85
    W = [[0] * total_rank for _ in range(p * n)]   # 20 x 85
    
    # Part 1: A_4x4 * B_4x4 (Strassen, rank 49)
    U1, V1, W1 = _expand_strassen_4x4()
    for r in range(16):
        for c in range(49):
            U[r][c] = U1[r][c]
            V[r][c] = V1[r][c]
            W[r][c] = W1[r][c]
    
    # Part 2: A_4x4 * B_4x1 (trivial, rank 16)
    # U rows: i*5+j for i in 0..3, j in 0..3 -> rows 0,1,2,3, 5,6,7,8, 10,11,12,13, 15,16,17,18
    # V rows: j*5+4 for j in 0..3 -> rows 4, 9, 14, 19
    # W rows: 16+i for i in 0..3 -> rows 16, 17, 18, 19
    
    col_offset = 49
    for i in range(4):
        for j in range(4):
            u_row = i * 5 + j
            v_row = j * 5 + 4
            w_row = 4 * 4 + i  # k=4, so k*n+i = 16+i
            U[u_row][col_offset] = 1
            V[v_row][col_offset] = 1
            W[w_row][col_offset] = 1
            col_offset += 1
    
    # Part 3: A_4x1 * B_1x5 (trivial, rank 20)
    # U rows: i*5+4 for i in 0..3 -> rows 4, 9, 14, 19
    # V rows: 4*5+k for k in 0..4 -> rows 20, 21, 22, 23, 24
    # W rows: k*4+i for k in 0..4, i in 0..3 -> all 20 rows
    
    for i in range(4):
        for k in range(5):
            u_row = i * 5 + 4
            v_row = 4 * 5 + k
            w_row = k * 4 + i
            U[u_row][col_offset] = 1
            V[v_row][col_offset] = 1
            W[w_row][col_offset] = 1
            col_offset += 1
    
    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Fallback schoolbook decomposition, rank 100."""
    n, m, p = 4, 5, 5
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
