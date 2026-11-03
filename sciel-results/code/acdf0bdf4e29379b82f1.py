import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 93.
    # Part 1: Strassen 4x4x4 for the first 4 columns of B. Rank 49.
    # Part 2: 4 blocks of (Strassen 2x2x2 + Schoolbook 2x2x1) for the last 3 columns of B.
    # Each block rank 11. Total 44.
    # Total rank 49 + 44 = 93.
    
    rank = 93
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen_2x2x2(u_row_offset, v_row_offset, w_row_offset, col_start):
        # Strassen 7 terms
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        
        # Indices in 2x2 block:
        # A: (0,0)->a, (0,1)->b, (1,0)->c, (1,1)->d
        # B: (0,0)->e, (0,1)->f, (1,0)->g, (1,1)->h
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # U entries for A
        # V entries for B
        # W entries for C
        
        # M1
        U[u_row_offset + 0][col_start + 0] += 1 # a
        U[u_row_offset + 3][col_start + 0] += 1 # d
        V[v_row_offset + 0][col_start + 0] += 1 # e
        V[v_row_offset + 1][col_start + 0] += 1 # h
        W[w_row_offset + 0][col_start + 0] += 1 # C00
        W[w_row_offset + 1][col_start + 0] += 1 # C11
        
        # M2
        U[u_row_offset + 2][col_start + 1] += 1 # c
        U[u_row_offset + 3][col_start + 1] += 1 # d
        V[v_row_offset + 0][col_start + 1] += 1 # e
        W[w_row_offset + 1][col_start + 1] += 1 # C10
        W[w_row_offset + 3][col_start + 1] += 1 # C11
        
        # M3
        U[u_row_offset + 0][col_start + 2] += 1 # a
        V[v_row_offset + 1][col_start + 2] += 1 # f
        V[v_row_offset + 3][col_start + 2] += -1 # h
        W[w_row_offset + 1][col_start + 2] += 1 # C01
        W[w_row_offset + 3][col_start + 2] += 1 # C11
        
        # M4
        U[u_row_offset + 3][col_start + 3] += 1 # d
        V[v_row_offset + 2][col_start + 3] += 1 # g
        V[v_row_offset + 0][col_start + 3] += -1 # e
        W[w_row_offset + 0][col_start + 3] += 1 # C00
        W[w_row_offset + 1][col_start + 3] += 1 # C10
        
        # M5
        U[u_row_offset + 0][col_start + 4] += 1 # a
        U[u_row_offset + 1][col_start + 4] += 1 # b
        V[v_row_offset + 3][col_start + 4] += 1 # h
        W[w_row_offset + 0][col_start + 4] += -1 # C00
        W[w_row_offset + 1][col_start + 4] += 1 # C01
        
        # M6
        U[u_row_offset + 2][col_start + 5] += 1 # c
        U[u_row_offset + 0][col_start + 5] += -1 # a
        V[v_row_offset + 0][col_start + 5] += 1 # e
        V[v_row_offset + 1][col_start + 5] += 1 # f
        W[w_row_offset + 3][col_start + 5] += 1 # C11
        
        # M7
        U[u_row_offset + 1][col_start + 6] += 1 # b
        U[u_row_offset + 3][col_start + 6] += -1 # d
        V[v_row_offset + 2][col_start + 6] += 1 # g
        V[v_row_offset + 3][col_start + 6] += 1 # h
        W[w_row_offset + 0][col_start + 6] += 1 # C00
        W[w_row_offset + 1][col_start + 6] += 1 # C11
        
    # Helper to add Schoolbook 2x2x1 (column vector multiplication)
    def add_schoolbook_2x2x1(u_row_offset, v_row_offset, w_row_offset, col_start):
        # A = [[a, b], [c, d]], B = [e, f]^T
        # C = [ae+bf, ce+df]^T
        # 4 terms
        # ae
        U[u_row_offset + 0][col_start + 0] += 1
        V[v_row_offset + 0][col_start + 0] += 1
        W[w_row_offset + 0][col_start + 0] += 1
        
        # bf
        U[u_row_offset + 1][col_start + 1] += 1
        V[v_row_offset + 1][col_start + 1] += 1
        W[w_row_offset + 0][col_start + 1] += 1
        
        # ce
        U[u_row_offset + 2][col_start + 2] += 1
        V[v_row_offset + 0][col_start + 2] += 1
        W[w_row_offset + 1][col_start + 2] += 1
        
        # df
        U[u_row_offset + 3][col_start + 3] += 1
        V[v_row_offset + 1][col_start + 3] += 1
        W[w_row_offset + 1][col_start + 3] += 1

    # Part 1: Strassen 4x4x4
    # We need to map 4x4 Strassen to global indices.
    # Global A: 4x4. Global B: 4x4 (cols 0-3). Global C: 4x4 (cols 0-3).
    # Strassen 4x4x4 is composed of 16 2x2x2 Strassen calls? No, it's recursive.
    # Easier: Just use the recursive structure or hardcode the 49 terms.
    # Let's use a recursive function to generate the 49 terms for 4x4x4.
    
    def generate_strassen_4x4x4():
        terms = []
        # Recursive Strassen for n=4
        # Split A, B, C into 2x2 blocks of 2x2 matrices.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mi is a 2x2x2 multiplication.
        # We need to decompose each Mi into 7 terms of 2x2x2 Strassen.
        # Total 7 * 7 = 49 terms.
        
        # We will build U, V, W contributions for these 49 terms.
        # This function returns a list of (u_indices, v_indices, w_indices, coeffs)
        # But it's easier to just call add_strassen_2x2x2 49 times with appropriate offsets.
        
        # The 49 terms correspond to the 7 Strassen steps for the 4x4 case,
        # each of which is a 2x2x2 Strassen.
        
        # Let's define the 7 high-level steps.
        # Step 1: M1 = (A00+A11)(B00+B11)
        # A00 is A[0:2, 0:2]. A11 is A[2:4, 2:4].
        # Sum A00+A11 is not a simple subset of A.
        # This makes direct mapping to U, V, W rows difficult if we just add to existing rows.
        # We need to create new "virtual" rows in U and V?
        # No, the tensor decomposition allows linear combinations in U and V.
        # So U rows can have multiple non-zero entries.
        
        # However, implementing the full recursive Strassen U, V, W generation is complex.
        # Alternative: Use the known 49-term decomposition for 4x4x4.
        # Or, simpler: Just use the 4x4x4 Strassen decomposition by calling a library?
        # No, I must write it.
        
        # Let's use a simpler approach for Part 1:
        # Use 4 independent 2x2x2 Strassen decompositions for the 4 blocks?
        # No, that would be rank 4*7 = 28? No, 4x4x4 rank is 49.
        # If I just do schoolbook for 4x4x4, rank is 64.
        # 64 + 44 = 108. Still better than 112.
        # But 49 is better.
        
        # Let's try to construct the 49 terms manually or recursively.
        # I will implement a recursive Strassen decomposition generator.
        
        pass

    # Since implementing recursive Strassen U,V,W mapping is prone to errors,
    # and the deadline is 120s, I will use a safer approach:
    # Use Schoolbook for 4x4x4 (rank 64) + 44 = 108.
    # This is a valid improvement over 112.
    # I can try to optimize the 4x4x4 part to 49 if time permits, but 108 is safe.
    
    # Let's stick to 108 first to ensure correctness.
    # Part 1: Schoolbook 4x4x4. Rank 64.
    # Part 2: 44.
    # Total 108.
    
    # Re-evaluating: Can I easily do Strassen 4x4x4?
    # Yes, if I just iterate over the 49 terms.
    # But I don't have the 49 terms hardcoded.
    # I will use Schoolbook for Part 1.
    
    # Part 1: Schoolbook 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 4 + j][col_offset] = 1
                V[j * 7 + k][col_offset] = 1
                W[k * 4 + i][col_offset] = 1
                col_offset += 1
    # col_offset is now 64.
    
    # Part 2: 4 blocks of 2x2x3 (rank 11 each)
    # Blocks:
    # 1. A00 * B_top (B cols 4-6, rows 0-1)
    # 2. A01 * B_bot (B cols 4-6, rows 2-3)
    # 3. A10 * B_top
    # 4. A11 * B_bot
    
    # Block 1: A00 (rows 0-1, cols 0-1 of A), B_top (rows 0-1, cols 4-6 of B)
    # u_row_offset = 0 (A rows 0-1, cols 0-1) -> indices 0, 1, 4, 5 in global U?
    # Global U index: i*4 + j.
    # A00: i=0,1; j=0,1. Indices: 0,1,4,5.
    # But add_strassen_2x2x2 expects u_row_offset such that u_row_offset + 0,1,2,3 map to the 2x2 block.
    # My add_strassen_2x2x2 uses indices 0,1,2,3 relative to offset.
    # It assumes the 2x2 block is contiguous in the offset?
    # No, I defined U[u_row_offset + 0], U[u_row_offset + 1], U[u_row_offset + 2], U[u_row_offset + 3].
    # This assumes the 4 rows of the 2x2 block are contiguous in U.
    # But in global U, A00 rows are 0,1 and 4,5. Not contiguous.
    
    # I need to modify the helper functions to take explicit indices or map them.
    # Let's rewrite the helpers to take explicit index lists.
    
    # Actually, it's easier to just fill the arrays directly in the loop.
    
    # Reset col_offset to 64.
    
    # Block 1: A00 * B_top
    # A00 indices: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # B_top indices: (0,4)->4, (0,5)->5, (0,6)->6, (1,4)->11, (1,5)->12, (1,6)->13
    # C00 indices: (4,0)->16, (5,0)->17, (6,0)->20? No.
    # W index: k*4 + i.
    # C00 is C[4:6, 0:2]? No.
    # C is 4x7.
    # C00 is C[0:2, 4:6].
    # W rows for C[0:2, 4:6]:
    # k=4, i=0 -> 16
    # k=4, i=1 -> 17
    # k=5, i=0 -> 20
    # k=5, i=1 -> 21
    
    # This is getting messy. I will use a generic add_term function.
    
    def add_term(u_idx, v_idx, w_idx, val):
        U[u_idx][col_offset] += val
        V[v_idx][col_offset] += val
        W[w_idx][col_offset] += val
        
    # But col_offset is shared. I need to increment it for each term.
    # The helpers above incremented col_offset implicitly? No, they took col_start.
    # I will rewrite the helpers to return the number of columns used.
    
    # Let's just use the schoolbook for the whole 4x4x7? Rank 112.
    # I want to improve it.
    # I will implement the 108 rank solution (Schoolbook 4x4x4 + 4x Rank 11 blocks).
    
    # I will rewrite the code to be robust.
    
    pass

# Final Plan:
# 1. Schoolbook 4x4x4 (rank 64).
# 2. 4 blocks of Rank 11 (2x2x2 Strassen + 2x2x1 Schoolbook).
# Total 108.

def construct():
    n, m, p = DIMENSIONS
    rank = 108
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Schoolbook 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    
    # Part 2: 4 blocks
    # Block 1: A00 * B_top
    # A00: U indices 0,1,4,5
    # B_top: V indices 4,5,6, 11,12,13
    # C00: W indices 16,17, 20,21
    
    # Strassen 2x2x2 for first 2 cols of B_top (cols 4,5)
    # A00: a=0, b=1, c=4, d=5
    # B_top: e=4, f=5, g=11, h=12
    # C00: c00=16, c01=17, c10=20, c11=21
    
    # M1: (a+d)(e+h)
    U[0][col] += 1; U[5][col] += 1
    V[4][col] += 1; V[12][col] += 1
    W[16][col] += 1; W[21][col] += 1
    col += 1
    
    # M2: (c+d)e
    U[4][col] += 1; U[5][col] += 1
    V[4][col] += 1
    W[20][col] += 1; W[21][col] += 1
    col += 1
    
    # M3: a(f-h)
    U[0][col] += 1
    V[5][col] += 1; V[12][col] += -1
    W[17][col] += 1; W[21][col] += 1
    col += 1
    
    # M4: d(g-e)
    U[5][col] += 1
    V[11][col] += 1; V[4][col] += -1
    W[16][col] += 1; W[20][col] += 1
    col += 1
    
    # M5: (a+b)h
    U[0][col] += 1; U[1][col] += 1
    V[12][col] += 1
    W[16][col] += -1; W[17][col] += 1
    col += 1
    
    # M6: (c-a)(e+f)
    U[4][col] += 1; U[0][col] += -1
    V[4][col] += 1; V[5][col] += 1
    W[21][col] += 1
    col += 1
    
    # M7: (b-d)(g+h)
    U[1][col] += 1; U[5][col] += -1
    V[11][col] += 1; V[12][col] += 1
    W[16][col] += 1; W[17][col] += 1
    col += 1
    
    # Schoolbook 2x2x1 for 3rd col of B_top (col 6)
    # B_top col 6: indices 6, 13
    # C00 col 6: indices 24, 25? No.
    # W index for k=6, i=0 is 24. i=1 is 25.
    
    # ae
    U[0][col] += 1; V[6][col] += 1; W[24][col] += 1
    col += 1
    # bf
    U[1][col] += 1; V[13][col] += 1; W[24][col] += 1
    col += 1
    # ce
    U[4][col] += 1; V[6][col] += 1; W[25][col] += 1
    col += 1
    # df
    U[5][col] += 1; V[13][col] += 1; W[25][col] += 1
    col += 1
    
    # Block 2: A01 * B_bot
    # A01: U indices 1,2,5,6? No.
    # A is 4x4.
    # A00: rows 0-1, cols 0-1.
    # A01: rows 0-1, cols 2-3.
    # U indices: i*4+j.
    # A01: i=0,1; j=2,3.
    # (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7.
    # B_bot: rows 2-3, cols 4-6.
    # V indices: j*7+k.
    # (2,4)->18, (2,5)->19, (2,6)->20, (3,4)->25, (3,5)->26, (3,6)->27.
    # C01: rows 0-1, cols 4-6.
    # W indices: k*4+i.
    # (4,0)->16, (4,1)->17, (5,0)->20, (5,1)->21, (6,0)->24, (6,1)->25.
    
    # Strassen 2x2x2 for B_bot cols 4,5
    # A01: a=2, b=3, c=6, d=7
    # B_bot: e=18, f=19, g=25, h=26
    # C01: c00=16, c01=17, c10=20, c11=21
    
    U[2][col] += 1; U[7][col] += 1
    V[18][col] += 1; V[26][col] += 1
    W[16][col] += 1; W[21][col] += 1
    col += 1
    
    U[6][col] += 1; U[7][col] += 1
    V[18][col] += 1
    W[20][col] += 1; W[21][col] += 1
    col += 1
    
    U[2][col] += 1
    V[19][col] += 1; V[26][col] += -1
    W[17][col] += 1; W[21][col] += 1
    col += 1
    
    U[7][col] += 1
    V[25][col] += 1; V[18][col] += -1
    W[16][col] += 1; W[20][col] += 1
    col += 1
    
    U[2][col] += 1; U[3][col] += 1
    V[26][col] += 1
    W[16][col] += -1; W[17][col] += 1
    col += 1
    
    U[6][col] += 1; U[2][col] += -1
    V[18][col] += 1; V[19][col] += 1
    W[21][col] += 1
    col += 1
    
    U[3][col] += 1; U[7][col] += -1
    V[25][col] += 1; V[26][col] += 1
    W[16][col] += 1; W[17][col] += 1
    col += 1
    
    # Schoolbook 2x2x1 for B_bot col 6
    # B_bot col 6: 20, 27
    # C01 col 6: 24, 25
    
    U[2][col] += 1; V[20][col] += 1; W[24][col] += 1
    col += 1
    U[3][col] += 1; V[27][col] += 1; W[24][col] += 1
    col += 1
    U[6][col] += 1; V[20][col] += 1; W[25][col] += 1
    col += 1
    U[7][col] += 1; V[27][col] += 1; W[25][col] += 1
    col += 1
    
    # Block 3: A10 * B_top
    # A10: rows 2-3, cols 0-1.
    # U indices: (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13.
    # B_top: 4,5,6, 11,12,13.
    # C10: rows 2-3, cols 4-6.
    # W indices: (4,2)->18, (4,3)->19, (5,2)->22, (5,3)->23, (6,2)->26, (6,3)->27.
    
    # Strassen
    U[8][col] += 1; U[13][col] += 1
    V[4][col] += 1; V[12][col] += 1
    W[18][col] += 1; W[23][col] += 1
    col += 1
    
    U[12][col] += 1; U[13][col] += 1
    V[4][col] += 1
    W[22][col] += 1; W[23][col] += 1
    col += 1
    
    U[8][col] += 1
    V[5][col] += 1; V[12][col] += -1
    W[19][col] += 1; W[23][col] += 1
    col += 1
    
    U[13][col] += 1
    V[11][col] += 1; V[4][col] += -1
    W[18][col] += 1; W[22][col] += 1
    col += 1
    
    U[8][col] += 1; U[9][col] += 1
    V[12][col] += 1
    W[18][col] += -1; W[19][col] += 1
    col += 1
    
    U[12][col] += 1; U[8][col] += -1
    V[4][col] += 1; V[5][col] += 1
    W[23][col] += 1
    col += 1
    
    U[9][col] += 1; U[13][col] += -1
    V[11][col] += 1; V[12][col] += 1
    W[18][col] += 1; W[19][col] += 1
    col += 1
    
    # Schoolbook
    U[8][col] += 1; V[6][col] += 1; W[26][col] += 1
    col += 1
    U[9][col] += 1; V[13][col] += 1; W[26][col] += 1
    col += 1
    U[12][col] += 1; V[6][col] += 1; W[27][col] += 1
    col += 1
    U[13][col] += 1; V[13][col] += 1; W[27][col] += 1
    col += 1
    
    # Block 4: A11 * B_bot
    # A11: rows 2-3, cols 2-3.
    # U indices: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15.
    # B_bot: 18,19,20, 25,26,27.
    # C11: rows 2-3, cols 4-6.
    # W indices: 18,19, 22,23, 26,27.
    
    # Strassen
    U[10][col] += 1; U[15][col] += 1
    V[18][col] += 1; V[26][col] += 1
    W[18][col] += 1; W[23][col] += 1
    col += 1
    
    U[14][col] += 1; U[15][col] += 1
    V[18][col] += 1
    W[22][col] += 1; W[23][col] += 1
    col += 1
    
    U[10][col] += 1
    V[19][col] += 1; V[26][col] += -1
    W[19][col] += 1; W[23][col] += 1
    col += 1
    
    U[15][col] += 1
    V[25][col] += 1; V[18][col] += -1
    W[18][col] += 1; W[22][col] += 1
    col += 1
    
    U[10][col] += 1; U[11][col] += 1
    V[26][col] += 1
    W[18][col] += -1; W[19][col] += 1
    col += 1
    
    U[14][col] += 1; U[10][col] += -1
    V[18][col] += 1; V[19][col] += 1
    W[23][col] += 1
    col += 1
    
    U[11][col] += 1; U[15][col] += -1
    V[25][col] += 1; V[26][col] += 1
    W[18][col] += 1; W[19][col] += 1
    col += 1
    
    # Schoolbook
    U[10][col] += 1; V[20][col] += 1; W[26][col] += 1
    col += 1
    U[11][col] += 1; V[27][col] += 1; W[26][col] += 1
    col += 1
    U[14][col] += 1; V[20][col] += 1; W[27][col] += 1
    col += 1
    U[15][col] += 1; V[27][col] += 1; W[27][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
