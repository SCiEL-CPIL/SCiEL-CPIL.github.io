import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indexing: U[i*2+j, c], V[i*2+j, c], W[i*2+j, c]
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen steps
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U entries (A part)
    # col 0 (M1): A00, A11
    U[0][0] = 1; U[3][0] = 1
    # col 1 (M2): A10, A11
    U[2][1] = 1; U[3][1] = 1
    # col 2 (M3): A00
    U[0][2] = 1
    # col 3 (M4): A11
    U[3][3] = 1
    # col 4 (M5): A00, A01
    U[0][4] = 1; U[1][4] = 1
    # col 5 (M6): A10, A00
    U[2][5] = 1; U[0][5] = -1
    # col 6 (M7): A01, A11
    U[1][6] = 1; U[3][6] = -1
    
    # V entries (B part)
    # col 0 (M1): B00, B11
    V[0][0] = 1; V[3][0] = 1
    # col 1 (M2): B00
    V[0][1] = 1
    # col 2 (M3): B01, B11
    V[1][2] = 1; V[3][2] = -1
    # col 3 (M4): B10, B00
    V[2][3] = 1; V[0][3] = -1
    # col 4 (M5): B11
    V[3][4] = 1
    # col 5 (M6): B00, B01
    V[0][5] = 1; V[1][5] = 1
    # col 6 (M7): B10, B11
    V[2][6] = 1; V[3][6] = 1
    
    # W entries (C part)
    # col 0 (M1): C00, C11
    W[0][0] = 1; W[3][0] = 1
    # col 1 (M2): C10, C11
    W[2][1] = 1; W[3][1] = -1
    # col 2 (M3): C01, C11
    W[1][2] = 1; W[3][2] = 1
    # col 3 (M4): C00, C10
    W[0][3] = 1; W[2][3] = 1
    # col 4 (M5): C00, C01
    W[0][4] = -1; W[1][4] = 1
    # col 5 (M6): C11
    W[3][5] = 1
    # col 6 (M7): C00
    W[0][6] = 1
    
    return U, V, W

def _build_trivial(n, m, p):
    """
    Returns U, V, W for n x m x p matrix multiplication using schoolbook.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decs, n, m, p):
    """
    Combines a list of decompositions for sub-problems into one for n x m x p.
    decs is a list of tuples (U, V, W, n_sub, m_sub, p_sub, i_offset, j_offset, k_offset).
    Wait, the offsets are complex.
    Let's just manually construct the 55 rank solution by concatenating the sub-decompositions.
    """
    pass

def construct():
    # Target: 2 x 5 x 6
    # Strategy:
    # 1. 2x3x6 part (from 2x3x3 + 2x3x3)
    # 2. 2x2x6 part (from 2x2x2 + 2x2x2 + 2x2x2)
    
    # First, get 2x3x3 decomposition (Rank 17)
    # 2x3x3 = 2x2x3 + 2x1x3
    # 2x2x3 = 2x2x2 + 2x2x1
    # 2x1x3 = trivial (Rank 6)
    
    # Let's build 2x2x2 (Rank 7)
    U_222, V_222, W_222 = _build_strassen_2x2x2()
    
    # Let's build 2x2x1 (Rank 4)
    U_221, V_221, W_221 = _build_trivial(2, 2, 1)
    
    # Let's build 2x1x3 (Rank 6)
    U_213, V_213, W_213 = _build_trivial(2, 1, 3)
    
    # Now combine to 2x3x3 (Rank 7 + 4 + 6 = 17)
    # A is 2x3. Split into A_left (2x2) and A_right (2x1).
    # B is 3x3. Split into B_top (2x3) and B_bottom (1x3).
    # C = A_left * B_top + A_right * B_bottom.
    
    # U_233: Rows 0-3 correspond to A_left, Rows 4-5 correspond to A_right.
    # Wait, A is 2x3. Indices:
    # A[0,0], A[0,1], A[0,2]
    # A[1,0], A[1,1], A[1,2]
    # A_left:
    # A[0,0], A[0,1]
    # A[1,0], A[1,1]
    # A_right:
    # A[0,2]
    # A[1,2]
    
    # B is 3x3.
    # B[0,0], B[0,1], B[0,2]
    # B[1,0], B[1,1], B[1,2]
    # B[2,0], B[2,1], B[2,2]
    # B_top:
    # B[0,0], B[0,1], B[0,2]
    # B[1,0], B[1,1], B[1,2]
    # B_bottom:
    # B[2,0], B[2,1], B[2,2]
    
    # C is 2x3.
    # C[0,0], C[0,1], C[0,2]
    # C[1,0], C[1,1], C[1,2]
    
    # Sub-problem 1: 2x2x2 (A_left, B_top_left, C_left)
    # B_top_left is 2x2: B[0,0], B[0,1], B[1,0], B[1,1].
    # C_left is 2x2: C[0,0], C[0,1], C[1,0], C[1,1].
    
    # Sub-problem 2: 2x2x1 (A_left, B_top_right, C_right)
    # B_top_right is 2x1: B[0,2], B[1,2].
    # C_right is 2x1: C[0,2], C[1,2].
    
    # Sub-problem 3: 2x1x3 (A_right, B_bottom, C_all)
    # A_right is 2x1: A[0,2], A[1,2].
    # B_bottom is 1x3: B[2,0], B[2,1], B[2,2].
    # C_all is 2x3: C[0,0..2], C[1,0..2].
    
    # This manual combination is complex. Let's just use the trivial 60 rank for now?
    # No, we need to improve.
    
    # Let's try a simpler improvement:
    # Use Strassen for the 2x2 blocks of the 2x5 and 5x6 matrices?
    # 5 is not a power of 2.
    
    # Let's implement the 55 rank solution properly.
    
    # Helper to extract sub-matrices and remap indices.
    def _remap(U, row_map, col_map):
        """
        U is (n*m, rank).
        row_map: list of new_row_index -> old_row_index (or None if not present)
        col_map: list of new_col_index -> old_col_index (or None if not present)
        Returns new U.
        """
        pass

    # This is getting too long for a single edit.
    # Let's just return the schoolbook for now?
    # No, the directive says to improve.
    
    # Let's try a random search for rank 50?
    # No, random search for exact tensor decomposition is hard.
    
    # Let's stick to the block decomposition.
    # I will write the code to construct the 55 rank decomposition explicitly.
    
    # 1. 2x2x2 Strassen (Rank 7)
    U_222, V_222, W_222 = _build_strassen_2x2x2()
    
    # 2. 2x2x1 Trivial (Rank 4)
    U_221, V_221, W_221 = _build_trivial(2, 2, 1)
    
    # 3. 2x1x3 Trivial (Rank 6)
    U_213, V_213, W_213 = _build_trivial(2, 1, 3)
    
    # Combine to 2x3x3 (Rank 17)
    # A: 2x3. B: 3x3. C: 2x3.
    # U_233: 6x17
    # V_233: 9x17
    # W_233: 6x17
    
    U_233 = [[0]*17 for _ in range(6)]
    V_233 = [[0]*17 for _ in range(9)]
    W_233 = [[0]*17 for _ in range(6)]
    
    # Part 1: 2x2x2 (A[0..1, 0..1], B[0..1, 0..1], C[0..1, 0..1])
    # A rows: 0,1,2,3 (0,1 are 0,0/0,1; 2,3 are 1,0/1,1) -> Wait, 2x2 A is 4 rows.
    # A_222 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # In 2x3 A:
    # 0:(0,0), 1:(0,1), 2:(0,2)
    # 3:(1,0), 4:(1,1), 5:(1,2)
    # So A_222 rows 0,1,2,3 map to 2x3 rows 0,1,3,4.
    
    # B_222 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # In 3x3 B:
    # 0:(0,0), 1:(0,1), 2:(0,2)
    # 3:(1,0), 4:(1,1), 5:(1,2)
    # 6:(2,0), 7:(2,1), 8:(2,2)
    # So B_222 rows 0,1,2,3 map to 3x3 rows 0,1,3,4.
    
    # C_222 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # In 2x3 C:
    # 0:(0,0), 1:(0,1), 2:(0,2)
    # 3:(1,0), 4:(1,1), 5:(1,2)
    # So C_222 rows 0,1,2,3 map to 2x3 rows 0,1,3,4.
    
    a_map = [0, 1, 3, 4]
    b_map = [0, 1, 3, 4]
    c_map = [0, 1, 3, 4]
    
    for c in range(7):
        for r in range(4):
            U_233[a_map[r]][c] = U_222[r][c]
            V_233[b_map[r]][c] = V_222[r][c]
            W_233[c_map[r]][c] = W_222[r][c]
            
    # Part 2: 2x2x1 (A[0..1, 0..1], B[0..1, 2], C[0..1, 2])
    # A rows: 0,1,2,3 -> 2x3 rows 0,1,3,4.
    # B rows: 0,1 -> 3x3 rows 2,5. (B[0,2], B[1,2])
    # C rows: 0,1 -> 2x3 rows 2,5. (C[0,2], C[1,2])
    
    a_map2 = [0, 1, 3, 4]
    b_map2 = [2, 5]
    c_map2 = [2, 5]
    
    for c in range(4):
        for r in range(4):
            U_233[a_map2[r]][c+7] = U_221[r][c]
        for r in range(2):
            V_233[b_map2[r]][c+7] = V_221[r][c]
        for r in range(2):
            W_233[c_map2[r]][c+7] = W_221[r][c]
            
    # Part 3: 2x1x3 (A[0..1, 2], B[2, 0..2], C[0..1, 0..2])
    # A rows: 0,1 -> 2x3 rows 2,5. (A[0,2], A[1,2])
    # B rows: 0,1,2 -> 3x3 rows 6,7,8. (B[2,0], B[2,1], B[2,2])
    # C rows: 0..5 -> 2x3 rows 0..5.
    
    a_map3 = [2, 5]
    b_map3 = [6, 7, 8]
    c_map3 = [0, 1, 2, 3, 4, 5]
    
    for c in range(6):
        for r in range(2):
            U_233[a_map3[r]][c+11] = U_213[r][c]
        for r in range(3):
            V_233[b_map3[r]][c+11] = V_213[r][c]
        for r in range(6):
            W_233[c_map3[r]][c+11] = W_213[r][c]
            
    # Now we have 2x3x3 (Rank 17).
    # We need 2x3x6.
    # Split 6 into 3+3.
    # B is 3x6. Split into B_left (3x3) and B_right (3x3).
    # C is 2x6. Split into C_left (2x3) and C_right (2x3).
    # A is 2x3. Same A.
    
    # U_236: 6x34
    # V_236: 18x34
    # W_236: 12x34
    
    U_236 = [[0]*34 for _ in range(6)]
    V_236 = [[0]*34 for _ in range(18)]
    W_236 = [[0]*34 for _ in range(12)]
    
    # Left part (cols 0-2 of B, cols 0-2 of C)
    # V rows: 0..8 map to 0..8.
    # W rows: 0..5 map to 0..5.
    for c in range(17):
        for r in range(6):
            U_236[r][c] = U_233[r][c]
            V_236[r][c] = V_233[r][c]
            W_236[r][c] = W_233[r][c]
            
    # Right part (cols 3-5 of B, cols 3-5 of C)
    # V rows: 9..17 map to 0..8.
    # W rows: 6..11 map to 0..5.
    for c in range(17):
        for r in range(6):
            U_236[r][c+17] = U_233[r][c]
            V_236[r+9][c+17] = V_233[r][c]
            W_236[r+6][c+17] = W_233[r][c]
            
    # Now we have 2x3x6 (Rank 34).
    # We need 2x5x6.
    # Split 5 into 3+2.
    # A is 2x5. Split into A_left (2x3) and A_right (2x2).
    # B is 5x6. Split into B_top (3x6) and B_bottom (2x6).
    # C is 2x6. Same C.
    
    # We need 2x2x6 decomposition.
    # Split 6 into 2+2+2.
    # 3 * 2x2x2 (Rank 7) = Rank 21.
    
    U_222, V_222, W_222 = _build_strassen_2x2x2()
    
    U_226 = [[0]*21 for _ in range(4)]
    V_226 = [[0]*21 for _ in range(12)]
    W_226 = [[0]*21 for _ in range(12)]
    
    # Block 1: B cols 0-1, C cols 0-1
    for c in range(7):
        for r in range(4):
            U_226[r][c] = U_222[r][c]
            V_226[r][c] = V_222[r][c]
            W_226[r][c] = W_222[r][c]
            
    # Block 2: B cols 2-3, C cols 2-3
    for c in range(7):
        for r in range(4):
            U_226[r][c+7] = U_222[r][c]
            V_226[r+4][c+7] = V_222[r][c]
            W_226[r+4][c+7] = W_222[r][c]
            
    # Block 3: B cols 4-5, C cols 4-5
    for c in range(7):
        for r in range(4):
            U_226[r][c+14] = U_222[r][c]
            V_226[r+8][c+14] = V_222[r][c]
            W_226[r+8][c+14] = W_222[r][c]
            
    # Now combine 2x3x6 and 2x2x6 to get 2x5x6.
    # Total Rank 34 + 21 = 55.
    
    U = [[0]*55 for _ in range(10)] # 2x5
    V = [[0]*55 for _ in range(30)] # 5x6
    W = [[0]*55 for _ in range(12)] # 6x2
    
    # A_left (2x3) rows: 0,1,2,3,4,5 map to 0,1,2,3,4,5 in 2x5?
    # 2x5 A:
    # 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(0,4)
    # 5:(1,0), 6:(1,1), 7:(1,2), 8:(1,3), 9:(1,4)
    # A_left (2x3):
    # 0:(0,0), 1:(0,1), 2:(0,2)
    # 3:(1,0), 4:(1,1), 5:(1,2)
    # Map: 0->0, 1->1, 2->2, 3->5, 4->6, 5->7
    
    a_map_left = [0, 1, 2, 5, 6, 7]
    
    # B_top (3x6) rows: 0..17 map to 0..17 in 5x6?
    # 5x6 B:
    # 0:(0,0)..5:(0,5)
    # 6:(1,0)..11:(1,5)
    # 12:(2,0)..17:(2,5)
    # 18:(3,0)..23:(3,5)
    # 24:(4,0)..29:(4,5)
    # B_top (3x6):
    # 0:(0,0)..5:(0,5)
    # 6:(1,0)..11:(1,5)
    # 12:(2,0)..17:(2,5)
    # Map: 0..17 -> 0..17
    
    b_map_top = list(range(18))
    
    # C (2x6) rows: 0..11 map to 0..11 in 6x2?
    # 6x2 C:
    # 0:(0,0), 1:(0,1)
    # 2:(1,0), 3:(1,1)
    # ...
    # 10:(5,0), 11:(5,1)
    # C is 2x6.
    # 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(0,4), 5:(0,5)
    # 6:(1,0), 7:(1,1), 8:(1,2), 9:(1,3), 10:(1,4), 11:(1,5)
    # Map to 6x2 C:
    # (0,0)->0, (0,1)->1, (0,2)->2, (0,3)->3, (0,4)->4, (0,5)->5
    # (1,0)->6, (1,1)->7, (1,2)->8, (1,3)->9, (1,4)->10, (1,5)->11
    # Wait, W index is k*n + i.
    # C[i,k].
    # W[k*n + i].
    # For 2x6 C, i in 0..1, k in 0..5.
    # W index k*2 + i.
    # (0,0)->0, (0,1)->1
    # (1,0)->2, (1,1)->3
    # (2,0)->4, (2,1)->5
    # (3,0)->6, (3,1)->7
    # (4,0)->8, (4,1)->9
    # (5,0)->10, (5,1)->11
    
    c_map = list(range(12))
    
    # Fill Left Part (2x3x6)
    for c in range(34):
        for r in range(6):
            U[a_map_left[r]][c] = U_236[r][c]
            V[b_map_top[r]][c] = V_236[r][c]
            W[c_map[r]][c] = W_236[r][c]
            
    # Fill Right Part (2x2x6)
    # A_right (2x2) rows: 0,1,2,3 map to 3,4,8,9 in 2x5?
    # A_right:
    # 0:(0,3), 1:(0,4)
    # 2:(1,3), 3:(1,4)
    # Map: 0->3, 1->4, 2->8, 3->9
    
    a_map_right = [3, 4, 8, 9]
    
    # B_bottom (2x6) rows: 0..11 map to 18..29 in 5x6?
    # B_bottom:
    # 0:(3,0)..5:(3,5)
    # 6:(4,0)..11:(4,5)
    # Map: 0..5 -> 18..23, 6..11 -> 24..29
    
    b_map_bottom = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    
    # C (2x6) same map.
    
    for c in range(21):
        for r in range(4):
            U[a_map_right[r]][c+34] = U_226[r][c]
            V[b_map_bottom[r]][c+34] = V_226[r][c]
            W[c_map[r]][c+34] = W_226[r][c]
            
    return {"U": U, "V": V, "W": W}
