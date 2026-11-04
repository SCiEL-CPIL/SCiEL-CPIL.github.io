import time
import sys

DIMENSIONS = [3, 4, 7]

def get_strassen_222():
    # Returns U, V, W for 2x2x2 Strassen
    # U: 4x7, V: 4x7, W: 4x7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 3: A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 4: A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 5: (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6: (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def get_rank1_2_2():
    # 1x2x2 schoolbook
    U = [[1,0],[0,1]]
    V = [[1,0],[0,1]]
    W = [[1,0],[0,1]]
    return U, V, W

def get_rank2_2_1():
    # 2x2x1 schoolbook
    U = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    W = [[1,1,1,1]]
    return U, V, W

def get_2_2_7():
    # 2x2x7
    # Split 7 into 2, 2, 2, 1
    # U: 4x25, V: 14x25, W: 14x25
    U = [[0]*25 for _ in range(4)]
    V = [[0]*25 for _ in range(14)]
    W = [[0]*25 for _ in range(14)]
    
    col = 0
    # Part 1: 2x2x2 (cols 0,1 of B)
    U1, V1, W1 = get_strassen_222()
    for i in range(4):
        for c in range(7):
            U[i][col+c] = U1[i][c]
            V[i*2 + c//2][col+c] = V1[i][c] # B index: row 0,1. col 0,1. idx = row*2+col? No, B is 2x2.
            # V index for 2x2 B: row*2+col.
            # Wait, V1 is 4x7. Rows 0..3.
            # In get_strassen, V rows are 0..3.
            # Here V rows are 0..13.
            # B part 1 is rows 0,1 of B (global), cols 0,1.
            # Local B index r. Global B index r.
            W[i*2 + c//2][col+c] = W1[i][c] # C index: row*2+col.
            # W1 rows 0..3.
            # C part 1 is rows 0,1, cols 0,1.
    
    col += 7
    # Part 2: 2x2x2 (cols 2,3 of B)
    U1, V1, W1 = get_strassen_222()
    for i in range(4):
        for c in range(7):
            U[i][col+c] = U1[i][c]
            V[i*2 + c//2 + 2][col+c] = V1[i][c] # B cols 2,3. idx = row*2 + (col+2)
            W[i*2 + c//2 + 2][col+c] = W1[i][c] # C cols 2,3. idx = row*2 + (col+2)
            
    col += 7
    # Part 3: 2x2x2 (cols 4,5 of B)
    U1, V1, W1 = get_strassen_222()
    for i in range(4):
        for c in range(7):
            U[i][col+c] = U1[i][c]
            V[i*2 + c//2 + 4][col+c] = V1[i][c] # B cols 4,5. idx = row*2 + (col+4)
            W[i*2 + c//2 + 4][col+c] = W1[i][c] # C cols 4,5. idx = row*2 + (col+4)
            
    col += 7
    # Part 4: 2x2x1 (col 6 of B)
    U4, V4, W4 = get_rank2_2_1()
    for i in range(4):
        U[i][col+i] = U4[i][i]
        V[i*2 + 6][col+i] = V4[i][i] # B col 6. idx = row*2+6
        W[i*2 + 6][col+i] = W4[i][i] # C col 6. idx = row*2+6
        
    return U, V, W

def get_1_2_7():
    # 1x2x7
    # U: 2x14, V: 14x14, W: 7x14
    U = [[0]*14 for _ in range(2)]
    V = [[0]*14 for _ in range(14)]
    W = [[0]*14 for _ in range(7)]
    
    for k in range(7):
        for j in range(2):
            idx = k*2 + j
            U[j][idx] = 1
            V[j*7 + k][idx] = 1 # B row j, col k. idx = j*7+k
            W[k][idx] = 1
    
    return U, V, W

def get_3_2_7():
    # 3x2x7
    # Split A into 2x2 and 1x2
    # U: 6x39, V: 14x39, W: 21x39
    U = [[0]*39 for _ in range(6)]
    V = [[0]*39 for _ in range(14)]
    W = [[0]*39 for _ in range(21)]
    
    col = 0
    # Part 1: 2x2x7 (A rows 0,1)
    U1, V1, W1 = get_2_2_7()
    for i in range(4):
        for c in range(25):
            U[i][col+c] = U1[i][c]
            V[i*2 + c//2][col+c] = V1[i][c] # Wait, V1 is 14x25.
            # V1 rows correspond to B (2x7).
            # Here V rows are 0..13.
            # B is same.
            # W1 rows correspond to C (2x7).
            # Here W rows are 0..20.
            # C part 1 is rows 0,1.
            W[i*2 + c//2][col+c] = W1[i][c] # C row 0,1. idx = row*7+col
            
    col += 25
    # Part 2: 1x2x7 (A row 2)
    U2, V2, W2 = get_1_2_7()
    for i in range(2):
        for c in range(14):
            U[4+i][col+c] = U2[i][c] # A row 2. Local row i. Global row 4+i? No.
            # A is 3x2. Rows 0,1,2. Cols 0,1.
            # U rows: 0..5.
            # Part 1 uses A rows 0,1. U rows 0..3.
            # Part 2 uses A row 2. U rows 4,5.
            # U2 is 2x14. Rows 0,1.
            # Map U2[0] -> U[4], U2[1] -> U[5].
            V[i*7 + c//2][col+c] = V2[i][c] # B is same.
            W[2*7 + c][col+c] = W2[i][c] # C row 2. idx = 2*7 + col
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # 3x4x7
    # Split B into 2x7 and 2x7
    # U: 12x78, V: 28x78, W: 21x78
    U = [[0]*78 for _ in range(12)]
    V = [[0]*78 for _ in range(28)]
    W = [[0]*78 for _ in range(21)]
    
    col = 0
    # Part 1: A_left (3x2) * B_top (2x7)
    # A_left: rows 0,1,2. cols 0,1.
    # B_top: rows 0,1. cols 0..6.
    # C: rows 0,1,2. cols 0..6.
    U1, V1, W1 = get_3_2_7()
    for i in range(6):
        for c in range(39):
            U[i][col+c] = U1[i][c] # A rows 0..2, cols 0..1. U idx = row*2+col.
            V[i*2 + c//2][col+c] = V1[i][c] # B rows 0..1, cols 0..6. V idx = row*7+col.
            W[i*2 + c//2][col+c] = W1[i][c] # C rows 0..2, cols 0..6. W idx = col*3+row?
            # Task says W[k*n+i, r]. k=col, i=row.
            # My W1 from get_3_2_7:
            # In get_3_2_7, W is 21x39.
            # W index in get_3_2_7:
            # Part 1 (2x2x7): W1 is 14x25. C rows 0,1.
            # In get_2_2_7, W index was row*2+col?
            # Let's check get_2_2_7 W index.
            # In get_2_2_7, W is 14x25.
            # I wrote: W[i*2 + c//2][col+c] = W1[i][c]
            # i is U index (0..3). c is col (0..6).
            # This mapping is wrong.
            # I need to be consistent with W[k*n+i, r].
            
    # Let's re-verify the index mapping in the helper functions.
    # It's safer to generate the final U, V, W directly with correct global indices.
    
    # I will rewrite the helpers to return local indices, and then map them.
    # Or just build the final matrices in construct() using the logic.
    
    # Given time, I will stick to the 84 rank if the mapping is too error prone.
    # But 78 is a good improvement.
    # Let's try to fix the mapping.
    
    # W index: k*n + i.
    # In get_3_2_7, C is 3x7.
    # Part 1 (2x2x7): C rows 0,1.
    # W1 from get_2_2_7: 14x25.
    # In get_2_2_7, I used W[i*2 + c//2].
    # i is 0..3 (U index). c is 0..6 (col).
    # This is not k*n+i.
    
    # I will create a new helper that explicitly handles the W index.
    
    return {"U": U, "V": W, "W": W} # Placeholder

# I will output the 84 rank program to be safe, as the 78 rank implementation is complex and error-prone in this format.
# However, the directive asks for a stronger algorithm.
# I will provide the 78 rank implementation with careful index mapping.

def construct():
    n, m, p = DIMENSIONS
    # 3x4x7
    # Rank 78
    
    U = [[0]*78 for _ in range(12)]
    V = [[0]*78 for _ in range(28)]
    W = [[0]*78 for _ in range(21)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    # a_rows, a_cols, b_rows, b_cols, c_rows, c_cols
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, offset):
        nonlocal col
        # U: 4x7
        # V: 4x7
        # W: 4x7
        # Mapping:
        # U local i (0..3) -> global U[(a_r+i//2)*2 + (a_c+i%2)]? No.
        # A is 2x2.
        # U rows: 0,1,2,3.
        # Global A index: row*2+col.
        # Local A index: i.
        # Map i -> (a_r + i//2, a_c + i%2).
        # Global U index: (a_r + i//2)*m + (a_c + i%2).
        
        # V local i -> global V[(b_r + i//2)*p + (b_c + i%2)].
        # W local i -> global W[(c_c + i%2)*n + (c_r + i//2)].
        
        U1, V1, W1 = get_strassen_222()
        for i in range(4):
            for k in range(7):
                U[(a_r + i//2)*m + (a_c + i%2)][offset+k] = U1[i][k]
                V[(b_r + i//2)*p + (b_c + i%2)][offset+k] = V1[i][k]
                W[(c_c + i%2)*n + (c_r + i//2)][offset+k] = W1[i][k]
        col += 7
        return col

    # Helper to add 2x2x1
    def add_221(a_r, a_c, b_r, b_c, c_r, c_c, offset):
        nonlocal col
        U4, V4, W4 = get_rank2_2_1()
        for i in range(4):
            U[(a_r + i//2)*m + (a_c + i%2)][offset+i] = U4[i][i]
            V[(b_r + i//2)*p + (b_c + i%2)][offset+i] = V4[i][i]
            W[(c_c + i%2)*n + (c_r + i//2)][offset+i] = W4[i][i]
        col += 4
        return col

    # Helper to add 1x2x7
    def add_127(a_r, a_c, b_r, b_c, c_r, c_c, offset):
        nonlocal col
        for k in range(7):
            for j in range(2):
                idx = offset + k*2 + j
                U[(a_r + j)*m + (a_c + 0)] # A is 1x2. Row a_r. Col a_c+j.
                # Wait, A is 1x2.
                # U row: (a_r)*m + (a_c+j).
                U[a_r*m + (a_c+j)][idx] = 1
                V[(b_r + j)*p + (b_c + k)][idx] = 1
                W[(c_c + k)*n + (c_r + 0)][idx] = 1
        col += 14
        return col

    # 3x4x7
    # Split B into B_top (2x7) and B_bot (2x7)
    # A_left (3x2) * B_top
    # A_right (3x2) * B_bot
    
    # A_left: rows 0,1,2. cols 0,1.
    # B_top: rows 0,1. cols 0..6.
    # C: rows 0,1,2. cols 0..6.
    
    # A_left * B_top
    # Split A_left into A_top (2x2) and A_bot (1x2).
    
    # A_top (2x2) * B_top (2x7)
    # Split B_top into B1(2x2), B2(2x2), B3(2x2), B4(2x1).
    
    add_strassen(0, 0, 0, 0, 0, 0, col)
    add_strassen(0, 0, 0, 2, 0, 2, col)
    add_strassen(0, 0, 0, 4, 0, 4, col)
    add_221(0, 0, 0, 6, 0, 6, col)
    
    # A_bot (1x2) * B_top (2x7)
    add_127(2, 0, 0, 0, 2, 0, col)
    
    # A_right (3x2) * B_bot (2x7)
    # A_right: rows 0,1,2. cols 2,3.
    # B_bot: rows 2,3. cols 0..6.
    
    # A_top_right (2x2) * B_bot (2x7)
    add_strassen(0, 2, 2, 0, 0, 0, col)
    add_strassen(0, 2, 2, 2, 0, 2, col)
    add_strassen(0, 2, 2, 4, 0, 4, col)
    add_221(0, 2, 2, 6, 0, 6, col)
    
    # A_bot_right (1x2) * B_bot (2x7)
    add_127(2, 2, 2, 0, 2, 0, col)
    
    return {"U": U, "V": V, "W": W}

