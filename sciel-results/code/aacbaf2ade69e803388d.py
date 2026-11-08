import numpy as np

def construct():
    n, m, p = 2, 4, 7
    
    # Base Tensor: Strassen for 2x2x2 (Rank 7)
    # U, V, W are lists of lists of coefficients
    # Indices: A(2x2), B(2x2), C(2x2)
    # Flattening: A[i*2+j], B[i*2+j], C[i*2+j] (Note: Task uses W[k*n+i], so C is column-major in k? 
    # Task: W[k*n+i, r]. So C index is k*2 + i.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7

    def get_strassen_222():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # A indices: 00:0, 01:1, 10:2, 11:3
        # B indices: 00:0, 01:1, 10:2, 11:3
        # C indices: 00:0, 01:1, 10:2, 11:3 (k*n+i -> 0*2+0=0, 0*2+1=1? No.
        # Task: W[k*n+i]. n=2.
        # C[0,0] -> k=0, i=0 -> 0
        # C[0,1] -> k=1, i=0 -> 2
        # C[1,0] -> k=0, i=1 -> 1
        # C[1,1] -> k=1, i=1 -> 3
        # So C indices: 00->0, 10->1, 01->2, 11->3.
        
        # M1: (A00+A11)(B00+B11)
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[3][0] += 1
        
        # M2: (A10+A11)B00
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[1][1] += 1; W[3][1] -= 1
        
        # M3: A00(B10-B11)
        U[0][2] += 1
        V[2][2] += 1; V[3][2] -= 1
        W[2][2] += 1; W[3][2] += 1
        
        # M4: A11(B01-B00)
        U[3][3] += 1
        V[1][3] += 1; V[0][3] -= 1
        W[0][3] += 1; W[1][3] += 1
        
        # M5: (A00+A01)B11
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] -= 1; W[2][4] += 1
        
        # M6: (A10-A00)(B00+B01)
        U[2][5] += 1; U[0][5] -= 1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1
        
        # M7: (A01-A11)(B10+B11)
        U[1][6] += 1; U[3][6] -= 1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1
        
        return U, V, W

    def get_schoolbook_221():
        # A(2x2), B(2x1), C(2x1)
        # B indices: 0, 1
        # C indices: k*n+i. n=2. k=0.
        # C[0,0] -> 0. C[1,0] -> 1.
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        # P1: A00 * B00 -> C00
        U[0][0] += 1; V[0][0] += 1; W[0][0] += 1
        # P2: A01 * B01 -> C00
        U[1][1] += 1; V[1][1] += 1; W[0][1] += 1
        # P3: A10 * B00 -> C10
        U[2][2] += 1; V[0][2] += 1; W[1][2] += 1
        # P4: A11 * B01 -> C10
        U[3][3] += 1; V[1][3] += 1; W[1][3] += 1
        
        return U, V, W

    # Assemble 2x4x7
    # Split m=4 into two 2s.
    # Split p=7 into 2,2,2,1.
    # For each m-part (2 of them):
    #   Use Strassen for p-parts 0,1; 2,3; 4,5.
    #   Use Schoolbook for p-part 6.
    
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # We need to map local indices to global indices.
    # Global A: 2x4. Indices i*4+j.
    # Global B: 4x7. Indices j*7+k.
    # Global C: 2x7. Indices k*2+i.
    
    def add_tensor(U_t, V_t, W_t, a_offset, b_offset, c_offset):
        # a_offset: row index of A block (0 or 1)
        # b_offset: col index of A block / row index of B block (0, 2 for m-split)
        # c_offset: col index of B block (0, 2, 4, 6)
        
        u_block = [[0]*len(U_t[0]) for _ in range(len(U_t))]
        v_block = [[0]*len(V_t[0]) for _ in range(len(V_t))]
        w_block = [[0]*len(W_t[0]) for _ in range(len(W_t))]
        
        # Copy and offset
        for i in range(len(U_t)):
            for k in range(len(U_t[0])):
                # A index in block: i (0..3)
                # A global index: a_offset*4 + (b_offset + i%2) + (i//2)*4? 
                # No, A is 2x4.
                # Block A is 2x2.
                # Local A indices: 00, 01, 10, 11.
                # Global A indices:
                # i=0 (00) -> row 0, col b_offset
                # i=1 (01) -> row 0, col b_offset+1
                # i=2 (10) -> row 1, col b_offset
                # i=3 (11) -> row 1, col b_offset+1
                # Wait, my local U indices were 0,1,2,3 corresponding to 00,01,10,11.
                # So:
                if i == 0: ga = 0 * 4 + b_offset
                elif i == 1: ga = 0 * 4 + b_offset + 1
                elif i == 2: ga = 1 * 4 + b_offset
                elif i == 3: ga = 1 * 4 + b_offset + 1
                
                u_block[i][k] = U_t[i][k]
                
                # We need to accumulate into the final U.
                # But we are building the final matrices.
                pass

    # This mapping is getting complex. Let's just build the final matrices directly.
    
    rank = 0
    U_final = []
    V_final = []
    W_final = []
    
    # Pre-allocate? No, we don't know rank until we count.
    # We know rank is 50.
    U_final = [[0]*50 for _ in range(8)] # n*m = 2*4 = 8
    V_final = [[0]*50 for _ in range(28)] # m*p = 4*7 = 28
    W_final = [[0]*50 for _ in range(14)] # p*n = 7*2 = 14
    
    col = 0
    
    S_U, S_V, S_W = get_strassen_222()
    SB_U, SB_V, SB_W = get_schoolbook_221()
    
    def add_strassen(a_row, b_col_start, c_col_start):
        nonlocal col
        for k in range(7):
            for i in range(4):
                if S_U[i][k] != 0:
                    # Map local A index i to global
                    if i == 0: ga = a_row * 4 + b_col_start
                    elif i == 1: ga = a_row * 4 + b_col_start + 1
                    elif i == 2: ga = (a_row + 1) * 4 + b_col_start
                    elif i == 3: ga = (a_row + 1) * 4 + b_col_start + 1
                    U_final[ga][col] += S_U[i][k]
                    
            for j in range(4):
                if S_V[j][k] != 0:
                    # Map local B index j to global
                    # B is 4x7. Block is 2x2.
                    # j=0 (00) -> row b_col_start, col c_col_start
                    # j=1 (01) -> row b_col_start, col c_col_start+1
                    # j=2 (10) -> row b_col_start+1, col c_col_start
                    # j=3 (11) -> row b_col_start+1, col c_col_start+1
                    if j == 0: gb = b_col_start * 7 + c_col_start
                    elif j == 1: gb = b_col_start * 7 + c_col_start + 1
                    elif j == 2: gb = (b_col_start + 1) * 7 + c_col_start
                    elif j == 3: gb = (b_col_start + 1) * 7 + c_col_start + 1
                    V_final[gb][col] += S_V[j][k]
                    
            for c in range(4):
                if S_W[c][k] != 0:
                    # Map local C index c to global
                    # C is 2x7. Block is 2x2.
                    # c=0 (00) -> row a_row, col c_col_start
                    # c=1 (10) -> row a_row+1, col c_col_start
                    # c=2 (01) -> row a_row, col c_col_start+1
                    # c=3 (11) -> row a_row+1, col c_col_start+1
                    # Global C index: k_global * 2 + i_global
                    if c == 0:
                        gc = c_col_start * 2 + a_row
                    elif c == 1:
                        gc = c_col_start * 2 + (a_row + 1)
                    elif c == 2:
                        gc = (c_col_start + 1) * 2 + a_row
                    elif c == 3:
                        gc = (c_col_start + 1) * 2 + (a_row + 1)
                    W_final[gc][col] += S_W[c][k]
        col += 7

    def add_schoolbook(a_row, b_col_start, c_col_start):
        nonlocal col
        for k in range(4):
            for i in range(4):
                if SB_U[i][k] != 0:
                    if i == 0: ga = a_row * 4 + b_col_start
                    elif i == 1: ga = a_row * 4 + b_col_start + 1
                    elif i == 2: ga = (a_row + 1) * 4 + b_col_start
                    elif i == 3: ga = (a_row + 1) * 4 + b_col_start + 1
                    U_final[ga][col] += SB_U[i][k]
                    
            for j in range(2):
                if SB_V[j][k] != 0:
                    # B block is 2x1.
                    # j=0 -> row b_col_start, col c_col_start
                    # j=1 -> row b_col_start+1, col c_col_start
                    if j == 0: gb = b_col_start * 7 + c_col_start
                    elif j == 1: gb = (b_col_start + 1) * 7 + c_col_start
                    V_final[gb][col] += SB_V[j][k]
                    
            for c in range(2):
                if SB_W[c][k] != 0:
                    # C block is 2x1.
                    # c=0 -> row a_row, col c_col_start
                    # c=1 -> row a_row+1, col c_col_start
                    if c == 0: gc = c_col_start * 2 + a_row
                    elif c == 1: gc = c_col_start * 2 + (a_row + 1)
                    W_final[gc][col] += SB_W[c][k]
        col += 4

    # Loop over m-halves (b_col_start = 0, 2)
    for b_start in [0, 2]:
        # p-halves (c_col_start = 0, 2, 4)
        for c_start in [0, 2, 4]:
            add_strassen(0, b_start, c_start)
        # p-tail (c_col_start = 6)
        add_schoolbook(0, b_start, 6)
        
    return {"U": U_final, "V": V_final, "W": W_final}
