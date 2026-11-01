"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    # Hybrid decomposition using Strassen for 2x2x2 blocks and schoolbook for remainder.
    # Matrix A (2x3), B (3x4), C (2x4).
    # Split inner dimension m=3 into m1=2 and m2=1.
    # A = [A1 (2x2) | A2 (2x1)]
    # B = [B1 (2x4); B2 (1x4)]
    # C = A1*B1 + A2*B2
    # 
    # Block 1: A1 (2x2) * B1 (2x4).
    # B1 is 2x4, so we can split it into two 2x2 blocks: B1a and B1b.
    # C1 = A1*B1a + A1*B1b.
    # Each A1*B1x is a 2x2x2 multiplication, which can be done with rank 7 (Strassen).
    # Total rank for Block 1 = 7 + 7 = 14.
    #
    # Block 2: A2 (2x1) * B2 (1x4).
    # This is a 2x1x4 multiplication.
    # C2[i,k] = A2[i] * B2[k].
    # This is a rank 2*1*4 = 8 decomposition (schoolbook).
    #
    # Total rank = 14 + 8 = 22.
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    rank = 22
    
    U = [[0.0] * rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 3*4 = 12 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 4*2 = 8 rows
    
    col = 0
    
    # --- Block 1: A1 (2x2) * B1 (2x4) using Strassen on two 2x2 blocks ---
    # A1 indices: i in {0,1}, j in {0,1} -> row index i*m + j = i*3 + j
    # B1 indices: j in {0,1}, k in {0,1,2,3} -> row index j*p + k = j*4 + k
    # C1 indices: i in {0,1}, k in {0,1,2,3} -> col index k*n + i = k*2 + i
    
    # We split B1 into B1a (cols 0,1) and B1b (cols 2,3).
    # For each 2x2 block of B1, we use Strassen.
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_block(a_rows, b_col_base, w_base, start_col):
        """
        Decompose 2x2x2 block.
        a_rows: list of 2 row indices in U for the 2x2 A block.
        b_col_base: base column index in V for the 2x2 B block (cols 0,1 relative).
        w_base: base column index in W for the 2x2 C block (cols 0,1 relative).
        start_col: starting column in U,V,W for this Strassen block.
        Returns next_col.
        """
        # a_rows[i] is the row index in U for A[i]
        # b_col_base is the row index in V for B[j,0] (j=0,1). So B[j,1] is b_col_base + 1.
        # w_base is the col index in W for C[0,0] (i=0, k=0). 
        # W index is k*n + i. So C[i,k] is at row k*n + i.
        # For a 2x2 block of C (i in {0,1}, k in {0,1} relative to block start),
        # the W row index is (k_block_start + k)*n + (i_block_start + i).
        
        # Let's map the local indices to global.
        # A1 is at rows: 
        # A[0,0] -> row 0*3+0 = 0
        # A[0,1] -> row 0*3+1 = 1
        # A[1,0] -> row 1*3+0 = 3
        # A[1,1] -> row 1*3+1 = 4
        # So a_rows = [0, 3] for rows 0,1 of A1? No, A1 is 2x2.
        # A1[0,0] is A[0,0] (row 0 in U).
        # A1[0,1] is A[0,1] (row 1 in U).
        # A1[1,0] is A[1,0] (row 3 in U).
        # A1[1,1] is A[1,1] (row 4 in U).
        # So for Strassen, we need to access A[0,0], A[0,1], A[1,0], A[1,1].
        # In U, these are at rows 0, 1, 3, 4.
        
        # B1a (cols 0,1 of B1) corresponds to B[0,0], B[0,1], B[1,0], B[1,1].
        # In V, these are at rows:
        # B[0,0] -> 0*4+0 = 0
        # B[0,1] -> 0*4+1 = 1
        # B[1,0] -> 1*4+0 = 4
        # B[1,1] -> 1*4+1 = 5
        # So b_rows = [0, 4] for rows 0,1 of B block? No, B is 2x2 here.
        # B[0,0], B[0,1] are row 0,1 of V? No, V index is j*p+k.
        # B[j,k] is at V row j*p+k.
        # So B[0,0] is row 0, B[0,1] is row 1.
        # B[1,0] is row 4, B[1,1] is row 5.
        
        # C1a (cols 0,1 of C1) corresponds to C[0,0], C[0,1], C[1,0], C[1,1].
        # In W, index is k*n+i.
        # C[0,0] -> 0*2+0 = 0
        # C[0,1] -> 1*2+0 = 2
        # C[1,0] -> 0*2+1 = 1
        # C[1,1] -> 1*2+1 = 3
        # So W rows for C[0,0], C[0,1], C[1,0], C[1,1] are 0, 2, 1, 3.
        
        # Let's hardcode the Strassen coefficients for this specific geometry.
        # U rows: A00=0, A01=1, A10=3, A11=4
        # V rows: B00=0, B01=1, B10=4, B11=5 (for B1a)
        #          B00=2, B01=3, B10=6, B11=7 (for B1b)
        # W rows: C00=0, C01=2, C10=1, C11=3 (for C1a)
        #          C00=4, C01=6, C10=5, C11=7 (for C1b)
        
        # We will create a helper to add a Strassen block.
        # Parameters:
        # u_rows: [u_A00, u_A01, u_A10, u_A11]
        # v_rows: [v_B00, v_B01, v_B10, v_B11]
        # w_rows: [w_C00, w_C01, w_C10, w_C11]
        # start_col: starting column index
        
        def add_strassen(u_r, v_r, w_r, c):
            # M1 = (A00+A11)(B00+B11)
            U[u_r[0]][c] += 1; V[v_r[0]][c] += 1; W[w_r[0]][c] += 1
            U[u_r[3]][c] += 1; V[v_r[3]][c] += 1; W[w_r[0]][c] += 1
            # M1 also contributes to C11 with +1
            W[w_r[3]][c] += 1
            c += 1
            
            # M2 = (A10+A11)B00
            U[u_r[2]][c] += 1; V[v_r[0]][c] += 1; W[w_r[2]][c] += 1
            U[u_r[3]][c] += 1; V[v_r[0]][c] += 1; W[w_r[2]][c] += 1
            # M2 contributes to C11 with -1
            W[w_r[3]][c] -= 1
            c += 1
            
            # M3 = A00(B10-B11)
            U[u_r[0]][c] += 1; V[v_r[2]][c] += 1; W[w_r[1]][c] += 1
            U[u_r[0]][c] += 1; V[v_r[3]][c] -= 1; W[w_r[1]][c] += 1
            # M3 contributes to C11 with +1
            W[w_r[3]][c] += 1
            c += 1
            
            # M4 = A11(B01-B00)
            U[u_r[3]][c] += 1; V[v_r[1]][c] += 1; W[w_r[0]][c] += 1
            U[u_r[3]][c] += 1; V[v_r[0]][c] -= 1; W[w_r[0]][c] += 1
            # M4 contributes to C11 with +1
            W[w_r[3]][c] += 1
            c += 1
            
            # M5 = (A00+A01)B11
            U[u_r[0]][c] += 1; V[v_r[3]][c] += 1; W[w_r[0]][c] -= 1
            U[u_r[1]][c] += 1; V[v_r[3]][c] += 1; W[w_r[0]][c] -= 1
            # M5 contributes to C01 with +1
            W[w_r[1]][c] += 1
            c += 1
            
            # M6 = (A10-A00)(B00+B01)
            U[u_r[2]][c] += 1; V[v_r[0]][c] += 1; W[w_r[3]][c] += 1
            U[u_r[2]][c] += 1; V[v_r[1]][c] += 1; W[w_r[3]][c] += 1
            U[u_r[0]][c] -= 1; V[v_r[0]][c] += 1; W[w_r[3]][c] += 1
            U[u_r[0]][c] -= 1; V[v_r[1]][c] += 1; W[w_r[3]][c] += 1
            c += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[u_r[1]][c] += 1; V[v_r[2]][c] += 1; W[w_r[0]][c] += 1
            U[u_r[1]][c] += 1; V[v_r[3]][c] += 1; W[w_r[0]][c] += 1
            U[u_r[3]][c] -= 1; V[v_r[2]][c] += 1; W[w_r[0]][c] += 1
            U[u_r[3]][c] -= 1; V[v_r[3]][c] += 1; W[w_r[0]][c] += 1
            c += 1
            
            return c
        
        # Block 1a: A1 * B1a (B cols 0,1)
        u_r = [0, 1, 3, 4]
        v_r = [0, 1, 4, 5]
        w_r = [0, 2, 1, 3]
        col = add_strassen(u_r, v_r, w_r, col)
        
        # Block 1b: A1 * B1b (B cols 2,3)
        # B1b corresponds to B[0,2], B[0,3], B[1,2], B[1,3]
        # V rows:
        # B[0,2] -> 0*4+2 = 2
        # B[0,3] -> 0*4+3 = 3
        # B[1,2] -> 1*4+2 = 6
        # B[1,3] -> 1*4+3 = 7
        # C1b corresponds to C[0,2], C[0,3], C[1,2], C[1,3]
        # W rows:
        # C[0,2] -> 2*2+0 = 4
        # C[0,3] -> 3*2+0 = 6
        # C[1,2] -> 2*2+1 = 5
        # C[1,3] -> 3*2+1 = 7
        u_r = [0, 1, 3, 4]
        v_r = [2, 3, 6, 7]
        w_r = [4, 6, 5, 7]
        col = add_strassen(u_r, v_r, w_r, col)
        
        # --- Block 2: A2 (2x1) * B2 (1x4) ---
        # A2 is A[0,2] (row 2) and A[1,2] (row 5).
        # B2 is B[2,0..3] (rows 8,9,10,11).
        # C2[i,k] = A2[i] * B2[k].
        # W row for C[i,k] is k*n + i.
        
        for i in range(2):
            for k in range(4):
                U[i*3 + 2][col] = 1
                V[2*4 + k][col] = 1
                W[k*2 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
