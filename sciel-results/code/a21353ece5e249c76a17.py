"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 96
    # Decomposition strategy:
    # Split n=4 into two blocks of size 2.
    # For each n-block, split m=5 into 2, 2, 1.
    # For each m-block of size 2, split p=5 into 3, 2.
    # Use Strassen (rank 7) for 2x2x2.
    # Use Schoolbook (rank 12) for 2x2x3.
    # Use Schoolbook (rank 10) for 2x1x5.
    
    rank_total = 0
    U = []
    V = []
    W = []
    
    def add_block(u_offset, v_offset, w_offset, u_rows, v_rows, w_rows, block_rank, get_block_matrices):
        nonlocal rank_total
        # get_block_matrices returns U_b, V_b, W_b of shapes (u_rows, block_rank), etc.
        U_b, V_b, W_b = get_block_matrices()
        for r in range(block_rank):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for i in range(u_rows):
                u_col[u_offset + i] = U_b[i][r]
            for j in range(v_rows):
                v_col[v_offset + j] = V_b[j][r]
            for k in range(w_rows):
                w_col[w_offset + k] = W_b[k][r]
                
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
        rank_total += block_rank

    def get_strassen_222():
        # Returns U (4x7), V (4x7), W (4x7) for 2x2x2
        # U indices: i*2+j
        # V indices: j*2+k
        # W indices: k*2+i
        U_b = [[0]*7 for _ in range(4)]
        V_b = [[0]*7 for _ in range(4)]
        W_b = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U_b[0][0] = 1; U_b[3][0] = 1
        V_b[0][0] = 1; V_b[3][0] = 1
        W_b[0][0] = 1; W_b[3][0] = 1 # C00, C11 parts? No, W is for C.
        # W_b[k*2+i] corresponds to C[i,k].
        # M1 contributes to C00 and C11.
        # C00 = M1... C11 = M1...
        # So W_b[0*2+0]=1 (C00), W_b[1*2+1]=1 (C11).
        W_b[0][0] = 1
        W_b[3][0] = 1
        
        # M2 = (A10+A11)B00
        U_b[2][1] = 1; U_b[3][1] = 1
        V_b[0][1] = 1
        W_b[2][1] = 1 # C10
        W_b[3][1] = -1 # C11
        
        # M3 = A00(B10-B11)
        U_b[0][2] = 1
        V_b[2][2] = 1; V_b[3][2] = -1
        W_b[1][2] = 1 # C01
        W_b[3][2] = 1 # C11
        
        # M4 = A11(B01+B11)
        U_b[3][3] = 1
        V_b[1][3] = 1; V_b[3][3] = 1
        W_b[0][3] = 1 # C00
        W_b[2][3] = 1 # C10
        
        # M5 = (A00+A01)B11
        U_b[0][4] = 1; U_b[1][4] = 1
        V_b[3][4] = 1
        W_b[0][4] = -1 # C00
        W_b[1][4] = 1 # C01
        
        # M6 = (A10-A00)(B00+B01)
        U_b[2][5] = 1; U_b[0][5] = -1
        V_b[0][5] = 1; V_b[1][5] = 1
        W_b[2][5] = 1 # C10
        W_b[3][5] = 1 # C11
        
        # M7 = (A01-A11)(B10+B11)
        U_b[1][6] = 1; U_b[3][6] = -1
        V_b[2][6] = 1; V_b[3][6] = 1
        W_b[0][6] = 1 # C00
        W_b[1][6] = 1 # C01
        
        return U_b, V_b, W_b

    def get_schoolbook_223():
        # Returns U (4x12), V (6x12), W (6x12) for 2x2x3
        # U indices: i*2+j
        # V indices: j*3+k
        # W indices: k*2+i
        U_b = [[0]*12 for _ in range(4)]
        V_b = [[0]*12 for _ in range(6)]
        W_b = [[0]*12 for _ in range(6)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U_b[i*2+j][col] = 1
                    V_b[j*3+k][col] = 1
                    W_b[k*2+i][col] = 1
                    col += 1
        return U_b, V_b, W_b

    def get_schoolbook_215():
        # Returns U (2x10), V (5x10), W (10x10) for 2x1x5
        # U indices: i*1+j -> i
        # V indices: j*5+k -> k
        # W indices: k*2+i
        U_b = [[0]*10 for _ in range(2)]
        V_b = [[0]*10 for _ in range(5)]
        W_b = [[0]*10 for _ in range(10)]
        col = 0
        for i in range(2):
            for k in range(5):
                U_b[i][col] = 1
                V_b[k][col] = 1
                W_b[k*2+i][col] = 1
                col += 1
        return U_b, V_b, W_b

    # Process I1 (i=0,1)
    # J1 (j=0,1)
    # K1 (k=0,1,2) -> 2x2x3
    add_block(0*5+0, 0*5+0, 0*4+0, 2, 2, 3, 12, get_schoolbook_223)
    # K2 (k=3,4) -> 2x2x2
    add_block(0*5+0, 0*5+3, 3*4+0, 2, 2, 2, 7, get_strassen_222)
    
    # J2 (j=2,3)
    # K1 (k=0,1,2) -> 2x2x3
    add_block(0*5+2, 2*5+0, 0*4+0, 2, 2, 3, 12, get_schoolbook_223)
    # K2 (k=3,4) -> 2x2x2
    add_block(0*5+2, 2*5+3, 3*4+0, 2, 2, 2, 7, get_strassen_222)
    
    # J3 (j=4)
    # K (k=0..4) -> 2x1x5
    add_block(0*5+4, 4*5+0, 0*4+0, 2, 1, 5, 10, get_schoolbook_215)

    # Process I2 (i=2,3)
    # J1 (j=0,1)
    # K1 (k=0,1,2) -> 2x2x3
    add_block(2*5+0, 0*5+0, 0*4+2, 2, 2, 3, 12, get_schoolbook_223)
    # K2 (k=3,4) -> 2x2x2
    add_block(2*5+0, 0*5+3, 3*4+2, 2, 2, 2, 7, get_strassen_222)
    
    # J2 (j=2,3)
    # K1 (k=0,1,2) -> 2x2x3
    add_block(2*5+2, 2*5+0, 0*4+2, 2, 2, 3, 12, get_schoolbook_223)
    # K2 (k=3,4) -> 2x2x2
    add_block(2*5+2, 2*5+3, 3*4+2, 2, 2, 2, 7, get_strassen_222)
    
    # J3 (j=4)
    # K (k=0..4) -> 2x1x5
    add_block(2*5+4, 4*5+0, 0*4+2, 2, 1, 5, 10, get_schoolbook_215)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
