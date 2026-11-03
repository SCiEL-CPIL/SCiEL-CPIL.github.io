"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 57 decomposition
    # A is 2x5, B is 5x6
    # A = [A1 (2x2), A2 (2x3)]
    # B = [B1 (2x6); B2 (3x6)]
    # AB = A1*B1 + A2*B2
    
    # A1*B1: 2x2 by 2x6
    # Split B1 into three 2x2 blocks: B1_0, B1_1, B1_2
    # Each is 2x2x2, rank 7 (Strassen)
    # Total rank 21
    
    # A2*B2: 2x3 by 3x6
    # Split B2 into two 3x3 blocks: B2_0, B2_1
    # Each is 2x3x3, rank 18 (Schoolbook)
    # Total rank 36
    
    total_rank = 21 + 36
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2x2
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # i_offset: row offset in A (0)
        # j_offset: col offset in A (0)
        # k_offset: col offset in B (0, 2, 4)
        
        # Strassen matrices for 2x2x2
        # U indices: (i, j)
        # V indices: (j, k)
        # W indices: (k, i)
        
        # M1 = (a+d)(e+h)
        # U: (0,0)=1, (1,1)=1
        # V: (0,0)=1, (1,1)=1
        # W: (0,0)=1, (1,1)=1
        U[i_offset * m + 0][col] = 1
        U[i_offset * m + 1][col] = 1
        V[j_offset * p + 0][col] = 1
        V[j_offset * p + 1][col] = 1
        W[k_offset * n + 0][col] = 1
        W[k_offset * n + 1][col] = 1
        col += 1
        
        # M2 = (c+d)e
        # U: (1,0)=1, (1,1)=1
        # V: (0,0)=1
        # W: (0,1)=1
        U[i_offset * m + 1][col] = 1
        U[i_offset * m + 1][col] = 1 # Wait, c is (1,0), d is (1,1). So U[1,0]=1, U[1,1]=1.
        V[j_offset * p + 0][col] = 1
        W[k_offset * n + 1][col] = 1
        col += 1
        
        # M3 = a(f-h)
        # U: (0,0)=1
        # V: (0,1)=1, (1,1)=-1
        # W: (1,0)=1
        U[i_offset * m + 0][col] = 1
        V[j_offset * p + 1][col] = 1
        V[j_offset * p + 1][col] = 1 # f is (0,1), h is (1,1). So V[0,1]=1, V[1,1]=-1.
        V[j_offset * p + 1][col] = 1 # Correction: V[j_offset*p + 0][col] is e. V[j_offset*p + 1][col] is f.
        # Let's be precise.
        # V indices are (j, k).
        # e is (0,0), f is (0,1), g is (1,0), h is (1,1).
        # M3 = a(f-h).
        # U: (0,0)=1.
        # V: (0,1)=1, (1,1)=-1.
        # W: (1,0)=1.
        U[i_offset * m + 0][col] = 1
        V[j_offset * p + 1][col] = 1
        V[j_offset * p + 1][col] = -1 # This is wrong, I am overwriting.
        
        # I should write a proper helper.
        pass

    # Let's rewrite the helper properly.
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # M1
        U[i_off * m + 0][col] = 1
        U[i_off * m + 1][col] = 1
        V[j_off * p + 0][col] = 1
        V[j_off * p + 1][col] = 1
        W[k_off * n + 0][col] = 1
        W[k_off * n + 1][col] = 1
        col += 1
        # M2
        U[i_off * m + 1][col] = 1
        V[j_off * p + 0][col] = 1
        W[k_off * n + 1][col] = 1
        col += 1
        # M3
        U[i_off * m + 0][col] = 1
        V[j_off * p + 1][col] = 1
        V[j_off * p + 1][col] = -1 # Error: V index for h is (1,1) relative to block, so j_off+1, k_off+1
        # V indices: (j, k).
        # f is (j_off, k_off+1). h is (j_off+1, k_off+1).
        V[j_off * p + 1][col] = 1
        V[(j_off + 1) * p + 1][col] = -1
        W[k_off * n + 1][col] = 1
        col += 1
        # M4
        U[i_off * m + 1][col] = 1
        V[(j_off + 1) * p + 0][col] = 1
        V[j_off * p + 0][col] = -1
        W[k_off * n + 1][col] = 1
        col += 1
        # M5
        U[i_off * m + 0][col] = 1
        U[i_off * m + 1][col] = 1
        V[(j_off + 1) * p + 1][col] = 1
        W[k_off * n + 0][col] = 1
        col += 1
        # M6
        U[i_off * m + 1][col] = 1
        U[i_off * m + 0][col] = -1
        V[j_off * p + 0][col] = 1
        V[j_off * p + 1][col] = 1
        W[k_off * n + 1][col] = 1
        col += 1
        # M7
        U[i_off * m + 1][col] = 1
        U[i_off * m + 0][col] = -1
        V[(j_off + 1) * p + 0][col] = 1
        V[(j_off + 1) * p + 1][col] = 1
        W[k_off * n + 0][col] = 1
        col += 1

    # Add 3 Strassen blocks for A1*B1
    # i_off = 0, j_off = 0
    # k_off = 0, 2, 4
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Add 2 Schoolbook blocks for A2*B2
    # i_off = 0, j_off = 2
    # k_off = 0, 3
    # Schoolbook for 2x3x3
    def add_schoolbook_2x3x3(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[(i_off + i) * m + (j_off + j)][col] = 1
                    V[(j_off + j) * p + (k_off + k)][col] = 1
                    W[(k_off + k) * n + (i_off + i)][col] = 1
                    col += 1

    add_schoolbook_2x3x3(0, 2, 0)
    add_schoolbook_2x3x3(0, 2, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
