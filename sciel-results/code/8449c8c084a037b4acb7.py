"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank 56 decomposition using 4 Strassen blocks for 2x2x4 subproblems.
    # Structure:
    # A is 2x4, B is 4x8, C is 2x8.
    # Split A into A1 (cols 0-1) and A2 (cols 2-3).
    # Split B into B1 (rows 0-1) and B2 (rows 2-3).
    # C = A1 * B1 + A2 * B2.
    # A1 is 2x2, B1 is 2x8.
    # Split B1 into B1a (cols 0-3) and B1b (cols 4-7).
    # A1 * B1a is 2x2x4. Use Rank 14 (2 Strassen).
    # A1 * B1b is 2x2x4. Use Rank 14 (2 Strassen).
    # Similarly for A2 * B2.
    # Total 4 * 14 = 56.
    
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block for 2x2x2
    def add_strassen(offset_u, offset_v, offset_w, col_start):
        # A block at U[0:2, 0:2], B block at V[0:2, 0:2], W block at W[0:2, 0:2]
        # Strassen 7 terms:
        # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-g), M4=d(g-h), M5=(a+b)h, M6=(c+d)(g-f), M7=(a-c)(e+g)
        # U indices: a=0, b=1, c=2, d=3 (relative to offset_u)
        # V indices: e=0, f=1, g=2, h=3 (relative to offset_v)
        # W indices: c11=0, c12=1, c21=2, c22=3 (relative to offset_w)
        
        # M1 = (a+d)(e+h)
        U[offset_u + 0][col_start] = 1
        U[offset_u + 3][col_start] = 1
        V[offset_v + 0][col_start] = 1
        V[offset_v + 3][col_start] = 1
        W[offset_w + 0][col_start] = 1
        W[offset_w + 1][col_start] = 1
        W[offset_w + 2][col_start] = 1
        W[offset_w + 3][col_start] = 1
        col_start += 1
        
        # M2 = (c+d)e
        U[offset_u + 2][col_start] = 1
        U[offset_u + 3][col_start] = 1
        V[offset_v + 0][col_start] = 1
        W[offset_w + 1][col_start] = 1
        W[offset_w + 3][col_start] = 1
        col_start += 1
        
        # M3 = a(f-g)
        U[offset_u + 0][col_start] = 1
        V[offset_v + 1][col_start] = 1
        V[offset_v + 2][col_start] = -1
        W[offset_w + 1][col_start] = 1
        W[offset_w + 3][col_start] = -1
        col_start += 1
        
        # M4 = d(g-h)
        U[offset_u + 3][col_start] = 1
        V[offset_v + 2][col_start] = 1
        V[offset_v + 3][col_start] = -1
        W[offset_w + 2][col_start] = 1
        W[offset_w + 3][col_start] = -1
        col_start += 1
        
        # M5 = (a+b)h
        U[offset_u + 0][col_start] = 1
        U[offset_u + 1][col_start] = 1
        V[offset_v + 3][col_start] = 1
        W[offset_w + 1][col_start] = 1
        W[offset_w + 3][col_start] = 1
        col_start += 1
        
        # M6 = (c+d)(g-f)
        U[offset_u + 2][col_start] = 1
        U[offset_u + 3][col_start] = 1
        V[offset_v + 2][col_start] = 1
        V[offset_v + 1][col_start] = -1
        W[offset_w + 2][col_start] = 1
        W[offset_w + 3][col_start] = -1
        col_start += 1
        
        # M7 = (a-c)(e+g)
        U[offset_u + 0][col_start] = 1
        U[offset_u + 2][col_start] = -1
        V[offset_v + 0][col_start] = 1
        V[offset_v + 2][col_start] = 1
        W[offset_w + 0][col_start] = 1
        W[offset_w + 2][col_start] = -1
        col_start += 1
        
        return col_start

    # Block 1: A1 (rows 0-1, cols 0-1) * B1a (rows 0-1, cols 0-3)
    # A1 indices in U: 0,1,2,3
    # B1a indices in V: 0,1,2,3 (rows 0-1, cols 0-1) and 4,5,6,7 (rows 0-1, cols 2-3)
    # Wait, V index is j*p + k.
    # B1a is rows 0-1, cols 0-3.
    # Sub-block 1: rows 0-1, cols 0-1. V indices: 0,1,2,3.
    # Sub-block 2: rows 0-1, cols 2-3. V indices: 0*8+2=2, 0*8+3=3, 1*8+2=10, 1*8+3=11.
    
    # W indices for C1a (rows 0-1, cols 0-3).
    # W index is k*n + i.
    # Sub-block 1: cols 0-1. W indices: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3.
    # Sub-block 2: cols 2-3. W indices: 2*2+0=4, 2*2+1=5, 3*2+0=6, 3*2+1=7.

    col = 0
    # A1 * B1a_sub1 (2x2x2)
    col = add_strassen(0, 0, 0, col)
    # A1 * B1a_sub2 (2x2x2)
    # V offset: 2 (row 0, col 2 start)
    # W offset: 4 (col 2 start)
    col = add_strassen(0, 2, 4, col)
    
    # Block 2: A1 * B1b (rows 0-1, cols 4-7)
    # V offset for cols 4-5: 0*8+4=4, 0*8+5=5, 1*8+4=12, 1*8+5=13. Start 4.
    # V offset for cols 6-7: 0*8+6=6, ... Start 6.
    # W offset for cols 4-5: 4*2+0=8, ... Start 8.
    # W offset for cols 6-7: 6*2+0=12, ... Start 12.
    
    col = add_strassen(0, 4, 8, col)
    col = add_strassen(0, 6, 12, col)
    
    # Block 3: A2 (rows 0-1, cols 2-3) * B2a (rows 2-3, cols 0-3)
    # A2 indices in U: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7.
    # U layout: i*m+j.
    # A2 corresponds to i=0,1 and j=2,3.
    # U indices: 2,3,6,7.
    # We need to map Strassen a,b,c,d to these.
    # a=U[2], b=U[3], c=U[6], d=U[7].
    # This requires custom offsets or a helper that takes explicit indices.
    
    # Let's generalize add_strassen to take explicit index lists.
    
    # I will restart the implementation with a generic helper.
    
    # Clear previous work
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen_generic(U_idx, V_idx, W_idx, col):
        # U_idx: [a, b, c, d]
        # V_idx: [e, f, g, h]
        # W_idx: [c11, c12, c21, c22]
        
        # M1
        U[U_idx[0]][col] = 1
        U[U_idx[3]][col] = 1
        V[V_idx[0]][col] = 1
        V[V_idx[3]][col] = 1
        W[W_idx[0]][col] = 1
        W[W_idx[1]][col] = 1
        W[W_idx[2]][col] = 1
        W[W_idx[3]][col] = 1
        col += 1
        
        # M2
        U[U_idx[2]][col] = 1
        U[U_idx[3]][col] = 1
        V[V_idx[0]][col] = 1
        W[W_idx[1]][col] = 1
        W[W_idx[3]][col] = 1
        col += 1
        
        # M3
        U[U_idx[0]][col] = 1
        V[V_idx[1]][col] = 1
        V[V_idx[2]][col] = -1
        W[W_idx[1]][col] = 1
        W[W_idx[3]][col] = -1
        col += 1
        
        # M4
        U[U_idx[3]][col] = 1
        V[V_idx[2]][col] = 1
        V[V_idx[3]][col] = -1
        W[W_idx[2]][col] = 1
        W[W_idx[3]][col] = -1
        col += 1
        
        # M5
        U[U_idx[0]][col] = 1
        U[U_idx[1]][col] = 1
        V[V_idx[3]][col] = 1
        W[W_idx[1]][col] = 1
        W[W_idx[3]][col] = 1
        col += 1
        
        # M6
        U[U_idx[2]][col] = 1
        U[U_idx[3]][col] = 1
        V[V_idx[2]][col] = 1
        V[V_idx[1]][col] = -1
        W[W_idx[2]][col] = 1
        W[W_idx[3]][col] = -1
        col += 1
        
        # M7
        U[U_idx[0]][col] = 1
        U[U_idx[2]][col] = -1
        V[V_idx[0]][col] = 1
        V[V_idx[2]][col] = 1
        W[W_idx[0]][col] = 1
        W[W_idx[2]][col] = -1
        col += 1
        
        return col

    # Define indices
    # A1: i in 0,1, j in 0,1. U: 0,1,2,3
    U_A1 = [0, 1, 2, 3]
    # A2: i in 0,1, j in 2,3. U: 2,3,6,7
    U_A2 = [2, 3, 6, 7]
    
    # B1a_sub1: j in 0,1, k in 0,1. V: 0,1,2,3
    V_B1a1 = [0, 1, 2, 3]
    # B1a_sub2: j in 0,1, k in 2,3. V: 2,3,10,11
    V_B1a2 = [2, 3, 10, 11]
    # B1b_sub1: j in 0,1, k in 4,5. V: 4,5,12,13
    V_B1b1 = [4, 5, 12, 13]
    # B1b_sub2: j in 0,1, k in 6,7. V: 6,7,14,15
    V_B1b2 = [6, 7, 14, 15]
    
    # B2a_sub1: j in 2,3, k in 0,1. V: 16,17,18,19
    V_B2a1 = [16, 17, 18, 19]
    # B2a_sub2: j in 2,3, k in 2,3. V: 18,19,20,21
    V_B2a2 = [18, 19, 20, 21]
    # B2b_sub1: j in 2,3, k in 4,5. V: 20,21,22,23
    V_B2b1 = [20, 21, 22, 23]
    # B2b_sub2: j in 2,3, k in 6,7. V: 22,23,24,25
    V_B2b2 = [22, 23, 24, 25]
    
    # W indices for C
    # C1a1: k in 0,1, i in 0,1. W: 0,1,2,3
    W_C1a1 = [0, 1, 2, 3]
    # C1a2: k in 2,3, i in 0,1. W: 4,5,6,7
    W_C1a2 = [4, 5, 6, 7]
    # C1b1: k in 4,5, i in 0,1. W: 8,9,10,11
    W_C1b1 = [8, 9, 10, 11]
    # C1b2: k in 6,7, i in 0,1. W: 12,13,14,15
    W_C1b2 = [12, 13, 14, 15]
    
    # C2a1: k in 0,1, i in 0,1. W: 0,1,2,3
    W_C2a1 = [0, 1, 2, 3]
    # C2a2: k in 2,3, i in 0,1. W: 4,5,6,7
    W_C2a2 = [4, 5, 6, 7]
    # C2b1: k in 4,5, i in 0,1. W: 8,9,10,11
    W_C2b1 = [8, 9, 10, 11]
    # C2b2: k in 6,7, i in 0,1. W: 12,13,14,15
    W_C2b2 = [12, 13, 14, 15]

    col = 0
    col = add_strassen_generic(U_A1, V_B1a1, W_C1a1, col)
    col = add_strassen_generic(U_A1, V_B1a2, W_C1a2, col)
    col = add_strassen_generic(U_A1, V_B1b1, W_C1b1, col)
    col = add_strassen_generic(U_A1, V_B1b2, W_C1b2, col)
    
    col = add_strassen_generic(U_A2, V_B2a1, W_C2a1, col)
    col = add_strassen_generic(U_A2, V_B2a2, W_C2a2, col)
    col = add_strassen_generic(U_A2, V_B2b1, W_C2b1, col)
    col = add_strassen_generic(U_A2, V_B2b2, W_C2b2, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
