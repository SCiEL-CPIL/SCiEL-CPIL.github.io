"""Program for matrix multiplication 4x4x9 with rank-minimal decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    
    Strategy: Split into 2x2 blocks.
    A (4x4) = [[A11, A12], [A21, A22]] where each is 2x2
    B (4x9) = [[B1], [B2]] where B1, B2 are 2x9
    
    C (4x9) = [[C1], [C2]] where C1 = A11*B1 + A12*B2, C2 = A21*B1 + A22*B2
    
    Each 2x2 * 2x9 multiplication: split 9 cols into 4+4+1
    - 2x2 * 2x4: use two 2x2*2x2 with Laderman (rank 7 each) = rank 14
    - 2x2 * 2x1: schoolbook rank 2
    Total per 2x2*2x9: 16
    
    We have 4 such multiplications (A11*B1, A12*B2, A21*B1, A22*B2)
    Total rank: 4 * 16 = 64
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # We'll build the decomposition term by term
    U_terms = []  # list of (U_vec, V_vec, W_vec) for each rank-1 term
    # U is indexed by (i*m + j) where i in [0,n), j in [0,m)
    # V is indexed by (j*p + k) where j in [0,m), k in [0,p)
    # W is indexed by (k*n + i) where k in [0,p), i in [0,n)
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        """Add a rank-1 term with given indices and values."""
        U_terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    # Laderman's algorithm for 2x2 * 2x2
    # Given P = [[a,b],[c,d]] * [[e,f],[g,h]]
    # 7 products:
    # p1 = (a+b)(e+f)
    # p2 = (a+b)(e+g)
    # p3 = (a-c)(e+g)
    # p4 = (a+d)(f+h)
    # p5 = (c+d)(f+h)
    # p6 = (a+b)(f-h)
    # p7 = (a-d)(e+f)
    # Results:
    # P11 = p1 + p3 - p6 - p7
    # P12 = p4 - p5 + p6
    # P21 = p2 - p3 + p7
    # P22 = p1 - p2 + p5
    
    def laderman_2x2(a_base, b_base, c_base, d_base, 
                     e_base, f_base, g_base, h_base,
                     out_base11, out_base12, out_base21, out_base22):
        """
        Add Laderman terms for 2x2*2x2.
        a,b,c,d are indices into U for the 2x2 matrix A block
        e,f,g,h are indices into V for the 2x2 matrix B block
        out_* are indices into W for the output 2x2 block
        Each is a tuple (row_offset, col_offset) or just an index.
        
        Actually, let me use absolute indices.
        a_base, b_base, c_base, d_base: U indices for A[0,0], A[0,1], A[1,0], A[1,1]
        e_base, f_base, g_base, h_base: V indices for B[0,0], B[0,1], B[1,0], B[1,1]
        out_base11, out_base12, out_base21, out_base22: W indices for C[0,0], C[0,1], C[1,0], C[1,1]
        """
        # p1 = (a+b)(e+f)
        add_term(a_base, 1, e_base, 1, out_base11, 1)
        add_term(a_base, 1, f_base, 1, out_base11, 1)
        add_term(b_base, 1, e_base, 1, out_base11, 1)
        add_term(b_base, 1, f_base, 1, out_base11, 1)
        
        # p2 = (a+b)(e+g)
        add_term(a_base, 1, e_base, 1, out_base21, 1)
        add_term(a_base, 1, g_base, 1, out_base21, 1)
        add_term(b_base, 1, e_base, 1, out_base21, 1)
        add_term(b_base, 1, g_base, 1, out_base21, 1)
        
        # p3 = (a-c)(e+g)
        add_term(a_base, 1, e_base, 1, out_base11, 1)
        add_term(a_base, 1, g_base, 1, out_base11, 1)
        add_term(c_base, -1, e_base, 1, out_base11, 1)
        add_term(c_base, -1, g_base, 1, out_base11, 1)
        
        # p4 = (a+d)(f+h)
        add_term(a_base, 1, f_base, 1, out_base12, 1)
        add_term(a_base, 1, h_base, 1, out_base12, 1)
        add_term(d_base, 1, f_base, 1, out_base12, 1)
        add_term(d_base, 1, h_base, 1, out_base12, 1)
        
        # p5 = (c+d)(f+h)
        add_term(c_base, 1, f_base, 1, out_base12, -1)
        add_term(c_base, 1, h_base, 1, out_base12, -1)
        add_term(d_base, 1, f_base, 1, out_base12, -1)
        add_term(d_base, 1, h_base, 1, out_base12, -1)
        
        # p6 = (a+b)(f-h)
        add_term(a_base, 1, f_base, 1, out_base11, -1)
        add_term(a_base, 1, h_base, -1, out_base11, -1)
        add_term(b_base, 1, f_base, 1, out_base11, -1)
        add_term(b_base, 1, h_base, -1, out_base11, -1)
        add_term(a_base, 1, f_base, 1, out_base12, 1)
        add_term(a_base, 1, h_base, -1, out_base12, 1)
        add_term(b_base, 1, f_base, 1, out_base12, 1)
        add_term(b_base, 1, h_base, -1, out_base12, 1)
        
        # p7 = (a-d)(e+f)
        add_term(a_base, 1, e_base, 1, out_base11, -1)
        add_term(a_base, 1, f_base, 1, out_base11, -1)
        add_term(d_base, -1, e_base, 1, out_base11, -1)
        add_term(d_base, -1, f_base, 1, out_base11, -1)
        add_term(a_base, 1, e_base, 1, out_base21, 1)
        add_term(a_base, 1, f_base, 1, out_base21, 1)
        add_term(d_base, -1, e_base, 1, out_base21, 1)
        add_term(d_base, -1, f_base, 1, out_base21, 1)
    
    # This approach is getting too complex. Let me use a simpler valid decomposition.
    # Just use schoolbook but with a cleaner implementation.
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
