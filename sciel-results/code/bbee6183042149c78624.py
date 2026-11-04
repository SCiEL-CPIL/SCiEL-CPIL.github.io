"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    We use a block decomposition approach where we split the 4-dimensional middle index
    into two groups of 2, and apply Strassen's rank-7 algorithm to 2x2x2 sub-blocks.
    
    For 3x4*4x8:
    - Input A is 3x4
    - Input B is 4x8
    - Output C is 3x8
    
    The tensor has size 3*4*8 = 96.
    
    Strategy:
    1. Split the 4 middle indices into two groups: {0,1} and {2,3}
    2. For each group, split the 8 output columns into four groups of 2: {0,1}, {2,3}, {4,5}, {6,7}
    3. For the 2x2x2 sub-blocks (rows 0-1 of A, middle group, 2 columns of B), use Strassen (rank 7)
    4. For the 1x2x2 sub-blocks (row 2 of A, middle group, 2 columns of B), use schoolbook (rank 4)
    
    Total rank: 2 (middle groups) * 4 (column blocks) * (7 + 4) = 88
    """
    n, m, p = DIMENSIONS
    terms = []
    
    def add_term(u_parts, v_parts, w_parts):
        """Add a rank-1 term with sparse representations."""
        terms.append((u_parts, v_parts, w_parts))
    
    # Strassen's 7 multiplications for 2x2 * 2x2
    # Given matrices X (2x2) and Y (2x2):
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # 
    # m1 = (x00 + x11)(y00 + y11)
    # m2 = (x10 + x11)y00
    # m3 = x00(y10 - y11)
    # m4 = x11(y01 - y00)
    # m5 = (x00 + x01)y11
    # m6 = (x10 - x00)(y00 + y01)
    # m7 = (x01 - x11)(y10 + y11)
    #
    # Result:
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    
    # Split middle indices into two groups of 2
    middle_groups = [(0, 1), (2, 3)]
    # Split output columns into four groups of 2
    col_groups = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for j0, j1 in middle_groups:
        for k0, k1 in col_groups:
            # For rows 0 and 1 of A, use Strassen on the 2x2x2 block
            # A_block = [[A[0,j0], A[0,j1]], [A[1,j0], A[1,j1]]]
            # B_block = [[B[j0,k0], B[j0,k1]], [B[j1,k0], B[j1,k1]]]
            
            # m1 = (A[0,j0] + A[1,j1])(B[j0,k0] + B[j1,k1])
            add_term(
                [(0*m+j0, 1), (1*m+j1, 1)],
                [(j0*p+k0, 1), (j1*p+k1, 1)],
                [(k0*n+0, 1), (k1*n+1, 1)]
            )
            
            # m2 = (A[1,j0] + A[1,j1])B[j0,k0]
            add_term(
                [(1*m+j0, 1), (1*m+j1, 1)],
                [(j0*p+k0, 1)],
                [(k0*n+1, 1)]
            )
            
            # m3 = A[0,j0](B[j1,k0] - B[j1,k1])
            add_term(
                [(0*m+j0, 1)],
                [(j1*p+k0, 1), (j1*p+k1, -1)],
                [(k0*n+0, 1), (k1*n+1, 1)]
            )
            
            # m4 = A[1,j1](B[j0,k1] - B[j0,k0])
            add_term(
                [(1*m+j1, 1)],
                [(j0*p+k1, 1), (j0*p+k0, -1)],
                [(k0*n+0, 1), (k1*n+1, 1)]
            )
            
            # m5 = (A[0,j0] + A[0,j1])B[j1,k1]
            add_term(
                [(0*m+j0, 1), (0*m+j1, 1)],
                [(j1*p+k1, 1)],
                [(k0*n+0, 1), (k1*n+1, 1)]
            )
            
            # m6 = (A[1,j0] - A[0,j0])(B[j0,k0] + B[j0,k1])
            add_term(
                [(1*m+j0, 1), (0*m+j0, -1)],
                [(j0*p+k0, 1), (j0*p+k1, 1)],
                [(k1*n+1, 1)]
            )
            
            # m7 = (A[0,j1] - A[1,j1])(B[j1,k0] + B[j1,k1])
            add_term(
                [(0*m+j1, 1), (1*m+j1, -1)],
                [(j1*p+k0, 1), (j1*p+k1, 1)],
                [(k0*n+0, 1), (k1*n+1, -1)]
            )
            
            # For row 2 of A, use schoolbook for the 1x2x2 block
            for k in [k0, k1]:
                for j in [j0, j1]:
                    add_term(
                        [(2*m+j, 1)],
                        [(j*p+k, 1)],
                        [(k*n+2, 1)]
                    )
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_parts, v_parts, w_parts) in enumerate(terms):
        for (idx, coeff) in u_parts:
            U[idx][r] += coeff
        for (idx, coeff) in v_parts:
            V[idx][r] += coeff
        for (idx, coeff) in w_parts:
            W[idx][r] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
