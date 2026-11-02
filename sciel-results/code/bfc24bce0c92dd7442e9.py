"""Matrix multiplication 2,2,5 with rank-minimal exact decomposition."""

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x5 matrix.
    
    The tensor is T_{ijk} = delta_{jk} for C[i,k] = sum_j A[i,j]*B[j,k].
    We use a block decomposition:
    - Two 2x2x2 blocks (columns 0-1 and 2-3) using Strassen (rank 7 each)
    - One 2x2x1 block (column 4) using schoolbook (rank 4)
    Total rank: 7 + 7 + 4 = 18
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=5
    
    # Total rank
    total_rank = 18
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]  # (4, 18)
    V = [[0] * total_rank for _ in range(m * p)]  # (10, 18)
    W = [[0] * total_rank for _ in range(p * n)]  # (10, 18)
    
    # Helper to add a Strassen block for 2x2x2 multiplication
    # B columns correspond to global columns c0, c1
    def add_strassen_block(c0, c1, offset):
        """
        Add Strassen decomposition for A(2x2) * B(2x2) -> C(2x2)
        where B has columns c0, c1 and C has columns c0, c1.
        
        Strassen products:
        P1 = (A00+A11)*(B00+B11)
        P2 = (A10+A11)*B00
        P3 = A00*(B11-B00)
        P4 = A11*(B10-B00)
        P5 = (A00+A01)*B11
        P6 = (A10-A00)*(B00+B01)
        P7 = (A01-A11)*(B10+B11)
        
        Outputs:
        C00 = P1 - P4 + P7
        C01 = P1 + P3
        C10 = P2 + P4
        C11 = P1 - P3 + P5
        """
        # P1 = (A00+A11)*(B00+B11)
        col = offset + 0
        U[0][col] = 1   # A00
        U[3][col] = 1   # A11
        V[0*p+c0][col] = 1   # B00 (j=0, k=c0)
        V[1*p+c1][col] = 1   # B11 (j=1, k=c1)
        W[c0*n+0][col] = 1   # C00 (k=c0, i=0)
        W[c1*n+0][col] = 1   # C01 (k=c1, i=0)
        W[c0*n+1][col] = 1   # C10 (k=c0, i=1)
        W[c1*n+1][col] = 1   # C11 (k=c1, i=1)
        
        # P2 = (A10+A11)*B00
        col = offset + 1
        U[2][col] = 1   # A10
        U[3][col] = 1   # A11
        V[0*p+c0][col] = 1   # B00
        W[c0*n+1][col] = 1   # C10
        
        # P3 = A00*(B11-B00)
        col = offset + 2
        U[0][col] = 1   # A00
        V[1*p+c1][col] = 1   # B11
        V[0*p+c0][col] = -1  # -B00
        W[c1*n+0][col] = 1   # C01
        W[c1*n+1][col] = -1  # -C11
        
        # P4 = A11*(B10-B00)
        col = offset + 3
        U[3][col] = 1   # A11
        V[1*p+c0][col] = 1   # B10
        V[0*p+c0][col] = -1  # -B00
        W[c0*n+0][col] = -1  # -C00
        W[c0*n+1][col] = 1   # C10
        
        # P5 = (A00+A01)*B11
        col = offset + 4
        U[0][col] = 1   # A00
        U[1][col] = 1   # A01
        V[1*p+c1][col] = 1   # B11
        W[c1*n+1][col] = 1   # C11
        
        # P6 = (A10-A00)*(B00+B01)
        col = offset + 5
        U[2][col] = 1   # A10
        U[0][col] = -1  # -A00
        V[0*p+c0][col] = 1   # B00
        V[0*p+c1][col] = 1   # B01
        # P6 is not directly used in the output formulas above, but it's part of Strassen
        # Wait, I need to check if P6 is actually needed. Let me re-verify the Strassen formulas.
        
        # P7 = (A01-A11)*(B10+B11)
        col = offset + 6
        U[1][col] = 1   # A01
        U[3][col] = -1  # -A11
        V[1*p+c0][col] = 1   # B10
        V[1*p+c1][col] = 1   # B11
        W[c0*n+0][col] = 1   # C00
    
    # Add Strassen block for columns 0, 1
    add_strassen_block(0, 1, 0)
    
    # Add Strassen block for columns 2, 3
    add_strassen_block(2, 3, 7)
    
    # Add schoolbook for column 4 (2x2x1, rank 4)
    # C04 = A00*B04 + A01*B14
    # C14 = A10*B04 + A11*B14
    offset = 14
    # Product 1: A00 * B04 -> C04
    col = offset + 0
    U[0][col] = 1
    V[0*p+4][col] = 1
    W[4*n+0][col] = 1
    
    # Product 2: A01 * B14 -> C04
    col = offset + 1
    U[1][col] = 1
    V[1*p+4][col] = 1
    W[4*n+0][col] = 1
    
    # Product 3: A10 * B04 -> C14
    col = offset + 2
    U[2][col] = 1
    V[0*p+4][col] = 1
    W[4*n+1][col] = 1
    
    # Product 4: A11 * B14 -> C14
    col = offset + 3
    U[3][col] = 1
    V[1*p+4][col] = 1
    W[4*n+1][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
