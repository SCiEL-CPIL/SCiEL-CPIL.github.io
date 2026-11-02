"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank-minimal decomposition for 2x3x8 matrix multiplication.
    # Strategy: Block-partitioned approach.
    # Split the output dimension p=8 into two blocks of size 4.
    # For each p=4 block, we need to compute a 2x3x4 multiplication.
    # For the 2x3x4 sub-problem, split the inner dimension m=3 into m=2 and m=1.
    #   - The m=2 part is a 2x2x4 problem. Split p=4 into two p=2 blocks.
    #     Each 2x2x2 block uses Strassen's algorithm (rank 7).
    #     Total rank for 2x2x4 = 7 + 7 = 14.
    #   - The m=1 part is a 2x1x4 problem. This is just 8 independent scalar multiplications.
    #     Total rank for 2x1x4 = 8.
    # Total rank per p=4 block = 14 + 8 = 22.
    # Total rank for p=8 = 22 + 22 = 44.
    
    rank = 44
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # Shape (2*3, 44) = (6, 44)
    V = [[0] * rank for _ in range(m * p)]  # Shape (3*8, 44) = (24, 44)
    W = [[0] * rank for _ in range(p * n)]  # Shape (8*2, 44) = (16, 44)
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        """
        Adds a 2x2x2 Strassen decomposition at the specified offset.
        i_off: offset in n dimension (0 or 1)
        j_off: offset in m dimension (0, 1, or 2)
        k_off: offset in p dimension (0, 2, 4, or 6)
        """
        nonlocal col
        # U index: (i, j) -> i*m + j
        # V index: (j, k) -> j*p + k
        # W index: (k, i) -> k*n + i  (Note: W[k*n+i, r] convention)
        
        # m1 = (a+d)(e+h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m2 = (c+d)e
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1
        
        # m3 = a(f-h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m4 = d(g-e)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1
        
        # m5 = (a+b)h
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        col += 1

    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        """
        Adds a 2x1x4 schoolbook decomposition.
        i_off: offset in n dimension (0)
        j_off: offset in m dimension (2)
        k_off: offset in p dimension (0 or 4)
        """
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off+0)][col] = 1
                V[(j_off+0)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Block 1: k in 0..3
    # Part 1: m=2 part (j in 0..1), split into two 2x2x2 Strassen blocks
    #   Block 1a: j in 0..1, k in 0..1
    add_strassen(0, 0, 0)
    #   Block 1b: j in 0..1, k in 2..3
    add_strassen(0, 0, 2)
    # Part 2: m=1 part (j=2), 2x1x4 schoolbook
    add_schoolbook_2x1x4(0, 2, 0)

    # Block 2: k in 4..7
    # Part 1: m=2 part (j in 0..1), split into two 2x2x2 Strassen blocks
    #   Block 2a: j in 0..1, k in 4..5
    add_strassen(0, 0, 4)
    #   Block 2b: j in 0..1, k in 6..7
    add_strassen(0, 0, 6)
    # Part 2: m=1 part (j=2), 2x1x4 schoolbook
    add_schoolbook_2x1x4(0, 2, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
