"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split B into two 3x4 blocks.
    # Rank of 2x3x4 is 22 (14 from 2x2x4 + 8 from 2x1x4).
    # Total rank = 22 + 22 = 44.
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Function to add a 2x2 Strassen block for A[0:2, 0:2] * B[0:2, k0:k0+2]
    def add_strassen_block(k0):
        nonlocal col
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k0 + 1][col] = 1
        W[k0 * n + 0][col] = 1
        W[(k0 + 1) * n + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[(k0 + 1) * n + 1][col] = -1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0 * m + 0][col] = 1
        V[0 * p + k0 + 1][col] = 1
        V[1 * p + k0 + 1][col] = -1
        W[(k0 + 1) * n + 0][col] = 1
        W[(k0 + 1) * n + 1][col] = 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k0 + 1][col] = 1
        W[k0 * n + 0][col] = -1
        W[(k0 + 1) * n + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k0 + 1][col] = 1
        W[(k0 + 1) * n + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k0 + 1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    # Function to add the tail part for A[0:2, 2] * B[2, k0:k0+2]
    def add_tail_block(k0):
        nonlocal col
        # P1 = A02 * B2k0
        U[0 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # P2 = A02 * B2k1
        U[0 * m + 2][col] = 1
        V[2 * p + k0 + 1][col] = 1
        W[(k0 + 1) * n + 0][col] = 1
        col += 1
        
        # P3 = A12 * B2k0
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P4 = A12 * B2k1
        U[1 * m + 2][col] = 1
        V[2 * p + k0 + 1][col] = 1
        W[(k0 + 1) * n + 1][col] = 1
        col += 1

    # Block 1: columns 0-3
    # We can process as two 2x2 Strassen blocks and two tail blocks
    # Or we can just call the width-2 helpers twice.
    add_strassen_block(0)
    add_tail_block(0)
    add_strassen_block(2)
    add_tail_block(2)
    
    # Block 2: columns 4-7
    add_strassen_block(4)
    add_tail_block(4)
    add_strassen_block(6)
    add_tail_block(6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
