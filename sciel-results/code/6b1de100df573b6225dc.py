"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(r0, c0, d0):
        # r0: start row of A block
        # c0: start col of A block / start row of B block
        # d0: start col of B block
        # Block is 2x2 * 2x2 -> 2x2
        # A indices: (r0, c0), (r0, c0+1), (r0+1, c0), (r0+1, c0+1)
        # B indices: (c0, d0), (c0, d0+1), (c0+1, d0), (c0+1, d0+1)
        # C indices: (r0, d0), (r0, d0+1), (r0+1, d0), (r0+1, d0+1)
        
        # M1: (A00+A11)(B00+B11)
        U[r0 * m + c0][col] += 1
        U[(r0+1) * m + (c0+1)][col] += 1
        V[c0 * p + d0][col] += 1
        V[(c0+1) * p + (d0+1)][col] += 1
        W[d0 * n + r0][col] += 1
        W[(d0+1) * n + r0][col] += 1
        W[d0 * n + (r0+1)][col] += 1
        W[(d0+1) * n + (r0+1)][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[(r0+1) * m + c0][col] += 1
        U[(r0+1) * m + (c0+1)][col] += 1
        V[c0 * p + d0][col] += 1
        W[d0 * n + (r0+1)][col] += 1
        W[(d0+1) * n + (r0+1)][col] -= 1
        col += 1
        
        # M3: A00(B10-B11)
        U[r0 * m + c0][col] += 1
        V[(c0+1) * p + d0][col] += 1
        V[(c0+1) * p + (d0+1)][col] -= 1
        W[(d0+1) * n + r0][col] += 1
        W[(d0+1) * n + (r0+1)][col] += 1
        col += 1
        
        # M4: A11(B01-B10)
        U[(r0+1) * m + (c0+1)][col] += 1
        V[c0 * p + (d0+1)][col] += 1
        V[(c0+1) * p + d0][col] -= 1
        W[d0 * n + r0][col] += 1
        W[d0 * n + (r0+1)][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[r0 * m + c0][col] += 1
        U[r0 * m + (c0+1)][col] += 1
        V[(c0+1) * p + (d0+1)][col] += 1
        W[d0 * n + r0][col] -= 1
        W[(d0+1) * n + r0][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[(r0+1) * m + c0][col] += 1
        U[r0 * m + c0][col] -= 1
        V[c0 * p + d0][col] += 1
        V[c0 * p + (d0+1)][col] += 1
        W[(d0+1) * n + (r0+1)][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[r0 * m + (c0+1)][col] += 1
        U[(r0+1) * m + (c0+1)][col] -= 1
        V[(c0+1) * p + d0][col] += 1
        V[(c0+1) * p + (d0+1)][col] += 1
        W[d0 * n + r0][col] += 1
        col += 1

    def add_schoolbook_221(r0, c0, d0):
        # r0: start row of A block
        # c0: start col of A block / start row of B block
        # d0: start col of B block (only 1 col)
        # Block is 2x2 * 2x1 -> 2x1
        
        # M1: A00 B00 -> C00
        U[r0 * m + c0][col] += 1
        V[c0 * p + d0][col] += 1
        W[d0 * n + r0][col] += 1
        col += 1
        
        # M2: A01 B10 -> C00
        U[r0 * m + (c0+1)][col] += 1
        V[(c0+1) * p + d0][col] += 1
        W[d0 * n + r0][col] += 1
        col += 1
        
        # M3: A10 B00 -> C10
        U[(r0+1) * m + c0][col] += 1
        V[c0 * p + d0][col] += 1
        W[d0 * n + (r0+1)][col] += 1
        col += 1
        
        # M4: A11 B10 -> C10
        U[(r0+1) * m + (c0+1)][col] += 1
        V[(c0+1) * p + d0][col] += 1
        W[d0 * n + (r0+1)][col] += 1
        col += 1

    col = 0
    # 6 Strassen blocks (2x2x2)
    # Split B cols into 0-1, 2-3, 4-5, 6.
    # Split A cols into 0-1, 2-3.
    
    # B cols 0-1
    add_strassen(0, 0, 0) # A(0:2, 0:2) * B(0:2, 0:2)
    add_strassen(0, 2, 0) # A(0:2, 2:4) * B(2:4, 0:2)
    
    # B cols 2-3
    add_strassen(0, 0, 2) # A(0:2, 0:2) * B(0:2, 2:4)
    add_strassen(0, 2, 2) # A(0:2, 2:4) * B(2:4, 2:4)
    
    # B cols 4-5
    add_strassen(0, 0, 4) # A(0:2, 0:2) * B(0:2, 4:6)
    add_strassen(0, 2, 4) # A(0:2, 2:4) * B(2:4, 4:6)
    
    # B col 6
    add_schoolbook_221(0, 0, 6) # A(0:2, 0:2) * B(0:2, 6:7)
    add_schoolbook_221(0, 2, 6) # A(0:2, 2:4) * B(2:4, 6:7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
