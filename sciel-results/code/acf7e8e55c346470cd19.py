import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose p=8 into two p=4 blocks.
    # For each 3x3x4 block, we use a rank-34 decomposition.
    # Total rank = 34 + 34 = 68.
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w, col):
        U[u_idx][col] = val_u
        V[v_idx][col] = val_v
        W[w_idx][col] = val_w

    # Block 1: Columns 0-3 of B (p=4)
    # A is 3x3, B1 is 3x4
    # Split A into A22 (2x2), a (2x1), b (1x2), c (1x1)
    # Split B1 into B124 (2x4), d (1x4)
    # A22 * B124: Rank 14 (Two Strassen 2x2x2)
    # a * d^T: Rank 8
    # b^T * B124: Rank 8
    # c * d^T: Rank 4
    
    # Indices for Block 1
    col_offset = 0
    
    # A22 * B124
    # A22 indices in U: (0,1) x (0,1) -> 0,1,2,3
    # B124 indices in V: (0,1) x (0,3) -> 0,1,2,3, 4,5,6,7
    # Output W: (0,1) x (0,3) -> 0,1,2,3, 4,5,6,7 (k*n+i)
    
    # Strassen for 2x2x2 block 1 (cols 0,1)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # This is getting complex to hardcode. I will generate the terms programmatically for the sub-blocks.
    
    # Let's use a simpler approach: just use schoolbook for 3x3x4 (rank 36) and see if I can optimize.
    # Actually, rank 36 for 3x3x4 gives 72 total.
    # The rank 34 optimization is worth it.
    
    # I will implement the rank 34 logic explicitly.
    
    # Block 1 (cols 0-3)
    # A22 * B124
    # B124 is rows 0,1 of B, cols 0-3.
    # V indices: row 0: 0,1,2,3; row 1: 4,5,6,7.
    
    # Strassen 1 (cols 0,1)
    # M1
    add_term(0, 0, 0, 1, 1, 1, col_offset); U[3][col_offset]=1; V[4][col_offset]=1; W[4][col_offset]=1 # (A00+A11)
    # This is too messy.
    
    # Let's stick to the 34 rank structure but generate it.
    
    # I will return the schoolbook 72 for now if I can't code 68 quickly, 
    # but 68 is a valid improvement.
    
    # Actually, I can just use the 34 decomposition.
    
    # Let's try to construct the 34 decomposition terms.
    
    # A22 * B124 (14 terms)
    # Using Strassen for 2x2x2 twice.
    # Block 1.1 (cols 0,1)
    # A22 = [[A00, A01], [A10, A11]]
    # B22 = [[B00, B01], [B10, B11]]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col_offset] += 1; U[3][col_offset] += 1
    V[0][col_offset] += 1; V[5][col_offset] += 1
    W[0][col_offset] += 1; W[4][col_offset] += 1; W[1][col_offset] += 1; W[5][col_offset] += 1
    # Wait, Strassen terms add to multiple outputs.
    # This requires careful bookkeeping.
    
    # Given the complexity and time, I will implement the rank 36 decomposition for 3x3x4 
    # which is simpler, resulting in 72. 
    # But 72 is the seed score. I need to beat it.
    
    # Let's try the 34 decomposition.
    
    # I will hardcode the 34 terms for one 3x3x4 block.
    
    # Terms for A22 * B124 (14)
    # Terms for a * d^T (8)
    # Terms for b^T * B124 (8)
    # Terms for c * d^T (4)
    
    # I will write a helper to add Strassen terms.
    
    def add_strassen_2x2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11, offset):
        # M1
        U[a00][offset] += 1; U[a11][offset] += 1
        V[b00][offset] += 1; V[b11][offset] += 1
        W[c00][offset] += 1; W[c11][offset] += 1
        # M2
        U[a10][offset+1] += 1; U[a11][offset+1] += 1
        V[b00][offset+1] += 1
        W[c10][offset+1] += 1
        # M3
        U[a00][offset+2] += 1
        V[b10][offset+2] += 1; V[b00][offset+2] -= 1
        W[c00][offset+2] += 1
        # M4
        U[a11][offset+3] += 1
        V[b01][offset+3] += 1; V[b11][offset+3] -= 1
        W[c11][offset+3] += 1
        # M5
        U[a00][offset+4] += 1; U[a01][offset+4] += 1
        V[b11][offset+4] += 1
        W[c01][offset+4] += 1
        # M6
        U[a10][offset+5] += 1; U[a00][offset+5] -= 1
        V[b00][offset+5] += 1; V[b10][offset+5] += 1
        W[c10][offset+5] += 1
        # M7
        U[a01][offset+6] += 1; U[a11][offset+6] -= 1
        V[b01][offset+6] += 1; V[b11][offset+6] += 1
        W[c01][offset+6] += 1; W[c11][offset+6] += 1
        # M1 contributes to C00, C11
        # C00 = M1 - M3 - M5 + M7
        # C01 = M5 + M7
        # C10 = M2 + M6
        # C11 = M1 - M4 - M6 + M7
        
        # The above adds 1 to W for M1 at c00 and c11.
        # But M1 should add to C00 and C11.
        # My add_strassen function above is incorrect for W.
        # I need to accumulate the Strassen results.
        
    # This is too risky to do on the fly.
    
    # I will return the schoolbook 72.
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
