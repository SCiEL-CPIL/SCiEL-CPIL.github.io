"""Matrix multiplication 2,2,5 with rank-minimal exact decomposition."""

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=5
    
    # Use Strassen for column pairs (0,1) and (2,3), schoolbook for column 4.
    # Strassen rank per 2x2x2 block: 7
    # Schoolbook rank for 2x2x1 block: 4
    # Total rank: 7 + 7 + 4 = 18
    
    total_rank = 18
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_strassen_block(col0, col1, offset):
        """Add a 2x2x2 Strassen block for columns col0, col1 starting at offset."""
        # Strassen products for A(2x2) * B(2x2) -> C(2x2)
        # A = [[A00, A01], [A10, A11]], B = [[B0c0, B0c1], [B1c0, B1c1]]
        # U indices: 0=A00, 1=A01, 2=A10, 3=A11
        # V indices: 0=B0c0, 1=B0c1, 2=B1c0, 3=B1c1 (within this block)
        # W indices: 0=C0c0, 1=C0c1, 2=C1c0, 3=C1c1 (within this block)
        
        # P1 = (A00+A11)*(B0c0+B1c1)
        U[0][offset] = 1; U[3][offset] = 1
        V[0*p+col0][offset] = 1; V[1*p+col1][offset] = 1
        W[0*n+0][offset] = 1  # C0c0
        W[0*n+1][offset] = 1  # C0c1
        W[1*n+0][offset] = -1 # C1c0 (from -M4 part... wait, need to be careful)
        W[1*n+1][offset] = 1  # C1c1
        # Actually, let me set up the W coefficients properly.
        # C00 = P1 - P4 + P7
        # C01 = P3 + P1
        # C10 = P2 + P4
        # C11 = P1 - P3 + P5
        # So P1 contributes to: C00(+1), C01(+1), C11(+1)
        W[0*n+0][offset] = 1
        W[0*n+1][offset] = 1
        W[1*n+1][offset] = 1
        W[1*n+0][offset] = 0
        
        # P2 = (A10+A11)*B0c0
        idx = offset + 1
        U[2][idx] = 1; U[3][idx] = 1
        V[0*p+col0][idx] = 1
        # C10 = P2 + P4, so P2 contributes +1 to C10
        W[1*n+0][idx] = 1
        W[0*n+0][idx] = 0
        W[0*n+1][idx] = 0
        W[1*n+1][idx] = 0
        
        # P3 = A00*(B0c1-B1c1)
        idx = offset + 2
        U[0][idx] = 1
        V[0*p+col1][idx] = 1; V[1*p+col1][idx] = -1
        # C01 = P3 + P1, C11 = P1 - P3 + P5, so P3: C01(+1), C11(-1)
        W[0*n+1][idx] = 1
        W[1*n+1][idx] = -1
        W[0*n+0][idx] = 0
        W[1*n+0][idx] = 0
        
        # P4 = A11*(B1c0-B0c0)
        idx = offset + 3
        U[3][idx] = 1
        V[1*p+col0][idx] = 1; V[0*p+col0][idx] = -1
        # C00 = P1 - P4 + P7, C10 = P2 + P4, so P4: C00(-1), C10(+1)
        W[0*n+0][idx] = -1
        W[1*n+0][idx] = 1
        W[0*n+1][idx] = 0
        W[1*n+1][idx] = 0
        
        # P5 = (A00+A01)*B1c1
        idx = offset + 4
        U[0][idx] = 1; U[1][idx] = 1
        V[1*p+col1][idx] = 1
        # C11 = P1 - P3 + P5, so P5: C11(+1)
        W[1*n+1][idx] = 1
        W[0*n+0][idx] = 0
        W[0*n+1][idx] = 0
        W[1*n+0][idx] = 0
        
        # P6 = (A10-A00)*(B0c0+B0c1)
        idx = offset + 5
        U[2][idx] = 1; U[0][idx] = -1
        V[0*p+col0][idx] = 1; V[0*p+col1][idx] = 1
        # This product doesn't appear in the C formulas I have... 
        # Wait, I think I have the wrong Strassen formula set. Let me use a verified one.
        # Actually, I'll just leave P6 and P7 for now and fix the formula.
        W[0*n+0][idx] = 0
        W[0*n+1][idx] = 0
        W[1*n+0][idx] = 0
        W[1*n+1][idx] = 0
        
        # P7 = (A01-A11)*(B1c0+B1c1)
        idx = offset + 6
        U[1][idx] = 1; U[3][idx] = -1
        V[1*p+col0][idx] = 1; V[1*p+col1][idx] = 1
        # C00 = P1 - P4 + P7, so P7: C00(+1)
        W[0*n+0][idx] = 1
        W[0*n+1][idx] = 0
        W[1*n+0][idx] = 0
        W[1*n+1][idx] = 0
    
    # This approach is getting error-prone. Let me just use a simple correct method.
    # I'll use the schoolbook for now to guarantee correctness, rank 20.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
