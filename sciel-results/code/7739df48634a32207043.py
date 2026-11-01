import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_R B_R (14 terms)
    # A_R is column 2 of A (indices 2, 5 in U)
    # B_R is row 2 of B (indices 14..20 in V)
    # C indices: k*n + i
    for i in range(2):
        for k in range(7):
            U[i * 3 + 2][col] = 1
            V[2 * 7 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: A_L B_L (25 terms)
    # A_L is cols 0,1 of A (indices 0,1 and 3,4 in U)
    # B_L is rows 0,1 of B (indices 0..12 in V)
    
    # Strassen for 2x2x2 blocks
    # Block 0: k=0,1
    # Block 1: k=2,3
    # Block 2: k=4,5
    
    def add_strassen_block(k0, k1, offset):
        # M1 = (A00+A11)(B00+B11)
        U[0][offset] = 1; U[4][offset] = 1
        V[k0][offset] = 1; V[7+k1][offset] = 1
        W[k0*2+0][offset] = 1
        W[k1*2+0][offset] = 1
        W[k0*2+1][offset] = 1
        W[k1*2+1][offset] = 1
        # Note: W entries are sums, so we add 1 to each relevant W[k*n+i]
        # M1 contributes to C00, C01, C10, C11 with coeff 1, 1, 1, 1?
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's map the 7 Strassen multiplications to columns
        # We are adding 7 columns for this block
        
        # M1
        c = offset
        U[0][c] = 1; U[4][c] = 1
        V[k0][c] = 1; V[7+k1][c] = 1
        W[k0*2+0][c] = 1
        W[k1*2+0][c] = 1
        W[k0*2+1][c] = 1
        W[k1*2+1][c] = 1
        # Wait, Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: (A00+A11)(B00+B11)
        # A part: A00, A11 -> U[0], U[4]
        # B part: B00, B11 -> V[k0], V[7+k1]
        # W part: contributes to C00, C01, C10, C11?
        # No, M1 is a scalar. It is added to C00 and C11.
        # C00 += M1
        # C11 += M1
        # So W[k0*2+0] += 1, W[k1*2+1] += 1
        
        # Let's restart the Strassen block logic carefully.
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will add 7 columns.
        
        # Col 0: M1
        c = offset
        U[0][c] = 1; U[4][c] = 1
        V[k0][c] = 1; V[7+k1][c] = 1
        W[k0*2+0][c] = 1
        W[k1*2+1][c] = 1
        
        # Col 1: M2
        c = offset + 1
        U[3][c] = 1; U[4][c] = 1
        V[k0][c] = 1
        W[k0*2+1][c] = 1
        W[k1*2+1][c] = -1
        
        # Col 2: M3
        c = offset + 2
        U[0][c] = 1
        V[7+k0][c] = 1; V[7+k1][c] = -1
        W[k1*2+0][c] = 1
        W[k1*2+1][c] = 1
        
        # Col 3: M4
        c = offset + 3
        U[4][c] = 1
        V[7+k1][c] = 1; V[k0][c] = -1
        W[k0*2+0][c] = 1
        W[k0*2+1][c] = 1
        
        # Col 4: M5
        c = offset + 4
        U[0][c] = 1; U[1][c] = 1
        V[7+k1][c] = 1
        W[k0*2+0][c] = -1
        W[k1*2+0][c] = 1
        
        # Col 5: M6
        c = offset + 5
        U[3][c] = 1; U[0][c] = -1
        V[k0][c] = 1; V[7+k1][c] = 1
        W[k1*2+1][c] = 1
        
        # Col 6: M7
        c = offset + 6
        U[1][c] = 1; U[4][c] = -1
        V[7+k0][c] = 1; V[7+k1][c] = 1
        W[k0*2+0][c] = 1
        W[k1*2+1][c] = -1
        # Wait, C00 has +M7. C11 has -M7?
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # M7 is only in C00.
        # So W[k0*2+0] += 1.
        # And C01 = M3 + M5. No M7.
        # C10 = M2 + M4. No M7.
        # C11 = ... No M7.
        
        # Let's re-verify Strassen W coefficients.
        # C00: M1(+), M4(+), M5(-), M7(+)
        # C01: M3(+), M5(+)
        # C10: M2(+), M4(+)
        # C11: M1(+), M2(-), M3(+), M6(+)
        
        # My W assignments above:
        # M1: W[0,0]=1, W[1,1]=1. (Correct: C00, C11)
        # M2: W[0,1]=1, W[1,1]=-1. (Correct: C10, C11)
        # M3: W[1,0]=1, W[1,1]=1. (Correct: C01, C11) -> Wait. C01 is row 0, col 1. Index 1*2+0? No.
        # Index k*n+i.
        # C01: k=1, i=0. Index 1*2+0 = 2.
        # C11: k=1, i=1. Index 1*2+1 = 3.
        # C00: k=0, i=0. Index 0.
        # C10: k=0, i=1. Index 1.
        
        # Let's map indices:
        # C00 -> W[0]
        # C10 -> W[1]
        # C01 -> W[2]
        # C11 -> W[3]
        
        # M1: C00(+), C11(+) -> W[0]+=1, W[3]+=1
        # M2: C10(+), C11(-) -> W[1]+=1, W[3]-=1
        # M3: C01(+), C11(+) -> W[2]+=1, W[3]+=1
        # M4: C00(+), C10(+) -> W[0]+=1, W[1]+=1
        # M5: C00(-), C01(+) -> W[0]-=1, W[2]+=1
        # M6: C11(+) -> W[3]+=1
        # M7: C00(+) -> W[0]+=1
        
        # Let's rewrite the block function correctly.
        
    # Redefine add_strassen_block with correct W indices
    
    def add_strassen_block(k0, k1, offset):
        # k0, k1 are global k indices for the 2x2 block
        # Local indices for B block:
        # B00 -> V[k0], B01 -> V[k1]
        # B10 -> V[7+k0], B11 -> V[7+k1]
        
        # M1
        c = offset
        U[0][c] = 1; U[4][c] = 1
        V[k0][c] = 1; V[7+k1][c] = 1
        W[k0*2+0][c] = 1
        W[k1*2+1][c] = 1
        
        # M2
        c = offset + 1
        U[3][c] = 1; U[4][c] = 1
        V[k0][c] = 1
        W[k0*2+1][c] = 1
        W[k1*2+1][c] = -1
        
        # M3
        c = offset + 2
        U[0][c] = 1
        V[7+k0][c] = 1; V[7+k1][c] = -1
        W[k1*2+0][c] = 1
        W[k1*2+1][c] = 1
        
        # M4
        c = offset + 3
        U[4][c] = 1
        V[7+k1][c] = 1; V[k0][c] = -1
        W[k0*2+0][c] = 1
        W[k0*2+1][c] = 1
        
        # M5
        c = offset + 4
        U[0][c] = 1; U[1][c] = 1
        V[7+k1][c] = 1
        W[k0*2+0][c] = -1
        W[k1*2+0][c] = 1
        
        # M6
        c = offset + 5
        U[3][c] = 1; U[0][c] = -1
        V[k0][c] = 1; V[7+k1][c] = 1
        W[k1*2+1][c] = 1
        
        # M7
        c = offset + 6
        U[1][c] = 1; U[4][c] = -1
        V[7+k0][c] = 1; V[7+k1][c] = 1
        W[k0*2+0][c] = 1

    # Add blocks 0, 1, 2
    add_strassen_block(0, 1, 14)
    add_strassen_block(2, 3, 21)
    add_strassen_block(4, 5, 28)
    
    # Block 3: k=6 (2x2x1)
    # C0,6 = A00 B06 + A01 B16
    # C1,6 = A10 B06 + A11 B16
    
    # M8 = A00 B06
    c = 35
    U[0][c] = 1
    V[6][c] = 1
    W[6*2+0][c] = 1
    
    # M9 = A01 B16
    c = 36
    U[1][c] = 1
    V[7+6][c] = 1
    W[6*2+0][c] = 1
    
    # M10 = A10 B06
    c = 37
    U[3][c] = 1
    V[6][c] = 1
    W[6*2+1][c] = 1
    
    # M11 = A11 B16
    c = 38
    U[4][c] = 1
    V[7+6][c] = 1
    W[6*2+1][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
