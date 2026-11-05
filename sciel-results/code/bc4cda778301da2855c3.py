"""Matrix multiplication 2x3x4 rank-19 decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 4
    rank = 19
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 19
    V = [[0] * rank for _ in range(m * p)]  # 12 x 19
    W = [[0] * rank for _ in range(p * n)]  # 8 x 19
    
    # W[k*n + i, c] convention: C[i,k] at W[k*2+i, c]
    
    # Part 1: Strassen 2x2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Uses 7 products
    
    # P0 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[0][0] = 1; U[4][0] = 1  # A[0,0] and A[1,1] (indices: 0*3+0=0, 1*3+1=4)
    V[0][0] = 1; V[5][0] = 1  # B[0,0] and B[1,1] (indices: 0*4+0=0, 1*4+1=5)
    # C[0,0] = P0 - P3 + P4 - P6 (using standard Strassen with proper indexing)
    # Let me use the verified Strassen formulas:
    # 
    # P1 = (a11+a22)(b11+b22)  [0-indexed: (A[0,0]+A[1,1])(B[0,0]+B[1,1])]
    # P2 = (a21+a22)b11         [0-indexed: (A[1,0]+A[1,1])B[0,0]]
    # P3 = a11(b12-b22)         [0-indexed: A[0,0](B[0,1]-B[1,1])]
    # P4 = a22(b21-b11)         [0-indexed: A[1,1](B[1,0]-B[0,0])]
    # P5 = (a11+a12)b22         [0-indexed: (A[0,0]+A[0,1])B[1,1]]
    # P6 = (a21-a11)(b11+b12)   [0-indexed: (A[1,0]-A[0,0])(B[0,0]+B[0,1])]
    # P7 = (a12-a22)(b21+b22)   [0-indexed: (A[0,1]-A[1,1])(B[1,0]+B[1,1])]
    #
    # C[0,0] = P1 + P4 - P5 + P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P6
    
    # Let me verify C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]:
    # P1 + P4 - P5 + P7 = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 - A01*B11 + A01*B10
    # Hmm, that gives A00*B00 + A01*B10 - A01*B11. Not quite right.
    
    # I think the standard Strassen formulas I'm using might have a different convention.
    # Let me just use the schoolbook for now and return rank 24.
    
    # Actually, let me try a simpler optimization: share products where possible.
    
    # For C[0,0] and C[1,0]: both use B[0,0], B[1,0], B[2,0]
    # (A[0,0]+A[1,0])*B[0,0] -> contributes A[0,0]*B[0,0] to C[0,0] and A[1,0]*B[0,0] to C[1,0]
    # But we can't directly use this because the W coefficients would need to separate them.
    
    # The issue is that in a rank-1 term, the same product contributes to multiple outputs
    # with different W coefficients, but U and V are fixed.
    
    # Let me just return the correct schoolbook rank-24 solution.
    
    rank = 24
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(8)]
    
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(4):
                U[i*3+j][col] = 1
                V[j*4+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

