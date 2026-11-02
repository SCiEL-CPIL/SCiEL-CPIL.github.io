"""Matrix multiplication 4x4x9 - Winograd Fourier Transform Technique"""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 9  # U: n*m=16, V: m*p=36, W: p*n=36
    
    # We'll use a block approach: partition the 9 columns into three 3-column blocks
    # For each 4x4x3 block, we use a rank-12 decomposition (schoolbook for 4x4x3 is 48, 
    # but we can do better with Winograd)
    # 
    # Actually, let me think about this differently. The key insight for Winograd FFT is:
    # 
    # For multiplying A(4x4) * B(4x9), we can write B as [B1 | B2 | B3] where each Bi is 4x3.
    # The result is [A*B1 | A*B2 | A*B3].
    # 
    # For each A(4x4) * B(4x3) multiplication, we can use the fact that 4 = 2*2 and 3 = 3*1.
    # 
    # Let me use a simpler but effective approach:
    # Partition the 4x4 matrix A into 2x2 blocks:
    # A = [A11 A12; A21 A22] where each is 2x2
    # B is 4x9, partition rows into two groups of 2: B = [B_top; B_bottom] where each is 2x9
    # 
    # Then A*B = [A11*B_top + A12*B_bottom, A21*B_top + A22*B_bottom]
    # 
    # This doesn't directly help. Let me try a different strategy.
    #
    # Strategy: Use the rank-14 decomposition for 2x2x2 (Strassen) and build up.
    #
    # Actually, for a clean improvement, let me use the following approach:
    # The 4x4x9 problem can be seen as 9 independent 4x4x1 multiplications (each column of B).
    # A 4x4x1 multiplication has rank at most 16 (schoolbook). Total: 9*16=144.
    # But we can share computations across the 9 columns.
    #
    # Better approach: Use the fact that we can do a 4x4 * 4x9 in rank 16*9/ something.
    #
    # Let me use a block-Winograd approach:
    # 1. Partition B(4x9) into three 4x3 blocks: B1, B2, B3
    # 2. For each block, compute A(4x4) * Bi(4x3) using a rank-16 method
    # 3. Total rank: 3 * 16 = 48
    #
    # For 4x4x3: We can use the following decomposition.
    # Think of it as computing C(4x3) = A(4x4) * B(4x3).
    # Partition A into 2x2 blocks: A = [A11 A12; A21 A22], each 2x2.
    # Partition B into top 2 rows and bottom 2 rows: B = [Bt; Bb], each 2x3.
    # Then C = [A11*Bt + A12*Bb; A21*Bt + A22*Bb]
    # 
    # Each term like A11*Bt is 2x2 * 2x3 = 2x3, which has rank at most 6.
    # We have 4 such terms, so naive total is 4*6=24. But we can share.
    #
    # Using Strassen-like sharing:
    # M1 = (A11+A22)*(Bt+Bb)  -> 2x3, rank 6
    # M2 = A12*Bb             -> 2x3, rank 6  
    # M3 = (A11+A21)*Bt       -> 2x3, rank 6
    # M4 = A22*(Bb-Bt)        -> 2x3, rank 6
    # 
    # C11 = M1 - M4
    # C12 = M1 + M2  -- wait, this doesn't work cleanly.
    #
    # Let me just use a direct but slightly optimized approach.
    # For 4x4x3, I'll use rank 24 (improvement from 48).
    #
    # Actually, the simplest reliable improvement: use the schoolbook for 4x4x9 (rank 144)
    # but that's the same as current. Let me just do a correct block decomposition.
    #
    # I'll use rank 48: three independent 4x4x3 blocks, each with rank 16.
    # For 4x4x3, I can use a rank-16 decomposition.
    
    # For a 4x4x3 matrix multiplication, here's a rank-16 decomposition:
    # We compute C(4x3) = A(4x4) * B(4x3)
    # Think of B as having 3 columns: b1, b2, b3 (each 4x1)
    # C = [A*b1, A*b2, A*b3]
    # 
    # For A(4x4)*b(4x1), schoolbook is rank 16. For 3 columns, naive is 48.
    # But we can share: the A*b computation shares the A matrix.
    # 
    # Here's a rank-16 approach for 4x4x3:
    # Use the fact that we can compute all 3 column multiplications simultaneously.
    # Let's define intermediate products.
    
    # I'll construct the full decomposition explicitly for 4x4x9 with rank 48.
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]  # 16 x 48
    V = [[0] * rank for _ in range(m * p)]  # 36 x 48
    W = [[0] * rank for _ in range(p * n)]  # 36 x 48
    
    # For each of the 3 blocks (columns 0-2, 3-5, 6-8), we use rank 16
    for block in range(3):
        base_col = block * 16
        k_offset = block * 3  # offset into the 9 columns
        
        # For the 4x4x3 block, use schoolbook: 4*4*3 = 48 terms, but we want rank 16.
        # Actually, let me use a different strategy.
        # 
        # Use the following rank-16 decomposition for 4x4x3:
        # We'll use 16 scalar multiplications.
        
        # Method: For 4x4 * 4x3, think of it as 4 independent (4x4 * 4x1) but sharing.
        # 
        # Let me use a simple but valid rank-24 decomposition for 4x4x3, giving total 72.
        # Or better, let me just verify the schoolbook is 144 and try to get something smaller.
        
        # I'll use a straightforward approach: for each block of 3 columns, use schoolbook rank 48/3=16 per column pair... 
        # This is getting complicated. Let me use a known good decomposition.
        
        # For 4x4x9, a valid rank-48 decomposition:
        # Process 3 groups of 3 output columns. For each group, use a rank-16 method.
        
        # Rank-16 for 4x4x3:
        # C_{i,k} = sum_j A_{i,j} * B_{j,k}
        # 
        # Use the following 16 terms:
        # We partition j into {0,1} and {2,3}
        # M_{a,b} for a in {0,1}, b in {0,1} where:
        #   M_{0,0} involves A rows 0,1 and B rows 0,1
        #   etc.
        
        # Actually, let me just use a clean rank-48 total by doing 3x4x4x3 with sharing.
        
        # Simplest correct approach: rank 144 schoolbook, but that's not an improvement.
        # Let me do rank 72: for each of the 9 columns, use rank 8 (improved 4x4x1).
        
        # For 4x4x1 (single column), can we do better than 16?
        # C_i = sum_j A_{i,j} * B_j
        # This is essentially computing 4 dot products of length 4.
        # Schoolbook: 16 multiplications.
        # Can we do it in fewer? The rank of the 4x4x1 tensor is 16 (it's the identity-like tensor).
        # Actually no, it's the "matrix-vector" tensor which has rank 16 for 4x4.
        
        # I think the best I can reliably do without extensive research is to use a block approach
        # that gives rank < 144. Let me use rank 72.
        
        # For 4x4x9, use 9 independent 4x4x1 with rank 8 each? No, 4x4x1 has rank 16.
        
        # OK, let me just use a valid rank-48 decomposition.
        # 
        # Approach: 4x4x9 = 2 x (2x4x9) where we partition A's rows into two groups of 2.
        # For 2x4x9: C(2x9) = A(2x4) * B(4x9)
        # C_i,k = sum_j A_{i,j} * B_{j,k}
        # Rank of 2x4x9 tensor: at most 2*4*9 = 72 schoolbook.
        # We can do 2x4x9 in rank 2*9=18? No.
        # 
        # Let me use: 2x4x9 can be done in rank 2*4*9/2 = 36 with some sharing.
        # Total: 2*36 = 72.
        
        # I'll construct a rank-72 decomposition.
        pass
    
    # Given the complexity, let me just return a correct but not improved solution
    # to ensure validity. The schoolbook rank 144 is correct.
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
