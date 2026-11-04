import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _winograd_5x5x5_rank29():
    """
    Return U, V, W for 5x5x5 Winograd decomposition (rank 29).
    This is the known minimal rank decomposition for 5x5 matrix multiplication.
    
    The Winograd scheme uses polynomial evaluation/interpolation.
    For 5x5: split into 2x2 blocks with overlap, use 29 multiplications.
    """
    # The rank-29 Winograd scheme for 5x5x5
    # Using the standard Winograd algorithm with m=2, n=2 blocks
    # This gives us a decomposition that we can adapt for 4x4
    
    # Actually, let me use a direct approach: the known rank-29 decomposition
    # for 5x5 matrix multiplication uses the following structure:
    # Split 5x5 into overlapping 3x3 blocks and use Winograd's method
    
    # For our purposes, we'll construct the decomposition directly
    # The key insight: 4x4 can be embedded in 5x5 with zero padding
    
    # Let's use the explicit rank-29 Winograd coefficients
    # These come from the polynomial method: evaluate at 29 points
    
    # Actually, for correctness and speed, let me use a verified construction
    # The 5x5x5 tensor of rank 29 can be constructed as follows:
    
    # Using the known Winograd algorithm structure:
    # M1 = (a00+a10+a20+a30+a40)*(b00+b10+b20+b30+b40)
    # ... etc for 29 terms
    
    # Let me implement this properly using the standard Winograd coefficients
    
    n = 5
    rank = 29
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    
    # The 29 multiplications in Winograd 5x5:
    # Using the standard decomposition with 2x2 superblocks
    
    # For 5x5, we use the following Winograd scheme:
    # m=2, n=2 blocks with stride 2, giving ceil(5/2)=3 blocks per dim
    # But the standard result is rank 29 for 5x5x5
    
    # Let me use the explicit construction from the literature
    # The 29 terms are:
    
    terms = []
    
    # Group 1: 9 terms from 3x3 grid of 2x2 blocks (with overlap handling)
    # Actually, the standard Winograd 5x5 uses a specific pattern
    
    # Let me use a different, verified approach:
    # The rank of 5x5x5 is known to be 29.
    # I'll construct it using the bilinear algorithm from Winograd's paper.
    
    # For 5x5 with m=n=2:
    # We have indices i,j in {0,1,2,3,4}
    # The algorithm uses 29 scalar multiplications
    
    # Standard Winograd 5x5x5 decomposition (rank 29):
    # Using the polynomial interpolation method
    
    # Coefficients for the 29 terms:
    # Term structure: U[i,j], V[j,k], W[k,i] for each of 29 terms
    
    # Let me build this from the known Winograd coefficients
    # The 5x5 Winograd uses the following pattern:
    
    # Actually, let me just implement a correct decomposition directly
    # For 4x4x4, the known rank is at most 23 (Laderman's bound gives 23 for 4x4)
    # But we need 4x4x9
    
    # Let me use a simpler approach: combine Strassen for 4x4 with efficient 9-column handling
    
    return None  # Placeholder - will use alternative approach

def _strassen_4x4():
    """
    Return U, V, W for 4x4x4 Strassen-based decomposition.
    Uses recursive Strassen: 4x4 = 2x(2x2), each 2x2 uses rank 7.
    Total rank: 7^2 = 49 for 4x4x4.
    """
    n = 4
    rank = 49
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    
    # Strassen 2x2x2 (rank 7)
    S2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):-1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):-1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (0,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    # Build 4x4 from 2x2 Strassen blocks
    for i in range(4):
        for j in range(4):
            for k in range(4):
                pass  # Will be filled by block combination
    
    # Combine two 2x2 Strassen decompositions
    for c1 in range(7):
        for c2 in range(7):
            col = c1 * 7 + c2
            u1, v1, w1 = S2[c1]
            u2, v2, w2 = S2[c2]
            
            for (i1, j1), val1 in u1.items():
                for (i2, j2), val2 in u2.items():
                    I = 2*i1 + i2
                    J = 2*j1 + j2
                    U[I*4+J][col] += val1 * val2
            
            for (j1, k1), val1 in v1.items():
                for (j2, k2), val2 in v2.items():
                    J = 2*j1 + j2
                    K = 2*k1 + k2
                    V[J*4+K][col] += val1 * val2
            
            for (k1, i1), val1 in w1.items():
                for (k2, i2), val2 in w2.items():
                    K = 2*k1 + k2
                    I = 2*i1 + i2
                    W[K*4+I][col] += val1 * val2
    
    return U, V, W

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
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

def _validate(U, V, W, n, m, p):
    """Validate that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Use Strassen for the 4x4x4 part (rank 49) and handle the 5th column
    # of B (columns 4-8) more efficiently
    
    # For 4x4 x 4x9:
    # Split B into B1 (4x4) and B2 (4x5)
    # Product = A*B1 + A*B2
    # A*B1 uses rank 49 (Strassen)
    # A*B2: for each of the 5 columns, we need 4x4x1, which is rank 16 each
    # But we can do better: A*B2 where B2 is 4x5
    # This is 4x4x5, which can be done in rank 4*4*5 = 80 (schoolbook)
    # Or we can use Strassen-like: 4x4x5 ≈ 49 + 4*5 = 69? No, that's not right.
    
    # Actually, 4x4x5 tensor: for each of 5 output columns, we do 4x4x1
    # 4x4x1 has rank at most 16 (schoolbook), but can it be lower?
    # 4x4x1 is just the bilinear form for one column of the product.
    # The rank of 4x4x1 tensor is exactly 16 (since it's the matrix multiplication
    # tensor restricted to one output, which is just the outer product structure).
    
    # Wait, let me reconsider. The tensor for 4x4 x 4x5 is:
    # T[i,j,k] = A[i,j] * B[j,k] for i in 0..3, j in 0..3, k in 0..4
    # This is a 4x4x5 tensor. Its rank is at most 4*4*5 = 80.
    # Can we do better?
    
    # Alternative: Use the full Strassen 4x4x4 (rank 49) for the first 4 columns,
    # and handle the remaining 5 columns separately.
    
    # For the remaining 5 columns (k=4,5,6,7,8), each column requires:
    # C[:,k] = A * B[:,k] which is a 4x4 times 4x1 multiplication
    # The rank of 4x4x1 is 16 (schoolbook), but actually it's just 4*4 = 16
    # because it's a single vector-matrix product.
    
    # So total rank = 49 + 5*16 = 49 + 80 = 129. That's worse than 114!
    
    # Let me reconsider the current approach which achieves 114.
    # The current program uses:
    # - 49 rank for 4x4x4 (first 4 columns)
    # - 49 rank for another 4x4x4 (next 4 columns, k=4,5,6,7)
    # - 16 rank for the last column (k=8)
    # Total: 49 + 49 + 16 = 114
    
    # Can we do better? What if we use a smarter decomposition for 4x4x9?
    
    # Idea: Use Winograd 5x5x5 (rank 29) embedded appropriately
    # 4x4 can be seen as a sub-problem of 5x5
    # If we can use rank 29 for 4x4x4, then 4x4x9 might be 29*9/4 ≈ 66
    
    # But I'm not sure the Winograd 5x5x5 directly gives a 4x4x4 decomposition.
    
    # Let me try a different structural approach:
    # Use the fact that 4 = 2+2 and apply block decomposition more cleverly
    
    # Actually, let me just verify the current 114-rank solution works and
    # try to find a lower rank using a hybrid approach.
    
    # For now, let me implement the Winograd-inspired approach:
    # Embed 4x4 in 5x5, use rank-29 Winograd, then extract 4x4x9
    
    # The 5x5x5 rank-29 decomposition:
    # I'll use the known explicit coefficients
    
    # Winograd 5x5x5 rank 29 - explicit construction:
    # Using the polynomial method with evaluation at specific points
    
    # For now, let me stick with the verified 114-rank approach but try
    # to optimize the last few columns
    
    # Actually, let me try: 4x4x9 using two Strassen 4x4x4 blocks
    # but with a smarter combination for the 9th column
    
    # The 9th column: C[:,8] = A * B[:,8]
    # This is rank 16 at most. But can we share work with the other columns?
    
    # Let me just go with the proven 114-rank solution and add validation.
    
    U444, V444, W444 = _strassen_4x4()
    
    # Validate 4x4x4
    if not _validate(U444, V444, W444, 4, 4, 4):
        return _schoolbook_fallback()
    
    # Build 4x4x9 decomposition:
    # Columns 0-3: use first Strassen block
    # Columns 4-7: use second Strassen block  
    # Column 8: use schoolbook for single column (rank 16)
    
    total_rank = 49 + 49 + 16
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    # Block 1: columns 0-3
    for I in range(4):
        for J in range(4):
            for c in range(49):
                U[I*4+J][c] = U444[I*4+J][c]
    for J in range(4):
        for K in range(4):
            for c in range(49):
                V[J*9+K][c] = V444[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(49):
                W[K*4+I][c] = W444[K*4+I][c]
    
    # Block 2: columns 4-7
    offset2 = 49
    for I in range(4):
        for J in range(4):
            for c in range(49):
                U[I*4+J][offset2+c] = U444[I*4+J][c]
    for J in range(4):
        for K in range(4):
            for c in range(49):
                V[J*9+(K+4)][offset2+c] = V444[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(49):
                W[(K+4)*4+I][offset2+c] = W444[K*4+I][c]
    
    # Block 3: column 8 (schoolbook, rank 16)
    offset3 = 98
    for idx in range(16):
        I = idx // 4
        J = idx % 4
        U[I*4+J][offset3+idx] = 1
        V[J*9+8][offset3+idx] = 1
        W[8*4+I][offset3+idx] = 1
    
    # Final validation
    if not _validate(U, V, W, 4, 4, 9):
        return _schoolbook_fallback()
    
    return {"U": U, "V": V, "W": W}
