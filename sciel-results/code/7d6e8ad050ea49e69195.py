"""Tensor decomposition for 2x5x6 matrix multiplication using a ladder/shifted approach."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Ladder/shifted decomposition for n=2.
    # We aim for rank = 2*mp - m = 55.
    # 
    # Construction:
    # For the first output row (i=0), use all mp terms:
    #   For each (j,k): term with U[0*m+j]=1, V[j*p+k]=1, W[k*n+0]=1
    # 
    # For the second output row (i=1), use mp terms but with a "shift":
    #   For each (j,k): term with U[1*m+j]=1, V[j*p+k]=1, W[k*n+1]=1
    #   But we can save m terms by noting that for j=0..m-1, k=0,
    #   we can combine with the first row's terms using linearity.
    #
    # Actually, let me use a cleaner ladder construction:
    # Rank = mp + (mp - m) = 2mp - m
    # 
    # Part 1: mp terms for row 0, all columns
    # Part 2: (mp - m) terms for row 1, excluding m specific terms
    # 
    # The m excluded terms for row 1 are handled by modifying Part 1 terms.
    #
    # Let me use a different, verified approach:
    # Use rank = 2*mp - m + 1 or similar. Let me just try 55 and verify.
    
    # I'll use a construction where:
    # - First mp terms handle row 0 completely
    # - Next (mp - m) terms handle row 1 for k >= 1 (all j) and k=0 for j >= 1
    # - The remaining m terms for row 1, k=0, j=0..m-1 need special handling
    #
    # Actually, let me use a simpler verified construction with rank 55:
    # 
    # Terms 0..mp-1: row 0, all (j,k)
    # Terms mp..2mp-m-1: row 1, all (j,k) except (j,0) for j=0..m-1
    # That's mp + (mp-m) = 2mp-m terms, but we're missing m terms for row 1.
    #
    # This won't work directly. Let me use a different strategy.
    
    # Strategy: Use rank = 2*mp - m.
    # We create 2*mp - m terms.
    # For i=0: all mp (j,k) pairs get a term.
    # For i=1: all mp (j,k) pairs get a term, EXCEPT we merge m of them.
    
    # Better: Let's use the following known technique.
    # Define terms for all (i,j,k) but with a twist:
    # For (i=1, j, k=0) for j=0..m-1, we don't create separate terms.
    # Instead, we modify the (i=0, j, k=0) terms to also contribute to row 1.
    
    # But that would require half-integer coefficients, which is allowed!
    
    # Let me try: rank = 2*mp - m = 55
    # 
    # For (j,k) where k > 0 or i=0: standard terms
    # For (i=1, j, k=0): we need to handle differently
    
    # Actually, the cleanest approach: just use rank 55 with a specific pattern.
    # Let me construct it and verify numerically.
    
    rank = 2 * m * p - m  # 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # First, add all terms for row 0 (i=0)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Now add terms for row 1 (i=1), but skip m terms (j, k=0)
    for j in range(m):
        for k in range(p):
            if k == 0:
                continue  # skip these m terms
            U[1 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 1][col] = 1
            col += 1
    
    # Now we have col = mp + (mp - m) = 2mp - m = 55 terms used.
    # But we're missing the contribution for C[1,0] = sum_j A[1,j]*B[j,0]
    # 
    # We need to add these m missing contributions. But we've used all our rank.
    # 
    # This construction doesn't work as-is. We need a different approach.
    #
    # Let me restart with a valid construction.
    #
    # Use rank = 2*mp. That's the schoolbook. But we want less.
    #
    # Known result: For M_{2,m,p}, the rank is at most 2mp - m + p - 1 or similar.
    # 
    # Let me just use the schoolbook for now to ensure correctness,
    # and try to improve incrementally.
    
    # Reset and use schoolbook
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
