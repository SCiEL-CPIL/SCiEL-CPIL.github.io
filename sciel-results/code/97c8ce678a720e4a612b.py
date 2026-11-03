# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    
    # Strategy: Block-partitioned decomposition
    # Split p=8 into four p=2 blocks.
    # For each p=2 block, use a rank-7 decomposition for 2x3x2 if available,
    # otherwise fall back to Strassen (7) + Schoolbook (4) = 11.
    
    # We will attempt to use a rank-7 decomposition for 2x3x2.
    # The tensor T_{ijk} for i in 0..1, j in 0..2, k in 0..1.
    # If rank 7 is possible, total rank = 4 * 7 = 28.
    
    # Let's define the rank 7 decomposition for 2x3x2.
    # This is a known result?
    # Actually, the rank of 2x3x2 is 7?
    # Let's assume it is 7 and define the factors.
    # If I cannot define them correctly, I will use the 11-rank fallback.
    
    # For now, I will stick to the 44-rank solution as it is verified.
    # The directive asks to "investigate improvements within its family".
    # The family is block-partitioned.
    # Is there a better partition?
    # What if we split m=3 into m=1 and m=2?
    # That's what we did.
    
    # Let's try to optimize the schoolbook part.
    # The schoolbook part is 2x1x2. Rank 4.
    # Can we share rank terms between the Strassen part and the Schoolbook part?
    # The Strassen part uses A(0,0), A(0,1), A(1,0), A(1,1) and B(0,0), B(0,1), B(1,0), B(1,1).
    # The Schoolbook part uses A(0,2), A(1,2) and B(2,0), B(2,1).
    # They use disjoint indices of A and B.
    # So no sharing is possible.
    
    # Therefore, the rank 44 is minimal for this split strategy.
    
    # Is there a different split?
    # Split p=8 into two p=4?
    # 2x3x4 rank?
    # If we use 2x3x2 (rank 11) twice, rank is 22.
    # Total 44.
    
    # What if we use a 2x2x4 and 2x1x4?
    # 2x2x4 rank?
    # 2x2x4 can be done with 2x2x2 (rank 7) + 2x2x2 (rank 7) = 14?
    # Or maybe lower?
    # Rank of 2x2x4 is 14?
    # If 2x2x4 is 14, and 2x1x4 is 8 (2x1x2 twice), total 22.
    # Same.
    
    # It seems 44 is a robust lower bound for simple splits.
    
    # I will return the current 44-rank solution.
    
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # Local U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global U index: (i_off + li) * m + (j_off + lj)
        # Local V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global V index: (j_off + lj) * p + (k_off + lk)
        # Local W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
        # Global W index: (k_off + lk) * n + (i_off + li)
        
        # m1: (a+d)(e+h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m2: (c+d)e
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1
        
        # m3: a(f-h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m4: d(g-e)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1
        
        # m5: (a+b)h
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1
        
        # m6: (c-a)(e+f)
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m7: (b-d)(g+h)
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        col += 1

    def add_schoolbook_2x1x2(i_off, j_off, k_off):
        nonlocal col
        # 2x1x2 schoolbook: 4 terms
        # i in 0..1, k in 0..1
        for i in range(2):
            for k in range(2):
                U[(i_off+i)*m + (j_off+0)][col] = 1
                V[(j_off+0)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Iterate over 4 blocks of p=2
    for block in range(4):
        k_off = block * 2
        
        # m=2 part (j=0,1)
        add_strassen_2x2x2(0, 0, k_off)
        
        # m=1 part (j=2)
        add_schoolbook_2x1x2(0, 2, k_off)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
