def construct():
    n, m, p = 2, 3, 5
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[i][j] -> i*m + j, i in 0..1, j in 0..2
    # B indices: B[j][k] -> j*p + k, j in 0..2, k in 0..4
    # C indices: C[i][k] -> k*n + i, i in 0..1, k in 0..4
    
    # Strategy: Use shared linear forms to reduce rank
    # 
    # Define shared linear combinations:
    # s0 = a00 + a10  (U[0][r] += 1, U[3][r] += 1)
    # s1 = a01 + a11  (U[1][r] += 1, U[4][r] += 1)
    # s2 = a02 + a12  (U[2][r] += 1, U[5][r] += 1)
    #
    # t0 = b0k for k=0..4
    # t1 = b1k for k=0..4  
    # t2 = b2k for k=0..4
    #
    # Products: s0*t0k, s1*t1k, s2*t2k for each k
    # These give: (a00+a10)*b0k, (a01+a11)*b1k, (a02+a12)*b2k
    #
    # C[0][k] = a00*b0k + a01*b1k + a02*b2k
    #          = (s0*t0k - a10*b0k) + (s1*t1k - a11*b1k) + (s2*t2k - a12*b2k)
    #          = s0*t0k + s1*t1k + s2*t2k - (a10*b0k + a11*b1k + a12*b2k)
    #
    # C[1][k] = a10*b0k + a11*b1k + a12*b2k
    #
    # So C[0][k] + C[1][k] = s0*t0k + s1*t1k + s2*t2k
    # C[0][k] - C[1][k] = 2*(a00*b0k + a01*b1k + a02*b2k) - (a10*b0k + a11*b1k + a12*b2k)
    #                    = a00*b0k + a01*b1k + a02*b2k - a10*b0k - a11*b1k - a12*b2k + (a00*b0k + a01*b1k + a02*b2k)
    # Hmm, this is getting complicated.
    
    # Let me try a cleaner approach.
    # 
    # For each k, we need to compute:
    # C[0][k] = a00*b0k + a01*b1k + a02*b2k
    # C[1][k] = a10*b0k + a11*b1k + a12*b2k
    #
    # That's 2 outputs from 6 inputs (3 from A, 3 from B for each k)
    # But B entries are shared across k!
    
    # Let me try: for each k, use 4 rank terms (like 2x2x1 but with 3x3)
    # Actually for 2x3x1: R(2,3,1) = 6 (naive) or can we do better?
    # R(2,3,1) >= max(2*3, 3*1, 2*1) = 6? No, lower bound is max(nm, mp, np) is not right.
    # Actually R(2,3,1) = 6 since it's just 6 independent products.
    
    # Total naive = 6 * 5 = 30.
    
    # Let me try to share across k values.
    # 
    # Idea: Use the same A-linear-forms across all k, and vary B-linear-forms.
    # 
    # Let's define 4 A-linear forms that are reused:
    # L0 = a00, L1 = a01, L2 = a02, L3 = a10, L4 = a11, L5 = a12
    # That's 6 forms, one per entry.
    # 
    # For B, for each k we need b0k, b1k, b2k.
    # 
    # Alternative: group k values.
    
    # Let me try a block approach:
    # Block 1: k=0,1 (2x3x2)
    # Block 2: k=2,3 (2x3x2)  
    # Block 3: k=4 (2x3x1)
    
    # R(2,3,2): Let's compute. C is 2x2, A is 2x3, B is 3x2.
    # C[0][0] = a00*b00 + a01*b10 + a02*b20
    # C[0][1] = a00*b01 + a01*b11 + a02*b21
    # C[1][0] = a10*b00 + a11*b10 + a12*b20
    # C[1][1] = a10*b01 + a11*b11 + a12*b21
    
    # Naive: 12 terms. Can we do better?
    # Use: (a00+a10)*b00, (a00+a10)*b01, (a01+a11)*b10, (a01+a11)*b11, (a02+a12)*b20, (a02+a12)*b21
    # That's 6 terms. Then:
    # C[0][0] = (a00+a10)*b00 - a10*b00 + a01*b10 + a02*b20
    # Still need a10*b00, a01*b10, a02*b20 separately.
    
    # Let me just try rank 25 and construct carefully.
    
    # Fresh start with a systematic construction:
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # I'll use a construction based on the following insight:
    # For the 2x3x5 multiplication, we can use:
    # - 5 pairs of (a_i, b_jk) products grouped cleverly
    
    # Let me try: use 5 rank terms for "diagonal" and additional for off-diagonal
    
    # Actually, let me try a completely different tactic:
    # Use the identity that for 2x3x5, we can write it as a sum of rank-1 tensors.
    
    # Let me try rank = 23 with a specific construction.
    
    # Reset
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A[i][j] index: i*3 + j
    # B[j][k] index: j*5 + k  
    # C[i][k] index: k*2 + i
    
    # Construction:
    # For each k in 0..4, we need C[0][k] and C[1][k]
    # C[0][k] = a0*b0k + a1*b1k + a2*b2k where a0=a00, a1=a01, a2=a02
    # C[1][k] = a3*b0k + a4*b1k + a5*b2k where a3=a10, a4=a11, a5=a12
    
    # Let me use the following shared forms:
    # For B: use b0k, b1k, b2k directly (15 terms total for B side if no sharing)
    # For A: use a0, a1, a2, a3, a4, a5 (6 terms)
    
    # Total naive: 6 * 5 = 30. Need to reduce by 7.
    
    # Sharing idea: 
    # (a0+a3)*b0k gives a0*b0k + a3*b0k
    # (a1+a4)*b1k gives a1*b1k + a4*b1k
    # (a2+a5)*b2k gives a2*b2k + a5*b2k
    # That's 3*5 = 15 terms, giving us C[0][k]+C[1][k] for each k.
    # Then we need C[0][k] - C[1][k] = a0*b0k - a3*b0k + a1*b1k - a4*b1k + a2*b2k - a5*b2k
    # = (a0-a3)*b0k + (a1-a4)*b1k + (a2-a5)*b2k
    # That's another 3*5 = 15 terms.
    # Total: 30. No improvement.
    
    # Better: 
    # C[0][k] = (a0+a3)*b0k/2 + (a1+a4)*b1k/2 + (a2+a5)*b2k/2 + (a0-a3)*b0k/2 + (a1-a4)*b1k/2 + (a2-a5)*b2k/2
    # C[1][k] = (a0+a3)*b0k/2 + (a1+a4)*b1k/2 + (a2+a5)*b2k/2 - (a0-a3)*b0k/2 - (a1-a4)*b1k/2 - (a2-a5)*b2k/2
    # This uses 6 forms per k, total 30. Same.
    
    # The key is to share across k. Let me define:
    # s0 = a0+a3, s1 = a1+a4, s2 = a2+a5
    # d0 = a0-a3, d1 = a1-a4, d2 = a2-a5
    # 
    # Then for each k:
    # C[0][k] = s0*b0k + s1*b1k + s2*b2k + d0*b0k + d1*b1k + d2*b2k (all /2)
    # C[1][k] = s0*b0k + s1*b1k + s2*b2k - d0*b0k - d1*b1k - d2*b2k (all /2)
    
    # If I use s0, s1, s2 as shared A-forms (3 terms), and d0, d1, d2 as shared A-forms (3 terms),
    # and for each k use b0k, b1k, b2k as B-forms (3 terms per k, 15 total),
    # then total rank = 3 + 3 + 15 = 21? No, that's not how it works.
    
    # Each rank term is one product of one A-form and one B-form.
    # So I need:
    # 3 (s-forms) * 5 (k-values) = 15 terms for s*b products
    # 3 (d-forms) * 5 (k-values) = 15 terms for d*b products
    # Total = 30. Still no savings.
    
    # The savings come from using the SAME A-form with DIFFERENT B-forms in the same rank term? No, each rank term is one A-form times one B-form.
    
    # I think the minimum is actually around 24-25. Let me just go with a clean 24.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Simple construction: for each (i,j,k) triple, but merge where possible
    # A[i][j] = index i*3+j
    # B[j][k] = index j*5+k
    # C[i][k] = index k*2+i
    
    r = 0
    for i in range(2):
        for j in range(3):
            for k in range(5):
                U[i*3+j][r] += 1
                V[j*5+k][r] += 1
                W[k*2+i][r] += 1
                r += 1
    
    # This gives rank 30. I need to reduce.
    
    # Let me try a smarter construction with rank 24.
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use the following decomposition:
    # For k=0,1,2: use 4 terms each (total 12)
    # For k=3,4: use 6 terms each (total 12)
    # Total: 24
    
    # For k=0:
    # C[0][0] = a0*b00 + a1*b10 + a2*b20
    # C[1][0] = a3*b00 + a4*b10 + a5*b20
    # Use:
    # r0: (a0+a3)*b00 -> contributes to both C[0][0] and C[1][0]
    # r1: a1*b10 -> C[0][0]
    # r2: a2*b20 -> C[0][0]
    # r3: a4*b10 -> C[1][0]
    # r4: a5*b20 -> C[1][0]
    # r5: a3*b00 -> C[1][0] (to complete, since r0 gives a0*b00+a3*b00)
    # Wait, that's 6 terms. Let me reconsider.
    
    # Actually: C[0][0] = a0*b00 + a1*b10 + a2*b20
    #            C[1][0] = a3*b00 + a4*b10 + a5*b20
    # 
    # If I use (a0+a3)*b00, I get a0*b00 + a3*b00. 
    # Then C[0][0] needs a1*b10 + a2*b20 more.
    # C[1][0] needs a4*b10 + a5*b20 more.
    # So 1 + 2 + 2 = 5 terms per k? But I also need the cross terms.
    # C[0][0] = (a0+a3)*b00 - a3*b00 + a1*b10 + a2*b20
    # C[1][0] = (a0+a3)*b00 - a0*b00 + a4*b10 + a5*b20
    # That's 1 + 1 + 1 + 1 + 1 + 1 = 6 terms. Same as naive.
    
    # I think for 2x3x1, the rank is indeed 6.
    # So 5 * 6 = 30 is the naive bound.
    
    # The only way to reduce is to share across k.
    # Let me try: use the same A-linear form for multiple k values.
    # 
    # Define: for A, use a0, a1, a2, a3, a4, a5 (6 forms)
    # For B, for each k, use b0k, b1k, b2k (3 forms per k, 15 total)
    # Each (A_form, B_form) pair that appears in the product gets a rank term.
    # 
    # The products that appear:
    # a0*b0k for k=0..4: 5 terms
    # a1*b1k for k=0..4: 5 terms
    # a2*b2k for k=0..4: 5 terms
    # a3*b0k for k=0..4: 5 terms
    # a4*b1k for k=0..4: 5 terms
    # a5*b2k for k=0..4: 5 terms
    # Total: 30 terms.
    
    # To reduce, I need to find A-forms and B-forms that can be combined.
    # For example, if I use (a0+a3) as an A-form, and b0k as a B-form,
    # I get (a0+a3)*b0k = a0*b0k + a3*b0k.
    # This replaces 2 rank terms with 1, saving 1 per k, total 5 savings.
    # But then I need to account for the fact that C[0][k] gets a3*b0k and C[1][k] gets a0*b0k incorrectly.
    # 
    # C[0][k] should have a0*b0k but gets a0*b0k + a3*b0k from this term.
    # C[1][k] should have a3*b0k but gets a0*b0k + a3*b0k from this term.
    # So I need to subtract a3*b0k from C[0][k] and a0*b0k from C[1][k].
    # That requires 2 more rank terms per k, netting 0 savings.
    
    # I'm stuck at 30 with this simple approach. Let me just output the naive 30 and accept a lower score, or try a different structure.
    
    # Actually, let me try a completely different approach: use the fact that we can have fractional coefficients.
    
    # Let me try rank = 25 with a specific construction.
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # I'll use a construction where I group the 5 k-values into 2+2+1
    # and use Strassen-like tricks for the 2x3x2 blocks.
    
    # For a 2x3x2 block (k=0,1):
    # C[0][0] = a0*b00 + a1*b10 + a2*b20
    # C[0][1] = a0*b01 + a1*b11 + a2*b21
    # C[1][0] = a3*b00 + a4*b10 + a5*b20
    # C[1][1] = a3*b01 + a4*b11 + a5*b21
    
    # Use 7 terms (Strassen-like for 2x2x2 extended):
    # s0 = (a0+a3)*b00 -> C[0][0]+=1, C[1][0]+=1 (but wrong split)
    
    # This is getting too complex. Let me just output a clean rank-30 solution.
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(2):
        for j in range(3):
            for k in range(5):
                U[i*3+j][r] = 1
                V[j*5+k][r] = 1
                W[k*2+i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}

