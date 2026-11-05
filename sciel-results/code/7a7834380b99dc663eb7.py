import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS

    U_list = []
    V_list = []
    W_list = []

    # rank-9 decomposition for 3x3x3 matrix multiplication
    # Each term: (i_idx, k_idx, j_idx, coeffs on C[1,1], C[2,2])
    # This gives exact 3x3 multiplication with 9 terms
    # (schoolbook with some cancellation - actually just schoolbook for 3x3 = 9 terms)
    # But we can do better by noting that for 3x3, the minimum is 7 (Strassen-like)
    # Let's use the known 7-term decomposition for 3x3x3

    # Strassen-like 7-term decomposition for 3x3x3
    # Term structure: each term has entries in U (3x3), V (3x3), W (3x3)
    # Using the standard bilinear map approach for 3x3

    # Actually for 3x3x3, rank 7 is not known. The best known is around 9-10.
    # Let's just use schoolbook: 9 terms for 3x3x3.
    # But we can group: use rank 7 for a 3x3 subproblem if we find one.

    # Simplest approach: schoolbook for 3x3 core = 9 terms
    # Then tails: k=3,4 contribute (4-3)*5*4 = 20, i=3 contributes (4-3)*3*5=15, j=3,4 contribute (3)*(3)*2=18... 
    # Wait, let me recalculate.
    # If core is 3x3x3 (indices 0,1,2), tail involves:
    #   k in {3,4}: sum over i in 0..3, j in 0..4: 4*5*2 = 40 terms
    #   i=3, k in {0,1,2}: sum over j in 0..4: 3*5 = 15 terms  
    #   j in {3,4}, i in {0,1,2}, k in {0,1,2}: 3*3*2 = 18 terms
    #   i=3, j in {3,4}, k in {3,4}: 2*2 = 4 terms
    #   But we need to be careful about double counting
    # Total tail = 9*2 + 1*3*5 + 3*3*2 + 1*1*2 ... let me just enumerate

    # Better: enumerate all (i,j,k) where at least one index >= 3 (in the 3-core case)
    # Core: i<3, k<3, j<3 -> 9 terms (schoolbook for 3x3)
    # Tail: all others
    
    # Actually, let me think about this differently.
    # Total terms in schoolbook 4x5x5 = 4*5*5 = 100
    # If we replace the 3x3x3 subcore (indices i,k,j all in 0,1,2) with a rank-r decomposition:
    #   saved = 9 - r terms in the core, but the tail structure changes
    
    # For a k-core of size c:
    # Core size: c*c*c terms -> replaced by r terms
    # Tail: terms where at least one of i,k,j >= c
    #   = total - core = (n*m*p) - (c*c*c) for the rectangular case? No, not exactly.
    #   In the 4x5x5 case: total = 100
    #   Core (c=3): 3*3*3 = 27 terms saved, replaced by r
    #   Tail = 100 - 27 = 73 terms
    #   Total = r + 73
    # For c=2: core = 8, tail = 92, total = r + 92 (r=7 for Strassen -> 99, worse)
    # For c=3: total = r + 73. If r=9 (schoolbook), total = 82. If we can get r=7, total=80.
    # For c=4: core = 64, tail = 36, total = r + 36 (r=7 -> 43, which is what we had with the 4x4x4 approach... wait no)
    # The current approach uses 4x4x4 with rank 28 (7 Strassen products, each expanded to 4 terms = 28), tail = 16+20 = 36, total = 64.
    
    # Wait, let me re-examine the current program. It uses 4x4x4 Strassen (7 products * 4 = 28 terms)
    # plus tail of 16 + 20 = 36, total = 64.
    
    # If I use 3x3x3 core: I need a rank-r decomposition for 3x3x3.
    # Best known for 3x3x3 is actually not much better than schoolbook (9).
    # Let me just use 9 terms for the core.
    # Total = 9 + (100 - 27) = 9 + 73 = 82. That's worse than 64.
    
    # So the 4x4x4 approach is better. Let me stick with that but optimize the tail.
    
    # Current: 28 (4x4x4 Strassen) + 16 (j=4) + 20 (k=4) = 64
    # Can I reduce the tail? 
    # j=4 terms: for i in 0..3, k in 0..4: C[i,4] += A[i,k]*B[k,4]. That's 4*5=20 terms.
    # k=4 terms: for i in 0..3, j in 0..5... wait j goes 0..4. C[i,j] += A[i,4]*B[4,j]. That's 4*5=20 terms.
    # But the current program has:
    #   for ii in range(4): for kk in range(4): emit_term(ii, kk, 4) -> 16 terms (j=4, k in 0..3)
    #   for ii in range(4): for jj in range(5): emit_term(ii, 4, jj) -> 20 terms (k=4, j in 0..4)
    # Missing: ii in 0..3, kk=4, jj=4 -> C[i,4] += A[i,4]*B[4,4]. That's 4 more terms!
    # Also missing: ii=3 is covered since range(4) gives 0,1,2,3.
    
    # Wait, the current program is MISSING the (i, 4, 4) terms for i=0,1,2,3!
    # That's 4 missing terms. The current rank is actually 64 but it's WRONG because it's missing 4 terms.
    # Let me check: the score is 1.0870 which suggests it's valid. Let me re-examine.
    
    # Actually looking more carefully at the Strassen part: the sp list has 7 products,
    # each expanded with li in range(2), lk in range(2), lj in range(2) = 8 terms per product? No.
    # Each product has aform and bform, and the inner loops go over li,lk,lj in range(2).
    # That's 2*2*2 = 8 iterations per Strassen product, times 7 = 56 terms? No...
    # Wait, for each (prod, ccontrib) in sp (7 items), we loop li(2)*lk(2)*lj(2)=8 times,
    # creating one term each. So 7*8 = 56 terms from Strassen? That doesn't match 28.
    
    # Let me re-read. The ag/bg/cg functions map local 2x2 blocks to global 4x4 indices.
    # So for each Strassen product, we're creating 8 terms that each handle a 2x2x2 subblock.
    # Total Strassen terms = 7 * 8 = 56? That would give total = 56+36 = 92.
    
    # Hmm, but the score is 1.0870. If seed is 100 and score = 100/rank, then rank = 100/1.0870 ≈ 92.
    # So the current program has rank ≈ 92, not 64. My earlier analysis was wrong.
    
    # OK so the actual rank is about 92. To beat this, I need rank < 92.
    # Schoolbook is 100. The current approach gets 92.
    
    # Let me try a different strategy: use a rank-8 decomposition for 3x3x3 (if it exists)
    # or just optimize the 4x4x4 approach.
    
    # Actually, let me just do clean schoolbook = 100 terms, which is valid but same as seed.
    # To beat 92, I need something better.
    
    # Let me try: use 2x2x2 Strassen (rank 7) for the even-indexed sub-blocks,
    # and handle odd indices with schoolbook.
    
    # Or: try a hybrid where I use the 4x4x4 approach but fix any bugs.
    
    # Let me just implement a correct, valid decomposition. Clean schoolbook is safe.
    # But to beat the current 92, let me try the Strassen hybrid more carefully.
    
    # Actually, let me just output schoolbook (rank 100) which is definitely correct,
    # and see if I can find an optimization.
    
    def emit_term(ii, kk, jj):
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        u[ii * m + kk] = 1.0
        v[kk * p + jj] = 1.0
        w[jj * n + ii] = 1.0
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    # Schoolbook: all (i,j,k) with i in 0..3, j in 0..4, k in 0..4
    # That's 4*5*5 = 100 terms
    
    # Try to be smarter: use Strassen for the 4x4x4 part, schoolbook for tail
    # But the previous attempt gave 92 which was already good.
    # Let me replicate that exactly to make sure it's valid.
    
    U_list.clear()
    V_list.clear()
    W_list.clear()
    
    # Strassen for 4x4x4 (indices 0..3 in all dims)
    sp = [
        ({'a': [(0,1),(3,1)], 'b': [(0,1),(3,1)]}, {0:1, 3:1}),
        ({'a': [(2,1),(3,1)], 'b': [(0,1)]},     {2:1, 3:-1}),
        ({'a': [(0,1)], 'b': [(1,1),(3,-1)]},    {1:1, 3:1}),
        ({'a': [(3,1)], 'b': [(2,1),(0,-1)]},    {0:1, 2:1}),
        ({'a': [(0,1),(1,1)], 'b': [(3,1)]},     {0:-1, 1:1}),
        ({'a': [(2,1),(0,-1)], 'b': [(0,1),(1,1)]}, {3:1}),
        ({'a': [(1,1),(3,-1)], 'b': [(2,1),(3,1)]}, {0:1}),
    ]
    
    def ag(bl, li, lk):
        if bl == 0: return (li, lk)
        if bl == 1: return (li, lk+2)
        if bl == 2: return (li+2, lk)
        return (li+2, lk+2)
    
    def bg(bl, lk, lj):
        if bl == 0: return (lk, lj)
        if bl == 1: return (lk, lj+2)
        if bl == 2: return (lk+2, lj)
        return (lk+2, lj+2)
    
    def cg(bl, li, lj):
        if bl == 0: return (li, lj)
        if bl == 1: return (li, lj+2)
        if bl == 2: return (li+2, lj)
        return (li+2, lj+2)
    
    for prod, ccontrib in sp:
        aform = prod['a']
        bform = prod['b']
        for li in range(2):
            for lk in range(2):
                for lj in range(2):
                    uv = [0.0] * (n * m)
                    vv = [0.0] * (m * p)
                    wv = [0.0] * (p * n)
                    for abl, a_sign in aform:
                        gi, gk = ag(abl, li, lk)
                        uv[gi * m + gk] = a_sign
                    for bbl, b_sign in bform:
                        gk, gj = bg(bbl, lk, lj)
                        vv[gk * p + gj] = b_sign
                    for cbl, c_sign in ccontrib.items():
                        gi, gj = cg(cbl, li, lj)
                        wv[gj * n + gi] = c_sign
                    U_list.append(uv)
                    V_list.append(vv)
                    W_list.append(wv)
    
    # Tail: all (i,j,k) where k>=4 or j>=4, but NOT double-counting the core
    # Core covers i in 0..3, k in 0..3, j in 0..3
    # Tail: 
    #   k in 4, i in 0..3, j in 0..4: 4*5 = 20 terms
    #   k in 0..3, j in 4, i in 0..3: 4*4 = 16 terms  
    #   (k=4 and j=4 already included above in the k=4 line)
    # Total tail = 20 + 16 = 36
    
    for ii in range(4):
        for jj in range(5):
            # k=4
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            u[ii * m + 4] = 1.0
            v[4 * p + jj] = 1.0
            w[jj * n + ii] = 1.0
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
    
    for ii in range(4):
        for kk in range(4):
            # j=4, k in 0..3
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            u[ii * m + kk] = 1.0
            v[kk * p + 4] = 1.0
            w[4 * n + ii] = 1.0
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
