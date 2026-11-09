"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        if not U:
            for _ in range(n*m): U.append(0)
            for _ in range(m*p): V.append(0)
            for _ in range(p*n): W.append(0)
        U.append(coeff if u_idx is not None else 0)
        V.append(coeff if v_idx is not None else 0)
        W.append(coeff if w_idx is not None else 0)
        # This approach is messy for accumulating coefficients.
        # Better to build lists of tuples.

    terms = []
    
    # Helper to add a term (u_idx, v_idx, w_idx, coeff)
    def add(u, v, w, c=1):
        terms.append((u, v, w, c))
        
    # 1. A11 * B1 (2x2 x 8)
    # A11 is A[0:2, 0:2]
    # B1 is B[0:2, 0:8]
    # Split B1 into 4 blocks of 2 columns: (0,1), (2,3), (4,5), (6,7)
    # For each block, use Strassen 2x2x2
    
    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map local indices to global U, V, W
    # Local A indices: 0->a, 1->b, 2->c, 3->d
    # Local B indices: 0->e, 1->f, 2->g, 3->h
    # Local C indices: 0->C11, 1->C12, 2->C21, 3->C22
    
    # Global A indices for A11:
    # a: A[0,0] -> 0
    # b: A[0,1] -> 1
    # c: A[1,0] -> 3
    # d: A[1,1] -> 4
    
    # Global B indices for B1 (rows 0,1):
    # e: B[0,k]
    # f: B[0,k+1]
    # g: B[1,k]
    # h: B[1,k+1]
    
    # Global C indices for C1 (rows 0,1):
    # C11: C[0,k] -> 3k
    # C12: C[0,k+1] -> 3(k+1)
    # C21: C[1,k] -> 3k+1
    # C22: C[1,k+1] -> 3(k+1)+1
    
    for block_idx in range(4):
        k_start = block_idx * 2
        # a,b,c,d global U indices
        ua, ub, uc, ud = 0, 1, 3, 4
        
        # e,f,g,h global V indices
        ve = k_start      # B[0, k]
        vf = k_start + 1  # B[0, k+1]
        vg = 8 + k_start      # B[1, k]
        vh = 8 + k_start + 1  # B[1, k+1]
        
        # C11, C12, C21, C22 global W indices
        wc11 = 3 * k_start
        wc12 = 3 * (k_start + 1)
        wc21 = 3 * k_start + 1
        wc22 = 3 * (k_start + 1) + 1
        
        # M1 = (a+d)(e+h)
        # This is not a single product, it's a product of sums.
        # In tensor decomposition, we need to expand or use the rank 7 decomposition directly.
        # The rank 7 decomposition has 7 "multiplications" which are products of linear forms.
        # We need to add 7 terms to our list, but each term in the list is a single product U*V*W.
        # Wait, the Strassen algorithm reduces the number of multiplications, but the multiplications themselves are of linear combinations.
        # To represent this in the U,V,W format, we must expand the linear combinations into the U and V vectors.
        # A term like (a+d)(e+h) corresponds to:
        # (1*a + 1*d) * (1*e + 1*h)
        # This expands to a*e + a*h + d*e + d*h.
        # This would increase the rank.
        # The "rank" of the Strassen algorithm is 7 because we define 7 intermediate variables M1..M7.
        # But in the final U,V,W representation, we don't have intermediate variables.
        # We must express the final C entries as sums of products of A and B entries.
        # Strassen says C11 = M1 + M4 - M5 + M7.
        # M1 = (a+d)(e+h) = ae + ah + de + dh.
        # So C11 contains ae, ah, de, dh from M1.
        # This effectively expands back to the schoolbook terms?
        # No, Strassen reduces the number of *scalar multiplications* if we consider the linear combinations as free.
        # But in the tensor decomposition model (rank), we count the number of bilinear terms $u_i v_i$.
        # The rank of the tensor is the minimum number of such terms.
        # Strassen's algorithm shows that the rank is at most 7 for 2x2x2.
        # How is this represented?
        # The 7 multiplications are of the form $L_A \cdot L_B$.
        # To convert this to the standard basis (monomials $A_{ij} B_{kl}$), we expand.
        # But the rank is defined on the monomial basis?
        # No, the rank is the minimum $r$ such that $T = \sum_{k=1}^r u_k \otimes v_k \otimes w_k$.
        # The vectors $u_k, v_k$ can be any linear combinations.
        # So, if we have a term $(a+d)(e+h)$, we can represent it as:
        # $u = a+d$, $v = e+h$.
        # This is ONE term in the rank decomposition!
        # We do NOT need to expand it into 4 monomials.
        # The U, V, W matrices allow arbitrary linear combinations.
        
        # So, for M1:
        # U vector: has 1 at a, 1 at d.
        # V vector: has 1 at e, 1 at h.
        # W vector: contributes to C11, C22.
        # Specifically, M1 contributes +1 to C11 and +1 to C22.
        
        # Let's add the 7 Strassen terms for this block.
        
        # M1 = (a+d)(e+h)
        u1 = {ua: 1, ud: 1}
        v1 = {ve: 1, vh: 1}
        w1 = {wc11: 1, wc22: 1}
        add_term_from_dicts(u1, v1, w1)
        
        # M2 = (c+d)e
        u2 = {uc: 1, ud: 1}
        v2 = {ve: 1}
        w2 = {wc21: 1, wc22: -1}
        add_term_from_dicts(u2, v2, w2)
        
        # M3 = a(f-h)
        u3 = {ua: 1}
        v3 = {vf: 1, vh: -1}
        w3 = {wc12: 1, wc22: 1}
        add_term_from_dicts(u3, v3, w3)
        
        # M4 = d(g-e)
        u4 = {ud: 1}
        v4 = {vg: 1, ve: -1}
        w4 = {wc11: 1, wc21: 1}
        add_term_from_dicts(u4, v4, w4)
        
        # M5 = (a+b)h
        u5 = {ua: 1, ub: 1}
        v5 = {vh: 1}
        w5 = {wc11: -1, wc12: 1}
        add_term_from_dicts(u5, v5, w5)
        
        # M6 = (c-a)(e+f)
        u6 = {uc: 1, ua: -1}
        v6 = {ve: 1, vf: 1}
        w6 = {wc22: 1}
        add_term_from_dicts(u6, v6, w6)
        
        # M7 = (b-d)(g+h)
        u7 = {ub: 1, ud: -1}
        v7 = {vg: 1, vh: 1}
        w7 = {wc11: 1}
        add_term_from_dicts(u7, v7, w7)

    # 2. A12 * B2 (2x1 x 8)
    # A12: A[0,2] (idx 2), A[1,2] (idx 5)
    # B2: B[2,k] (idx 16+k)
    # C1: C[0,k] (idx 3k), C[1,k] (idx 3k+1)
    for k in range(8):
        # A[0,2] * B[2,k] -> C[0,k]
        add(2, 16+k, 3*k, 1)
        # A[1,2] * B[2,k] -> C[1,k]
        add(5, 16+k, 3*k+1, 1)
        
    # 3. A21 * B1 (1x2 x 8)
    # A21: A[2,0] (idx 6), A[2,1] (idx 7)
    # B1: B[0,k] (idx k), B[1,k] (idx 8+k)
    # C2: C[2,k] (idx 3k+2)
    for k in range(8):
        # A[2,0] * B[0,k] -> C[2,k]
        add(6, k, 3*k+2, 1)
        # A[2,1] * B[1,k] -> C[2,k]
        add(7, 8+k, 3*k+2, 1)
        
    # 4. A22 * B2 (1x1 x 8)
    # A22: A[2,2] (idx 8)
    # B2: B[2,k] (idx 16+k)
    # C2: C[2,k] (idx 3k+2)
    for k in range(8):
        add(8, 16+k, 3*k+2, 1)
        
    # Build U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
            
    return {"U": U, "V": V, "W": W}

def add_term_from_dicts(u, v, w):
    # This function is called inside construct, so it needs access to 'terms'
    # I will refactor to pass terms or use a closure.
    pass
# EVOLVE-BLOCK-END
