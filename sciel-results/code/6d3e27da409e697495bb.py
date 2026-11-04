import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 52
    rank = 52
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a term
    # u_indices: list of (index, coeff) for U
    # v_indices: list of (index, coeff) for V
    # w_indices: list of (index, coeff) for W
    def add_term(col, u_idx, v_idx, w_idx):
        for idx, coeff in u_idx:
            U[idx, col] += coeff
        for idx, coeff in v_idx:
            V[idx, col] += coeff
        for idx, coeff in w_idx:
            W[idx, col] += coeff

    col = 0
    
    # Part 1: 4x4x4 Strassen (49 terms)
    # Indices for 4x4x4:
    # i in 0..3, j in 0..3, k in 0..3
    # U index: i*4 + j
    # V index: j*4 + k
    # W index: k*4 + i
    
    # Define 2x2 Strassen terms (7 terms)
    # For 2x2: i in 0..1, j in 0..1, k in 0..1
    # U idx: i*2+j
    # V idx: j*2+k
    # W idx: k*2+i
    
    strassen_2x2 = []
    # M1: (A00+A11)(B00+B11) -> C00 +1, C11 +1
    strassen_2x2.append([
        [(0, 1), (3, 1)], # A00, A11
        [(0, 1), (3, 1)], # B00, B11
        [(0, 1), (3, 1)]  # C00, C11 (k=0,i=0 -> 0; k=1,i=1 -> 3)
    ])
    # M2: (A21+A22)B11 -> C10 +1
    # A21 (i=1,j=0) idx 2, A22 (i=1,j=1) idx 3
    # B11 (j=0,k=0) idx 0
    # C10 (k=0,i=1) idx 2
    strassen_2x2.append([
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(2, 1)]
    ])
    # M3: A11(B12-B22) -> C01 +1
    # A11 (i=0,j=1) idx 1
    # B12 (j=0,k=1) idx 1, B22 (j=1,k=1) idx 3
    # C01 (k=1,i=0) idx 1
    strassen_2x2.append([
        [(1, 1)],
        [(1, 1), (3, -1)],
        [(1, 1)]
    ])
    # M4: A22(B21-B11) -> C10 -1? 
    # Formula: C10 = M2 + M4. So M4 coeff +1 in C10.
    # A22 (i=1,j=1) idx 3
    # B21 (j=1,k=0) idx 2, B11 (j=0,k=0) idx 0
    # C10 (k=0,i=1) idx 2
    strassen_2x2.append([
        [(3, 1)],
        [(2, 1), (0, -1)],
        [(2, 1)]
    ])
    # M5: (A11+A12)B22 -> C01 -1? C00 +1?
    # Formula: C00 = M1 - M4 + M5 - M7. C01 = M3 + M5.
    # So M5 contributes +1 to C00 and +1 to C01.
    # A11 (i=0,j=1) idx 1, A12 (i=0,j=1? No, A12 is i=0, j=1? 
    # Wait, A is 2x2. A00, A01, A10, A11.
    # My indices:
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    # A11 is i=1, j=1? No, standard notation A11 is bottom-right.
    # In my indices:
    # A00=0, A01=1, A10=2, A11=3.
    # Strassen M5: (A11 + A12) B22.
    # Here A11 is i=1, j=1? No, A11 is i=1, j=1 is A[1,1].
    # A12 is i=1, j=2? No, 2x2 matrix only has indices 0,1.
    # Standard Strassen:
    # A = [[A11, A12], [A21, A22]]
    # My indices:
    # A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22.
    # So A11 is idx 0, A12 is idx 1, A21 is idx 2, A22 is idx 3.
    # Let's re-map.
    # M1: (A11+A22)(B11+B22) -> A[0,0]+A[1,1], B[0,0]+B[1,1].
    # M2: (A21+A22)B11 -> A[1,0]+A[1,1], B[0,0].
    # M3: A11(B12-B22) -> A[0,0], B[0,1]-B[1,1].
    # M4: A22(B21-B11) -> A[1,1], B[1,0]-B[0,0].
    # M5: (A11+A12)B22 -> A[0,0]+A[0,1], B[1,1].
    # M6: (A21-A11)(B11+B12) -> A[1,0]-A[0,0], B[0,0]+B[0,1].
    # M7: (A12-A22)(B21+B22) -> A[0,1]-A[1,1], B[1,0]+B[1,1].
    
    # Let's re-verify C formulas with this mapping.
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Re-define terms with correct indices (0-based):
    # A idx: 00=0, 01=1, 10=2, 11=3
    # B idx: 00=0, 01=1, 10=2, 11=3
    # C idx: 00=0, 01=1, 10=2, 11=3 (k*2+i)
    
    strassen_2x2 = []
    # M1
    strassen_2x2.append([[(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]])
    # M2
    strassen_2x2.append([[(2,1), (3,1)], [(0,1)], [(2,1)]])
    # M3
    strassen_2x2.append([[(0,1)], [(1,1), (3,-1)], [(1,1)]])
    # M4
    strassen_2x2.append([[(3,1)], [(2,1), (0,-1)], [(2,1)]])
    # M5
    strassen_2x2.append([[(0,1), (1,1)], [(3,1)], [(0,1), (1,1)]])
    # M6
    strassen_2x2.append([[(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]])
    # M7
    strassen_2x2.append([[(1,1), (3,-1)], [(2,1), (3,1)], [(0,-1), (3,-1)]]) # C11 gets -M7?
    # C11 = M1 - M4 + M5 - M7. So M7 coeff is -1.
    # C11 is idx 3.
    # So W for M7 should have -1 at idx 3.
    # Does M7 contribute to other C?
    # No, M7 is only in C11.
    # So W is [(3, -1)].
    
    # Correct M7
    strassen_2x2[6] = [[(1,1), (3,-1)], [(2,1), (3,1)], [(3,-1)]]

    # Now build 4x4x4
    # Outer Strassen on 2x2 blocks of 2x2.
    # Each "scalar" in outer is a 2x2 matrix.
    # We iterate through the 7 outer terms.
    # For each outer term, we have a set of (A_block, B_block) pairs.
    # Actually, the outer term is a product of sums of blocks.
    # e.g. M1_outer = (A11_block + A22_block) * (B11_block + B22_block).
    # This expands to 4 products of 2x2 blocks.
    # Each 2x2 block product is decomposed into 7 inner terms.
    # So each outer term generates 4 * 7 = 28 inner terms?
    # No, the outer term is ONE term in the rank decomposition of the block matrix multiplication.
    # The rank of 4x4x4 is 49.
    # This means we need 49 total terms.
    # The recursive structure:
    # The 7 outer terms correspond to the 7 multiplications in the block Strassen.
    # Each block multiplication (2x2 * 2x2) has rank 7.
    # So total rank is 7 * 7 = 49.
    
    # We need to map the global indices (0..15 for U, 0..15 for V, 0..15 for W)
    # to the local indices of the blocks.
    
    # Global U index: i*4 + j.
    # Block (bi, bj) where bi, bj in {0,1}.
    # i = bi*2 + ii, j = bj*2 + jj.
    # Global U idx = (bi*2+ii)*4 + (bj*2+jj) = 8*bi + 4*ii + 2*bj + jj.
    
    # Global V index: j*4 + k.
    # j = bj*2 + jj, k = bk*2 + kk.
    # Global V idx = (bj*2+jj)*4 + (bk*2+kk) = 8*bj + 4*jj + 2*bk + kk.
    
    # Global W index: k*4 + i.
    # k = bk*2 + kk, i = bi*2 + ii.
    # Global W idx = (bk*2+kk)*4 + (bi*2+ii) = 8*bk + 4*kk + 2*bi + ii.

    # The outer Strassen terms define coefficients for the blocks.
    # For each outer term t (0..6), we have:
    # A_blocks involved: dict { (bi, bj): coeff }
    # B_blocks involved: dict { (bj, bk): coeff }
    # C_blocks involved: dict { (bk, bi): coeff }
    
    # From the 2x2 Strassen definitions above:
    # M1: A: (0,0)=1, (1,1)=1. B: (0,0)=1, (1,1)=1. C: (0,0)=1, (1,1)=1.
    # M2: A: (1,0)=1, (1,1)=1. B: (0,0)=1. C: (0,1)=1. (C21? No, C21 is bi=1, bk=0. W idx k*2+i. k=0, i=1 -> 2. In block terms, bk=0, bi=1. So C block (0,1) in bk,bi order? 
    # W index is k*n+i. Block indices bk, bi.
    # C21 corresponds to k=1, i=0? No, C21 is row 2, col 1. i=1, k=0.
    # W idx = 0*2+1 = 2.
    # Block bk=0, bi=1.
    # So C block (bk=0, bi=1).
    
    # Let's define the outer terms in terms of block coefficients.
    
    outer_terms = []
    # M1
    outer_terms.append({
        'A': {(0,0): 1, (1,1): 1},
        'B': {(0,0): 1, (1,1): 1},
        'C': {(0,0): 1, (1,1): 1}
    })
    # M2
    outer_terms.append({
        'A': {(1,0): 1, (1,1): 1},
        'B': {(0,0): 1},
        'C': {(0,1): 1} # bk=0, bi=1
    })
    # M3
    outer_terms.append({
        'A': {(0,0): 1},
        'B': {(0,1): 1, (1,1): -1},
        'C': {(1,0): 1} # bk=1, bi=0 -> C12 (i=0, k=1). W idx 1*2+0=1.
    })
    # M4
    outer_terms.append({
        'A': {(1,1): 1},
        'B': {(1,0): 1, (0,0): -1},
        'C': {(0,1): 1} # bk=0, bi=1 -> C21 (i=1, k=0). W idx 0*2+1=2.
    })
    # M5
    outer_terms.append({
        'A': {(0,0): 1, (0,1): 1},
        'B': {(1,1): 1},
        'C': {(0,0): 1, (1,0): 1} # C11 (i=0,k=0) and C12 (i=0,k=1).
    })
    # M6
    outer_terms.append({
        'A': {(1,0): 1, (0,0): -1},
        'B': {(0,0): 1, (0,1): 1},
        'C': {(1,1): 1} # C22 (i=1,k=1).
    })
    # M7
    outer_terms.append({
        'A': {(0,1): 1, (1,1): -1},
        'B': {(1,0): 1, (1,1): 1},
        'C': {(1,1): -1} # C22 (i=1,k=1).
    })

    # Now, for each outer term, expand into 7 inner terms.
    # The inner terms are the 7 Strassen terms for 2x2.
    # The inner term r has local U, V, W coefficients.
    # We need to map local indices to global.
    
    for t in range(7):
        outer = outer_terms[t]
        # For each combination of A block and B block in the outer term?
        # No, the outer term is a product of sums.
        # e.g. (A00 + A11) * (B00 + B11).
        # This is A00*B00 + A00*B11 + A11*B00 + A11*B11.
        # Each of these 4 products is a 2x2 matrix multiplication.
        # Each 2x2 matrix multiplication is decomposed into 7 inner terms.
        # So we iterate over the 4 pairs of blocks.
        
        a_blocks = outer['A'].items()
        b_blocks = outer['B'].items()
        c_blocks = outer['C']
        
        # Wait, the C coefficients are for the final assembly.
        # The inner decomposition computes the product of the two 2x2 blocks.
        # The result is added to the C blocks with the coefficients from 'C'.
        
        for (bi_a, bj_a), coeff_a in a_blocks:
            for (bj_b, bk_b), coeff_b in b_blocks:
                # Check if bj_a == bj_b? 
                # A block (bi_a, bj_a) and B block (bj_b, bk_b).
                # They must match on the middle dimension.
                # If bj_a != bj_b, the product is zero.
                if bj_a != bj_b:
                    continue
                
                # This specific product contributes to C block (bk_b, bi_a).
                # The coefficient for this C block in the outer term is c_blocks.get((bk_b, bi_a), 0).
                # If it's 0, this product doesn't contribute to the final C?
                # No, the outer term defines which C blocks are affected.
                # If c_blocks doesn't have (bk_b, bi_a), it means this specific block product
                # is not part of the outer term's contribution to C?
                # Actually, the outer term IS the contribution.
                # So if (bk_b, bi_a) is not in c_blocks, it means this product is 0?
                # No, the outer term is defined by the Strassen formulas.
                # The formulas ensure that only certain block products are formed.
                # e.g. M1: A00*B00, A00*B11, A11*B00, A11*B11.
                # All these contribute to C00 and C11.
                # So c_blocks should have entries for all resulting (bk, bi).
                
                coeff_c = c_blocks.get((bk_b, bi_a), 0)
                if coeff_c == 0:
                    continue
                
                # Now decompose the 2x2 block multiplication (bi_a, bj_a) * (bj_b, bk_b)
                # using the 7 inner Strassen terms.
                # The result is added to global U, V, W.
                
                for r in range(7):
                    inner_u = strassen_2x2[r][0]
                    inner_v = strassen_2x2[r][1]
                    inner_w = strassen_2x2[r][2]
                    
                    u_idx = []
                    for (ii_jj, coeff_u) in inner_u:
                        ii, jj = divmod(ii_jj, 2)
                        # Global U idx: 8*bi + 4*ii + 2*bj + jj
                        g_idx = 8*bi_a + 4*ii + 2*bj_a + jj
                        u_idx.append((g_idx, coeff_a * coeff_u))
                    
                    v_idx = []
                    for (jj_kk, coeff_v) in inner_v:
                        jj, kk = divmod(jj_kk, 2)
                        # Global V idx: 8*bj + 4*jj + 2*bk + kk
                        g_idx = 8*bj_b + 4*jj + 2*bk_b + kk
                        v_idx.append((g_idx, coeff_b * coeff_v))
                    
                    w_idx = []
                    for (kk_ii, coeff_w) in inner_w:
                        kk, ii = divmod(kk_ii, 2)
                        # Global W idx: 8*bk + 4*kk + 2*bi + ii
                        g_idx = 8*bk_b + 4*kk + 2*bi_a + ii
                        w_idx.append((g_idx, coeff_c * coeff_w))
                    
                    add_term(col, u_idx, v_idx, w_idx)
                    col += 1
    
    # Part 2: 4x4x2 (2 terms)
    # i in 0..3, j in 0..3, k in 4..5
    # U idx: i*4 + j
    # V idx: j*6 + k
    # W idx: k*4 + i
    
    # Term 1: k=4
    # U: all 1
    # V: all 1 for j=0..3, k=4
    # W: 1 for k=4, all i
    u_idx = [(i*4+j, 1) for i in range(4) for j in range(4)]
    v_idx = [(j*6+4, 1) for j in range(4)]
    w_idx = [(4*4+i, 1) for i in range(4)]
    add_term(col, u_idx, v_idx, w_idx)
    col += 1
    
    # Term 2: k=5
    u_idx = [(i*4+j, 1) for i in range(4) for j in range(4)]
    v_idx = [(j*6+5, 1) for j in range(4)]
    w_idx = [(5*4+i, 1) for i in range(4)]
    add_term(col, u_idx, v_idx, w_idx)
    col += 1
    
    # Part 3: 4x1x6 (1 term)
    # i in 0..3, j=4, k in 0..5
    # U idx: i*4 + 4
    # V idx: 4*6 + k
    # W idx: k*4 + i
    
    u_idx = [(i*4+4, 1) for i in range(4)]
    v_idx = [(4*6+k, 1) for k in range(6)]
    w_idx = [(k*4+i, 1) for k in range(6) for i in range(4)]
    # Wait, W index is k*n+i.
    # For a fixed term, W can have multiple non-zeros.
    # Here, the term is A_{i,4} B_{4,k}.
    # This contributes to C_{i,k}.
    # So W_{k, i} = 1 for all k, i.
    # So w_idx should have 1 for all (k, i).
    add_term(col, u_idx, v_idx, w_idx)
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

