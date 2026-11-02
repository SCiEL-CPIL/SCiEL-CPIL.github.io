import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank = 52
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We will use a dictionary to accumulate coefficients for each (u_row, v_row, w_row)
    # Then assign them to columns.
    # But we need to ensure we don't exceed rank.
    # Since we expect exactly 52 unique terms, we can just list them.
    
    terms = []
    
    # Part 1: A_L B_L (Rank 28)
    # A_L: i in 0..3, j in 0..3
    # B_L: j in 0..3, k in 0..3
    # C: i in 0..3, k in 0..3
    
    # Strassen M1..M7
    # A' and B' definitions
    # M1: A' = A11 + A22, B' = B11 + B22
    # M2: A' = A21 + A22, B' = B11
    # M3: A' = A11, B' = B12 - B22
    # M4: A' = A22, B' = B21 - B11
    # M5: A' = A11 + A12, B' = B22
    # M6: A' = A21 - A11, B' = B11 + B12
    # M7: A' = A12 - A22, B' = B21 + B22
    
    # A blocks: 11, 12, 21, 22
    # 11: i 0,1 j 0,1
    # 12: i 0,1 j 2,3
    # 21: i 2,3 j 0,1
    # 22: i 2,3 j 2,3
    
    # B blocks: 11, 12, 21, 22
    # 11: j 0,1 k 0,1
    # 12: j 0,1 k 2,3
    # 21: j 2,3 k 0,1
    # 22: j 2,3 k 2,3
    
    # Target C blocks: 11, 12, 21, 22
    # 11: i 0,1 k 0,1
    # 12: i 0,1 k 2,3
    # 21: i 2,3 k 0,1
    # 22: i 2,3 k 2,3
    
    # Coefficients for A'
    # Each Mq has a set of (block, coeff)
    A_defs = [
        [(11, 1), (22, 1)], # M1
        [(21, 1), (22, 1)], # M2
        [(11, 1)],          # M3
        [(22, 1)],          # M4
        [(11, 1), (12, 1)], # M5
        [(21, 1), (11, -1)],# M6
        [(12, 1), (22, -1)] # M7
    ]
    
    B_defs = [
        [(11, 1), (22, 1)], # M1
        [(11, 1)],          # M2
        [(12, 1), (22, -1)],# M3
        [(21, 1), (11, -1)],# M4
        [(22, 1)],          # M5
        [(11, 1), (12, 1)], # M6
        [(21, 1), (22, 1)]  # M7
    ]
    
    # Target C blocks and coeffs for each Mq
    # M1 -> C11(1), C22(1)
    # M2 -> C21(1), C22(-1)
    # M3 -> C12(1), C22(1)
    # M4 -> C11(1), C21(1)
    # M5 -> C11(-1), C12(1)
    # M6 -> C22(1)
    # M7 -> C11(1)
    
    C_defs = [
        [(11, 1), (22, 1)],
        [(21, 1), (22, -1)],
        [(12, 1), (22, 1)],
        [(11, 1), (21, 1)],
        [(11, -1), (12, 1)],
        [(22, 1)],
        [(11, 1)]
    ]
    
    for q in range(7):
        for r in range(2):
            for c in range(2):
                for d in range(2):
                    # A' rc
                    for ab, ca in A_defs[q]:
                        # ab is block, ca is coeff
                        # r, c are local indices
                        # Global i, j
                        bx, by = divmod(ab - 1, 2) # ab=11 -> 0,0; 12 -> 0,1; 21 -> 1,0; 22 -> 1,1
                        gi = bx * 2 + r
                        gj = by * 2 + c
                        u_row = gi * 5 + gj
                        
                        for ef, cb in B_defs[q]:
                            # ef is block, cb is coeff
                            # c, d are local indices
                            # Global j, k
                            # B block ef: row e, col f
                            ex, ey = divmod(ef - 1, 2)
                            gj2 = ex * 2 + c
                            gk = ey * 2 + d
                            # Check j consistency?
                            # A' col c and B' row c must match global j.
                            # A' col c comes from block ab col by. gj = by*2+c.
                            # B' row c comes from block ef row ex. gj2 = ex*2+c.
                            # They are consistent if by == ex.
                            # In Strassen, A' and B' are multiplied.
                            # A' is 2x2, B' is 2x2.
                            # The index c is the shared index.
                            # So we just use gj from A and gk from B.
                            # But we must ensure that the global j is the same.
                            # Actually, the term is A[gi, gj] * B[gj, gk].
                            # So gj must be the same.
                            # If by != ex, then this combination is invalid?
                            # No, A' is a matrix. A'_{rc} = sum ca * A_{ab, rc}.
                            # B' is a matrix. B'_{cd} = sum cb * B_{ef, cd}.
                            # The product is sum_{rcd} A'_{rc} B'_{cd}.
                            # This expands to sum_{abcd} ca cb A_{ab, rc} B_{ef, cd}.
                            # The index c is the same in A and B.
                            # So gj from A (by*2+c) and gj from B (ex*2+c) must be equal.
                            # This implies by == ex.
                            # If by != ex, the term is 0?
                            # No, the matrices A' and B' are formed by adding blocks.
                            # If we add A11 and A22, A'_{rc} has contributions from A11 and A22.
                            # A11 has j in 0,1. A22 has j in 2,3.
                            # So for a fixed r,c, A'_{rc} is a sum of terms with different j.
                            # Similarly for B'.
                            # So we just sum over all valid (ab, ef) pairs.
                            # The global j is determined by ab and ef.
                            # If by != ex, then the j indices are different.
                            # But the multiplication A*B requires matching j.
                            # So if by != ex, this specific combination of blocks does not contribute to the same j.
                            # Wait.
                            # A'_{rc} = sum_{ab} ca_{ab} A_{ab, rc}.
                            # B'_{cd} = sum_{ef} cb_{ef} B_{ef, cd}.
                            # Product = sum_{ab, ef} ca_{ab} cb_{ef} A_{ab, rc} B_{ef, cd}.
                            # The term A_{ab, rc} has global j = 2(by-1)+c.
                            # The term B_{ef, cd} has global j = 2(ex-1)+c.
                            # For the product to be non-zero in the tensor, the j indices must match.
                            # So we only keep terms where 2(by-1)+c == 2(ex-1)+c, i.e., by == ex.
                            
                            if by != ex:
                                continue
                                
                            v_row = gj * 5 + gk
                            
                            for xy, cc in C_defs[q]:
                                # xy is C block, cc is coeff
                                # Global i, k
                                cx, cy = divmod(xy - 1, 2)
                                # i from C block: cx*2+r
                                # k from C block: cy*2+d
                                gi2 = cx * 2 + r
                                gk2 = cy * 2 + d
                                
                                # Check i consistency: gi from A and gi2 from C must match.
                                # gi = bx*2+r. gi2 = cx*2+r.
                                # So bx must equal cx.
                                if bx != cx:
                                    continue
                                    
                                # Check k consistency: gk from B and gk2 from C must match.
                                # gk = ey*2+d. gk2 = cy*2+d.
                                # So ey must equal cy.
                                if ey != cy:
                                    continue
                                    
                                w_row = gk * 4 + gi
                                
                                coeff = ca * cb * cc
                                terms.append((u_row, v_row, w_row, coeff))
    
    # Part 2: A_L B_T (16 terms)
    # A_L: i 0..3, j 0..3
    # B_T: j 0..3, k 4
    # C: i 0..3, k 4
    for i in range(4):
        for j in range(4):
            u_row = i * 5 + j
            v_row = j * 5 + 4
            w_row = 4 * 4 + i
            terms.append((u_row, v_row, w_row, 1))
            
    # Part 3: A_R B_B (4 terms)
    # A_R: i 0..3, j 4
    # B_B: j 4, k 0..3
    # C: i 0..3, k 0..3
    for i in range(4):
        for k in range(4):
            u_row = i * 5 + 4
            v_row = 4 * 5 + k
            w_row = k * 4 + i
            terms.append((u_row, v_row, w_row, 1))
            
    # Part 4: A_R B_BR (4 terms)
    # A_R: i 0..3, j 4
    # B_BR: j 4, k 4
    # C: i 0..3, k 4
    for i in range(4):
        u_row = i * 5 + 4
        v_row = 4 * 5 + 4
        w_row = 4 * 4 + i
        terms.append((u_row, v_row, w_row, 1))
        
    # Aggregate terms
    from collections import defaultdict
    agg = defaultdict(float)
    for u, v, w, c in terms:
        agg[(u, v, w)] += c
        
    # Assign to U, V, W
    col = 0
    for (u, v, w), c in agg.items():
        U[u][col] = c
        V[v][col] = c
        W[w][col] = c
        col += 1
        
    # If col < rank, we have extra zeros.
    # If col > rank, we have a problem.
    # We expect 52.
    
    return {"U": U, "V": V, "W": W}
