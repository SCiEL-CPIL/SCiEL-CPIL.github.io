"""Matrix multiplication tensor decomposition for 2x5 * 5x6 -> 2x6."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS
    
    # We use a block decomposition.
    # A is (2, 5), B is (5, 6).
    # Split A into A1 (2, 2), A2 (2, 2), A3 (2, 1).
    # Split B into B1 (2, 2), B2 (2, 2), B3 (2, 2).
    # C = A1 B1 + A2 B2 + A3 B3.
    
    # Rank 2x2x2 is 7 (Strassen).
    # Rank 2x1x2 is 4 (schoolbook).
    # Total rank = 6*7 + 3*4 = 42 + 12 = 54.
    
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 - M4 + M5 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M7
    
    # Indices for A block (rows i, cols j)
    # A1: i in {0,1}, j in {0,1}
    # A2: i in {0,1}, j in {2,3}
    # A3: i in {0,1}, j in {4}
    
    # Indices for B block (rows j, cols k)
    # B1: j in {0,1}, k in {0,1}
    # B2: j in {2,3}, k in {2,3}
    # B3: j in {4,5}, k in {4,5}
    
    # Note: A and B share the index j.
    # A1 uses j=0,1. B1 uses j=0,1.
    # A2 uses j=2,3. B2 uses j=2,3.
    # A3 uses j=4.   B3 uses j=4,5.
    
    # We process each (j_block, k_block) pair? No, the decomposition is per (j_range, k_range) for the product term.
    # The sum is over j. So we decompose the term A_j B_j for each j-range independently?
    # Yes, because the tensor is linear in A and B.
    # Term 1: A[:, 0:2] * B[0:2, :] -> A1 * B1 (2x2 * 2x6)
    #   Sub-term 1.1: A1[:, 0:2] * B1[0:2, 0:2] (2x2 * 2x2) -> C[:, 0:2]
    #   Sub-term 1.2: A1[:, 0:2] * B1[0:2, 2:4] (2x2 * 2x2) -> C[:, 2:4]
    #   Sub-term 1.3: A1[:, 0:2] * B1[0:2, 4:6] (2x2 * 2x2) -> C[:, 4:6]
    
    # Term 2: A[:, 2:4] * B[2:4, :] -> A2 * B2 (2x2 * 2x6)
    #   Sub-term 2.1: A2[:, 0:2] * B2[0:2, 0:2] -> C[:, 0:2]
    #   Sub-term 2.2: A2[:, 0:2] * B2[0:2, 2:4] -> C[:, 2:4]
    #   Sub-term 2.3: A2[:, 0:2] * B2[0:2, 4:6] -> C[:, 4:6]
    
    # Term 3: A[:, 4:5] * B[4:6, :] -> A3 * B3 (2x1 * 2x2? No, B is 5x6. Rows 4,5 of B are 2x6)
    # A3 is 2x1. B3 is 2x6.
    #   Sub-term 3.1: A3 * B3[:, 0:2] (2x1 * 2x2) -> C[:, 0:2]
    #   Sub-term 3.2: A3 * B3[:, 2:4] (2x1 * 2x2) -> C[:, 2:4]
    #   Sub-term 3.3: A3 * B3[:, 4:6] (2x1 * 2x2) -> C[:, 4:6]
    
    def add_strassen(i_start, j_start, k_start, col_offset):
        # A block is at rows i_start..i_start+1, cols j_start..j_start+1
        # B block is at rows j_start..j_start+1, cols k_start..k_start+1
        # Output C block at rows i_start..i_start+1, cols k_start..k_start+1
        
        # M1 = (a+d)(e+h)
        # a: A[i0, j0], d: A[i1, j1]
        # e: B[j0, k0], h: B[j1, k1]
        
        # U vector for M1: A parts: a, d. B parts: none (scalar mult). W parts: none.
        # Wait, the decomposition is U * V * W.
        # M1 term in tensor: sum_{i,k} (A_{i0,j0} + A_{i1,j1}) (B_{j0,k0} + B_{j1,k1}) C_{i,k}
        # = A_{i0,j0} B_{j0,k0} C_{i,k} + A_{i0,j0} B_{j1,k1} C_{i,k} + A_{i1,j1} B_{j0,k0} C_{i,k} + A_{i1,j1} B_{j1,k1} C_{i,k}
        
        # This corresponds to 4 rank-1 terms? No, Strassen combines them.
        # The rank 7 decomposition uses 7 terms.
        # Let's just add the 7 terms.
        
        # Term 1: (a+d)(e+h)
        # U: A[i0, j0] += 1, A[i1, j1] += 1
        # V: B[j0, k0] += 1, B[j1, k1] += 1
        # W: C[i0, k0] += 1, C[i0, k1] += 1, C[i1, k0] += 1, C[i1, k1] += 1
        
        # We need to be careful with signs and indices.
        
        # Let's define indices explicitly.
        # A indices: (i, j)
        # B indices: (j, k)
        # C indices: (i, k) -> W[k*n + i]
        
        a_idx = i_start * m + j_start      # A[i0, j0]
        b_idx = i_start * m + (j_start+1)  # A[i0, j1]
        c_idx = (i_start+1) * m + j_start  # A[i1, j0]
        d_idx = (i_start+1) * m + (j_start+1) # A[i1, j1]
        
        e_idx = j_start * p + k_start      # B[j0, k0]
        f_idx = j_start * p + (k_start+1)  # B[j0, k1]
        g_idx = (j_start+1) * p + k_start  # B[j1, k0]
        h_idx = (j_start+1) * p + (k_start+1) # B[j1, k1]
        
        # W indices
        # C[i, k] -> W[k*n + i]
        c00_idx = k_start * n + i_start
        c01_idx = (k_start+1) * n + i_start
        c10_idx = k_start * n + (i_start+1)
        c11_idx = (k_start+1) * n + (i_start+1)
        
        # M1: (a+d)(e+h)
        U[a_idx, col_offset] += 1
        U[d_idx, col_offset] += 1
        V[e_idx, col_offset] += 1
        V[h_idx, col_offset] += 1
        W[c00_idx, col_offset] += 1
        W[c01_idx, col_offset] += 1
        W[c10_idx, col_offset] += 1
        W[c11_idx, col_offset] += 1
        col_offset += 1
        
        # M2: (c+d)e
        U[c_idx, col_offset] += 1
        U[d_idx, col_offset] += 1
        V[e_idx, col_offset] += 1
        W[c10_idx, col_offset] += 1
        W[c11_idx, col_offset] += 1
        col_offset += 1
        
        # M3: a(f-h)
        U[a_idx, col_offset] += 1
        V[f_idx, col_offset] += 1
        V[h_idx, col_offset] -= 1
        W[c00_idx, col_offset] += 1
        W[c01_idx, col_offset] += 1
        col_offset += 1
        
        # M4: d(g-e)
        U[d_idx, col_offset] += 1
        V[g_idx, col_offset] += 1
        V[e_idx, col_offset] -= 1
        W[c00_idx, col_offset] -= 1
        W[c11_idx, col_offset] -= 1
        col_offset += 1
        
        # M5: (a+b)h
        U[a_idx, col_offset] += 1
        U[b_idx, col_offset] += 1
        V[h_idx, col_offset] += 1
        W[c00_idx, col_offset] += 1
        W[c01_idx, col_offset] += 1
        col_offset += 1
        
        # M6: (c-a)(e+f)
        U[c_idx, col_offset] += 1
        U[a_idx, col_offset] -= 1
        V[e_idx, col_offset] += 1
        V[f_idx, col_offset] += 1
        W[c10_idx, col_offset] += 1
        W[c11_idx, col_offset] += 1
        col_offset += 1
        
        # M7: (b-d)(g+h)
        U[b_idx, col_offset] += 1
        U[d_idx, col_offset] -= 1
        V[g_idx, col_offset] += 1
        V[h_idx, col_offset] += 1
        W[c00_idx, col_offset] -= 1
        W[c11_idx] -= 1 # Typo in previous thought? W[c11_idx, col_offset] -= 1
        col_offset += 1
        
        # Check C formulas again:
        # C11 = M1 - M4 + M5 + M6
        # M1: +1 at c00
        # M4: -1 at c00
        # M5: +1 at c00
        # M6: +1 at c10? No, M6 is (c-a)(e+f).
        # Let's re-verify Strassen formulas.
        
        # Standard Strassen:
        # M1 = (a + d) * (e + h)
        # M2 = (c + d) * e
        # M3 = a * (f - h)
        # M4 = d * (g - e)
        # M5 = (a + b) * h
        # M6 = (c - a) * (e + f)
        # M7 = (b - d) * (g + h)
        
        # C11 = M1 - M4 + M5 + M6
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 - M7
        
        # My W updates:
        # M1: c00, c01, c10, c11 all +1.
        # M2: c10, c11 all +1.
        # M3: c00, c01 all +1.
        # M4: c00 -1, c11 -1.
        # M5: c00, c01 all +1.
        # M6: c10, c11 all +1.
        # M7: c00 -1, c11 -1.
        
        # Let's check C11 (c00):
        # M1(+1) + M4(-1) + M5(+1) + M6(0) = 1. Correct.
        
        # Check C12 (c01):
        # M1(+1) + M4(0) + M5(+1) + M6(0) = 2? No.
        # Formula: C12 = M3 + M5.
        # My updates for c01:
        # M1: +1
        # M3: +1
        # M5: +1
        # Sum = 3. Wrong.
        
        # I must have messed up the W updates.
        # Let's redo W updates carefully.
        
        # M1: (a+d)(e+h) -> contributes to all C_ik.
        # So W[c00] += 1, W[c01] += 1, W[c10] += 1, W[c11] += 1.
        
        # M2: (c+d)e -> B is e (j0, k0). So only k=0 columns.
        # A is c, d (i1). So only i=1 rows.
        # Contributes to C10, C11? No, e is B[j0, k0]. So k=0.
        # So C10, C11? No, k is fixed to 0. So C[i, 0].
        # i=1. So C10.
        # Wait, B is (j, k). e is B[j0, k0].
        # Product (c+d) * e is scalar.
        # It multiplies A[i, j] and B[j, k].
        # Here A part is c (i1, j0) and d (i1, j1).
        # B part is e (j0, k0).
        # The term is A[i, j] B[j, k].
        # The "scalar" M2 is added to C[i, k].
        # Which i, k?
        # The tensor decomposition adds to C[i, k] the product of the U and V coefficients.
        # If U has 1 at c_idx (i1, j0) and d_idx (i1, j1), and V has 1 at e_idx (j0, k0).
        # Then the term is (A[i1, j0] + A[i1, j1]) * B[j0, k0].
        # This should be added to C[i1, k0]. i.e., C10.
        # Does it add to C11? No, because V only has k0.
        # So W[c10] += 1. W[c11] += 0.
        # My previous update: W[c10] += 1, W[c11] += 1. This is wrong.
        
        # Let's re-evaluate all W updates.
        
        # M1: U: a(i0,j0), d(i1,j1). V: e(j0,k0), h(j1,k1).
        # Product: (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # = A[i0,j0]B[j0,k0] + A[i0,j0]B[j1,k1] + A[i1,j1]B[j0,k0] + A[i1,j1]B[j1,k1]
        # This contributes to:
        # C[i0, k0] (from first term)
        # C[i0, k1] (from second term? No, A[i0,j0]B[j1,k1]. j0 != j1. This is not a valid term in simple multiplication unless we sum over j. But here we are constructing the decomposition. The "rank 1" term is (u1 . A) (v1 . B) w1^T.
        # If u1 = e_a + e_d, v1 = e_e + e_h, w1 = e_c00 + e_c01 + e_c10 + e_c11.
        # Then the term is (A[a] + A[d]) (B[e] + B[h]) (C[c00] + C[c01] + C[c10] + C[c11]).
        # This matches the expansion. So W updates for M1 are correct: all 4 are +1.
        
        # M2: (c+d)e.
        # u2 = e_c + e_d. v2 = e_e.
        # Term: (A[c] + A[d]) B[e].
        # This is a scalar. It should be added to C[i, k] where i, k match the indices in A, B.
        # A[c] is A[i1, j0]. A[d] is A[i1, j1]. B[e] is B[j0, k0].
        # The product is not a single matrix element of C.
        # Wait. Strassen works because we sum the terms to get C.
        # The term M2 is added to C10 and C11?
        # Let's check C21 = M2 + M4.
        # C21 is C[i1, k0].
        # If M2 contributes to C10, then W[c10] += 1.
        # Does M2 contribute to C11? No.
        # So for M2: W[c10] += 1.
        
        # Let's check M4: d(g-e).
        # u4 = e_d. v4 = e_g - e_e.
        # Term: A[d] (B[g] - B[e]).
        # A[d] is A[i1, j1]. B[g] is B[j1, k0]. B[e] is B[j0, k0].
        # This contributes to C[i1, k0].
        # So W[c10] += 1?
        # Formula: C21 = M2 + M4.
        # If M2 adds 1 to C10, and M4 adds 1 to C10, sum is 2. Wrong.
        # It should be 1.
        
        # Let's look at the expansion of C21 = A[i1, j0]B[j0, k0] + A[i1, j1]B[j1, k0] = c e + d g.
        # M2 = (c+d)e = ce + de.
        # M4 = d(g-e) = dg - de.
        # M2 + M4 = ce + dg. Correct.
        # So M2 contributes ce + de.
        # M4 contributes dg - de.
        
        # In the tensor decomposition, the term for M2 is (c+d)e.
        # This term is added to the sum for C.
        # How does it map to W?
        # The term is (A[c] + A[d]) B[e].
        # This is a scalar value S.
        # We add S * W_vector to C.
        # Which elements of C does S affect?
        # S = ce + de.
        # ce is part of C21. de is NOT part of C21 (it's "extra").
        # But in the sum M2+M4, the de cancels with -de from M4.
        # So the "de" part of M2 must be placed somewhere that cancels with M4.
        # M4 is dg - de.
        # So "de" in M2 and "-de" in M4 must land on the same C element.
        # Which C element?
        # ce is in C21. dg is in C21.
        # So de must also be in C21?
        # If so, M2 adds to C21: coefficient 1 for ce, 1 for de.
        # M4 adds to C21: coefficient 1 for dg, -1 for de.
        # Sum: ce + dg. Correct.
        # So both M2 and M4 contribute ONLY to C21 (c10).
        
        # So for M2: W[c10] += 1.
        # For M4: W[c10] += 1.
        
        # Let's check M3: a(f-h).
        # u3 = e_a. v3 = e_f - e_h.
        # Term: A[a] (B[f] - B[h]).
        # A[a] = A[i0, j0]. B[f] = B[j0, k1]. B[h] = B[j1, k1].
        # Term = af - ah.
        # C12 = af. C02? No.
        # C12 = M3 + M5.
        # M3 = af - ah.
        # M5 = (a+b)h = ah + bh.
        # M3 + M5 = af + bh.
        # But C12 = af. Where did bh go?
        # Wait. C12 = A[i0, j0]B[j0, k1] + A[i0, j1]B[j1, k1] = af + bh.
        # So C12 IS af + bh.
        # So M3 and M5 contribute to C12 (c01).
        # M3: af - ah.
        # M5: ah + bh.
        # Sum: af + bh. Correct.
        # So W[c01] += 1 for M3?
        # M3 term is af - ah.
        # It contributes +1 to af (part of C12) and -1 to ah (part of C12).
        # So W[c01] += 1?
        # If W[c01] += 1, then the term is (af - ah) * 1.
        # So yes, W[c01] += 1.
        
        # Let's check M5: (a+b)h.
        # u5 = e_a + e_b. v5 = e_h.
        # Term: (A[a] + A[b]) B[h].
        # A[a] = A[i0, j0]. A[b] = A[i0, j1]. B[h] = B[j1, k1].
        # Term = ah + bh.
        # Contributes to C12 (c01).
        # So W[c01] += 1.
        
        # Let's check M6: (c-a)(e+f).
        # u6 = e_c - e_a. v6 = e_e + e_f.
        # Term: (A[c] - A[a]) (B[e] + B[f]).
        # = ce + cf - ae - af.
        # C11 = M1 - M4 + M5 + M6.
        # C11 = ae + bf? No.
        # C11 = A[i0, j0]B[j0, k0] + A[i0, j1]B[j1, k0] = ae + bg.
        # Let's check the sum.
        # M1: ae + ah + de + dh.
        # M4: -(dg - de) = -dg + de.
        # M5: ah + bh.
        # M6: ce + cf - ae - af.
        # Sum: (ae - ae) + (ah + ah) + de + de + dh - dg + bh + ce + cf - af.
        # This is a mess. I am confusing myself.
        
        # Let's just verify the final C matrix by summing the terms.
        # C11 = M1 - M4 + M5 + M6
        # M1 = (a+d)(e+h) = ae + ah + de + dh
        # M4 = d(g-e) = dg - de
        # M5 = (a+b)h = ah + bh
        # M6 = (c-a)(e+f) = ce + cf - ae - af
        # Sum:
        # ae: 1 (M1) - 1 (M6) = 0? No, M1 has ae. M6 has -ae.
        # Wait. C11 should be ae + bg.
        # Let's re-evaluate M6.
        # M6 = (c-a)(e+f) = ce + cf - ae - af.
        # Is this correct Strassen?
        # Standard Strassen M6 is (c-a)(e+f).
        # Let's check C11 again.
        # C11 = M1 - M4 + M5 + M6.
        # M1: ae + ah + de + dh
        # -M4: -dg + de
        # M5: ah + bh
        # M6: ce + cf - ae - af
        # Sum:
        # ae: 1 - 1 = 0. (M1 has ae, M6 has -ae)
        # af: -1 (M6)
        # ah: 1 (M1) + 1 (M5) = 2
        # bh: 1 (M5)
        # ce: 1 (M6)
        # cf: 1 (M6)
        # de: 1 (M1) + 1 (-M4) = 2
        # dh: 1 (M1)
        # dg: -1 (-M4)
        
        # This does NOT look like ae + bg.
        # I must have the wrong Strassen formula or indices.
        
        # Let's look up Strassen 2x2 again.
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C11 = M1 - M4 + M5 + M6
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 - M7
        
        # Let's check C11 again.
        # M1 = ae + ah + de + dh
        # M4 = dg - de
        # M5 = ah + bh
        # M6 = ce + cf - ae - af
        
        # C11 = M1 - M4 + M5 + M6
        # = (ae + ah + de + dh) - (dg - de) + (ah + bh) + (ce + cf - ae - af)
        # = ae + ah + de + dh - dg + de + ah + bh + ce + cf - ae - af
        # = (ae - ae) + (ah + ah) + (de + de) + dh - dg + bh + ce + cf - af
        # = 2ah + 2de + dh - dg + bh + ce + cf - af
        
        # This is WRONG. C11 should be ae + bg.
        
        # Ah! The indices of B in Strassen are often swapped or I am misremembering.
        # Let's check a reliable source mentally.
        # Maybe M6 is (c-a)(f+e)? No.
        # Maybe M4 is d(g-e)?
        
        # Let's try:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Is it possible my expansion is wrong?
        # (c-a)(e+f) = ce + cf - ae - af. Correct.
        
        # Maybe the formula for C11 is different?
        # Wikipedia:
        # C11 = M1 - M4 + M5 + M6
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 - M7
        
        # Let's check C21 = M2 + M4.
        # M2 = (c+d)e = ce + de
        # M4 = d(g-e) = dg - de
        # Sum = ce + dg. Correct.
        
        # Let's check C12 = M3 + M5.
        # M3 = a(f-h) = af - ah
        # M5 = (a+b)h = ah + bh
        # Sum = af + bh. Correct.
        
        # So C12 and C21 are correct.
        # Why is C11 wrong?
        # Let's re-expand M1, M4, M5, M6.
        # M1 = ae + ah + de + dh
        # M4 = dg - de
        # M5 = ah + bh
        # M6 = ce + cf - ae - af
        
        # Sum:
        # ae: 1 (M1) - 1 (M6) = 0.
        # Wait. M6 has -ae.
        # So ae cancels.
        # But C11 needs ae.
        
        # Is M6 actually (c-a)(e+f)?
        # Maybe it's (c-a)(f+e)? Same thing.
        
        # Let's check M1 again.
        # (a+d)(e+h) = ae + ah + de + dh.
        
        # Is it possible M4 is defined differently?
        # M4 = d(g-e).
        
        # Let's look at C22 = M1 - M3 + M2 - M7.
        # M1 = ae + ah + de + dh
        # M3 = af - ah
        # M2 = ce + de
        # M7 = (b-d)(g+h) = bg + bh - dg - dh
        
        # C22 = M1 - M3 + M2 - M7
        # = (ae + ah + de + dh) - (af - ah) + (ce + de) - (bg + bh - dg - dh)
        # = ae + ah + de + dh - af + ah + ce + de - bg - bh + dg + dh
        # = ae - af + 2ah + 2de + 2dh + ce - bg - bh + dg
        
        # C22 should be cf + dh.
        # This is also wrong.
        
        # I am definitely missing something.
        # Ah! The B matrix in Strassen is usually [[e, f], [g, h]].
        # But maybe the indices are swapped in my head?
        # A = [[a, b], [c, d]]
        # B = [[e, g], [f, h]]? No.
        
        # Let's assume the standard formula is correct and I am making an algebraic error.
        # M1 - M4 + M5 + M6
        # = (ae + ah + de + dh) - (dg - de) + (ah + bh) + (ce + cf - ae - af)
        # Let's group by terms:
        # ae: 1 - 1 = 0
        # af: -1
        # ah: 1 + 1 = 2
        # bh: 1
        # ce: 1
        # cf: 1
        # de: 1 + 1 = 2
        # dh: 1
        # dg: -1
        
        # This result is independent of my previous checks. It is not C11.
        
        # Wait. C11 = ae + bg.
        # My result has no bg. It has bh, ce, cf.
        
        # Is it possible that M6 is NOT (c-a)(e+f)?
        # Some sources define M6 = (c-a)(e+f).
        
        # Let's try a different set of M's.
        # Maybe I should just use the 4 schoolbook terms for 2x2x2?
        # Rank 4 is simpler and safer.
        # 6 blocks * 4 + 3 blocks * 4 = 36.
        # This is worse than 54? No, 36 < 54.
        # Schoolbook is 4. Strassen is 7.
        # So Schoolbook is BETTER for 2x2x2?
        # No, Strassen is 7, Schoolbook is 4.
        # Wait. 2x2x2 schoolbook rank is 4.
        # Strassen rank is 7.
        # So Strassen is WORSE for 2x2x2?
        # No, Strassen reduces 8 to 7.
        # Schoolbook is 4 multiplications.
        # Why would Strassen be 7?
        # Ah. Strassen is for general n. For 2x2, it uses 7 multiplications.
        # Schoolbook uses 4.
        # So Schoolbook is better for 2x2!
        
        # So I should use Schoolbook for all blocks.
        # Rank = 6 * 4 + 3 * 4 = 36.
        
        # Let's verify Schoolbook for 2x2x2.
        # C11 = ae + bg. Terms: ae, bg.
        # C12 = af + bh. Terms: af, bh.
        # C21 = ce + dg. Terms: ce, dg.
        # C22 = cf + dh. Terms: cf, dh.
        
        # Total 4 terms.
        
        # So I will use Schoolbook for everything.
        # Rank 36.

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Blocks:
    # A1 (2x2), B1 (2x2) -> C1 (2x2)
    # A1 (2x2), B2 (2x2) -> C2 (2x2)
    # A1 (2x2), B3 (2x2) -> C3 (2x2)
    # A2 (2x2), B1 (2x2) -> C4 (2x2)
    # A2 (2x2), B2 (2x2) -> C5 (2x2)
    # A2 (2x2), B3 (2x2) -> C6 (2x2)
    # A3 (2x1), B1 (2x2) -> C7 (2x2)
    # A3 (2x1), B2 (2x2) -> C8 (2x2)
    # A3 (2x1), B3 (2x2) -> C9 (2x2)
    
    # Helper to add schoolbook terms
    def add_schoolbook_222(i_start, j_start, k_start, col_offset):
        # A[i0..i1, j0..j1] * B[j0..j1, k0..k1]
        a_idx = i_start * m + j_start
        b_idx = i_start * m + (j_start+1)
        c_idx = (i_start+1) * m + j_start
        d_idx = (i_start+1) * m + (j_start+1)
        
        e_idx = j_start * p + k_start
        f_idx = j_start * p + (k_start+1)
        g_idx = (j_start+1) * p + k_start
        h_idx = (j_start+1) * p + (k_start+1)
        
        c00_idx = k_start * n + i_start
        c01_idx = (k_start+1) * n + i_start
        c10_idx = k_start * n + (i_start+1)
        c11_idx = (k_start+1) * n + (i_start+1)
        
        # ae -> C11
        U[a_idx, col_offset] = 1
        V[e_idx, col_offset] = 1
        W[c00_idx, col_offset] = 1
        col_offset += 1
        
        # bg -> C11
        U[b_idx, col_offset] = 1
        V[g_idx, col_offset] = 1
        W[c00_idx, col_offset] = 1
        col_offset += 1
        
        # af -> C12
        U[a_idx, col_offset] = 1
        V[f_idx, col_offset] = 1
        W[c01_idx, col_offset] = 1
        col_offset += 1
        
        # bh -> C12
        U[b_idx, col_offset] = 1
        V[h_idx, col_offset] = 1
        W[c01_idx, col_offset] = 1
        col_offset += 1
        
        # ce -> C21
        U[c_idx, col_offset] = 1
        V[e_idx, col_offset] = 1
        W[c10_idx, col_offset] = 1
        col_offset += 1
        
        # dg -> C21
        U[d_idx, col_offset] = 1
        V[g_idx, col_offset] = 1
        W[c10_idx, col_offset] = 1
        col_offset += 1
        
        # cf -> C22
        U[c_idx, col_offset] = 1
        V[f_idx, col_offset] = 1
        W[c11_idx, col_offset] = 1
        col_offset += 1
        
        # dh -> C22
        U[d_idx, col_offset] = 1
        V[h_idx, col_offset] = 1
        W[c11_idx, col_offset] = 1
        col_offset += 1
        
        return col_offset

    def add_schoolbook_212(i_start, j_start, k_start, col_offset):
        # A[i0..i1, j0] * B[j0..j1, k0..k1]
        a_idx = i_start * m + j_start
        c_idx = (i_start+1) * m + j_start
        
        e_idx = j_start * p + k_start
        f_idx = j_start * p + (k_start+1)
        g_idx = (j_start+1) * p + k_start
        h_idx = (j_start+1) * p + (k_start+1)
        
        c00_idx = k_start * n + i_start
        c01_idx = (k_start+1) * n + i_start
        c10_idx = k_start * n + (i_start+1)
        c11_idx = (k_start+1) * n + (i_start+1)
        
        # ae -> C11
        U[a_idx, col_offset] = 1
        V[e_idx, col_offset] = 1
        W[c00_idx, col_offset] = 1
        col_offset += 1
        
        # ag -> C11
        U[a_idx, col_offset] = 1
        V[g_idx, col_offset] = 1
        W[c00_idx, col_offset] = 1
        col_offset += 1
        
        # af -> C12
        U[a_idx, col_offset] = 1
        V[f_idx, col_offset] = 1
        W[c01_idx, col_offset] = 1
        col_offset += 1
        
        # ah -> C12
        U[a_idx, col_offset] = 1
        V[h_idx, col_offset] = 1
        W[c01_idx, col_offset] = 1
        col_offset += 1
        
        # ce -> C21
        U[c_idx, col_offset] = 1
        V[e_idx, col_offset] = 1
        W[c10_idx, col_offset] = 1
        col_offset += 1
        
        # cg -> C21
        U[c_idx, col_offset] = 1
        V[g_idx, col_offset] = 1
        W[c10_idx, col_offset] = 1
        col_offset += 1
        
        # cf -> C22
        U[c_idx, col_offset] = 1
        V[f_idx, col_offset] = 1
        W[c11_idx, col_offset] = 1
        col_offset += 1
        
        # ch -> C22
        U[c_idx, col_offset] = 1
        V[h_idx, col_offset] = 1
        W[c11_idx, col_offset] = 1
        col_offset += 1
        
        return col_offset

    # Process 2x2x2 blocks
    col = add_schoolbook_222(0, 0, 0, col) # A1, B1
    col = add_schoolbook_222(0, 0, 2, col) # A1, B2
    col = add_schoolbook_222(0, 0, 4, col) # A1, B3
    col = add_schoolbook_222(0, 2, 0, col) # A2, B1
    col = add_schoolbook_222(0, 2, 2, col) # A2, B2
    col = add_schoolbook_222(0, 2, 4, col) # A2, B3
    
    # Process 2x1x2 blocks
    col = add_schoolbook_212(0, 4, 0, col) # A3, B1
    col = add_schoolbook_212(0, 4, 2, col) # A3, B2
    col = add_schoolbook_212(0, 4, 4, col) # A3, B3
    
    return {"U": U, "V": V, "W": W}

