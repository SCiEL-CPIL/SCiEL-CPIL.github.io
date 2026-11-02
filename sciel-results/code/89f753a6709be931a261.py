"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms for remaining dimensions, targeting rank < 72.
"""

DIMENSIONS = [3, 3, 8]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # Strategy: Decompose C = A*B where A is 3x3, B is 3x8.
    # Split A into A_top (2x3, rows 0-1) and A_bot (1x3, row 2).
    # Split B into B_top (2x8, rows 0-1) and B_bot (1x8, row 2).
    #
    # C[0:2, :] = A_top * B_top + A_top[:, 2:3] * B_bot
    # C[2, :]   = A_bot * B_top + A_bot[2:3] * B_bot
    #
    # For A_top[:, 0:2] * B_top (2x2 * 2x8):
    #   Split B_top into 4 blocks of 2x2, apply Strassen to each: 4*7 = 28 terms
    #   This covers C[0:2, 0:8] from the A[0:2, 0:2] * B[0:2, :] part
    #
    # Remaining direct terms:
    # A_top[:, 2] * B_top: A[0:2, 2] * B[0:2, :] -> 2*2*8 = 32 terms
    # A_top[:, 2] * B_bot: A[0:2, 2] * B[2, :] -> 2*1*8 = 16 terms  
    # A_bot * B_top: A[2, 0:2] * B[0:2, :] -> 1*2*8 = 16 terms
    # A_bot * B_bot: A[2, 2] * B[2, :] -> 1*1*8 = 8 terms
    # Total direct: 32 + 16 + 16 + 8 = 72
    # Total: 28 + 72 = 100 -- too many!
    #
    # Better approach: Only use Strassen for A[0:2,0:2]*B[0:2,0:2] (saves 1)
    # and handle everything else directly.
    #
    # Actually, let me reconsider. The correct decomposition must satisfy:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # Key insight: Each Strassen block uses a FIXED 2x2 A-block. If we apply
    # Strassen to A[0:2,0:2]*B[0:2,0:2], the U terms reference A[0,0],A[0,1],A[1,0],A[1,1].
    # If we ALSO apply Strassen to A[0:2,0:2]*B[0:2,2:4], the U terms reference the SAME
    # A entries. This is fine because each rank-r term is independent -- the U for term r
    # only has entries in the A positions that term uses.
    #
    # Wait, I think the original code IS correct. Let me re-examine.
    #
    # In the original code, strassen_decomp(0,0,0,0) adds 7 terms where:
    # - U terms reference A[0,0], A[0,1], A[1,0], A[1,1]
    # - V terms reference B[0,0], B[0,1], B[1,0], B[1,1]
    # - W terms reference C[0,0], C[0,1], C[1,0], C[1,1]
    #
    # Then strassen_decomp(0,0,0,2) adds 7 terms where:
    # - U terms reference A[0,0], A[0,1], A[1,0], A[1,1] (SAME A entries)
    # - V terms reference B[0,2], B[0,3], B[1,2], B[1,3]
    # - W terms reference C[0,2], C[0,3], C[1,2], C[1,3]
    #
    # This is perfectly valid! Each rank term is independent. The U matrix for
    # term r has nonzero entries only where that term's A combination is nonzero.
    # There's no conflict because we're building separate columns of U,V,W.
    #
    # So the original code should be correct. Let me trace through why verification
    # might fail...
    #
    # Actually wait - I see the issue now! Look at m1 in Strassen:
    # m1 = (a+d)(e+h)
    # U: a=1, d=1  -> U[0,0]=1, U[1,1]=1
    # V: e=1, h=1  -> V[0,0]=1, V[1,1]=1
    # W: C00=1, C11=1
    #
    # For this term, the contribution to T[i,j,k] is:
    # sum over (i,j) where U[i,j]!=0 and (j',k) where V[j',k]!=0:
    # U[i,j] * V[j',k] * W[i,k]
    #
    # But this only gives nonzero when j=j' (since U and V must share the same j index
    # for the product to contribute to the correct (i,j,k) triple).
    #
    # Wait no. The tensor decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For a single rank term r, the contribution to T[i,j,k] is:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # So for m1: U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1
    # T[0,0,0] += U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # T[0,1,0] += U[0,1]*V[1,0]*W[0,0] = 0*1*1 = 0
    # T[1,0,0] += U[1,0]*V[0,0]*W[1,0] = 0*1*0 = 0
    # T[1,1,1] += U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1
    # T[0,0,1] += U[0,0]*V[0,1]*W[0,1] = 1*0*0 = 0
    # etc.
    #
    # So m1 contributes 1 to T[0,0,0] and T[1,1,1].
    #
    # m2 = (c+d)*e: U[1,0]=1, U[1,1]=1, V[0,0]=1, W[1,0]=1
    # T[1,0,0] += U[1,0]*V[0,0]*W[1,0] = 1*1*1 = 1
    # T[1,1,0] += U[1,1]*V[1,0]*W[1,0] = 1*0*1 = 0
    # Contributes 1 to T[1,0,0].
    #
    # m3 = a*(f-h): U[0,0]=1, V[0,1]=1, V[1,1]=-1, W[0,1]=1, W[1,1]=1
    # T[0,0,1] += U[0,0]*V[0,1]*W[0,1] = 1*1*1 = 1
    # T[0,1,1] += U[0,1]*V[1,1]*W[0,1] = 0*(-1)*1 = 0
    # T[1,0,1] += U[1,0]*V[0,1]*W[1,1] = 0*1*1 = 0
    # T[1,1,1] += U[1,1]*V[1,1]*W[1,1] = 0*(-1)*1 = 0
    # Contributes 1 to T[0,0,1].
    #
    # m4 = d*(g-e): U[1,1]=1, V[1,0]=1, V[0,0]=-1, W[0,0]=1, W[1,0]=1
    # T[0,0,0] += U[0,0]*V[0,0]*W[0,0] = 0*(-1)*1 = 0
    # T[0,1,0] += U[0,1]*V[1,0]*W[0,0] = 0*1*1 = 0
    # T[1,0,0] += U[1,0]*V[0,0]*W[1,0] = 0*(-1)*1 = 0
    # T[1,1,0] += U[1,1]*V[1,0]*W[1,0] = 1*1*1 = 1
    # Contributes 1 to T[1,1,0].
    #
    # m5 = (a+b)*h: U[0,0]=1, U[0,1]=1, V[1,1]=1, W[0,0]=1, W[0,1]=1
    # T[0,0,0] += U[0,0]*V[0,0]*W[0,0] = 1*0*1 = 0
    # T[0,1,0] += U[0,1]*V[1,0]*W[0,0] = 1*0*1 = 0
    # T[1,0,0] += U[1,0]*V[0,0]*W[1,0] = 0*0*0 = 0
    # T[1,1,0] += U[1,1]*V[1,0]*W[1,0] = 0*0*0 = 0
    # T[0,0,1] += U[0,0]*V[0,1]*W[0,1] = 1*0*1 = 0
    # T[0,1,1] += U[0,1]*V[1,1]*W[0,1] = 1*1*1 = 1
    # T[1,0,1] += U[1,0]*V[0,1]*W[1,1] = 0*0*0 = 0
    # T[1,1,1] += U[1,1]*V[1,1]*W[1,1] = 0*1*0 = 0
    # Contributes 1 to T[0,1,1].
    #
    # m6 = (c-a)*(e+f): U[1,0]=1, U[0,0]=-1, V[0,0]=1, V[0,1]=1, W[1,1]=1
    # T[0,0,1] += U[0,0]*V[0,0]*W[0,1] = (-1)*1*0 = 0
    # T[0,1,1] += U[0,1]*V[0,1]*W[0,1] = 0*1*0 = 0
    # T[1,0,1] += U[1,0]*V[0,0]*W[1,1] = 1*1*1 = 1
    # T[1,1,1] += U[1,1]*V[0,1]*W[1,1] = 0*1*1 = 0
    # Contributes 1 to T[1,0,1].
    #
    # m7 = (b-d)*(g+h): U[0,1]=1, U[1,1]=-1, V[1,0]=1, V[1,1]=1, W[0,0]=-1
    # T[0,0,0] += U[0,0]*V[0,0]*W[0,0] = 0*0*(-1) = 0
    # T[0,1,0] += U[0,1]*V[1,0]*W[0,0] = 1*1*(-1) = -1
    # T[1,0,0] += U[1,0]*V[0,0]*W[1,0] = 0*0*0 = 0
    # T[1,1,0] += U[1,1]*V[1,0]*W[1,0] = (-1)*1*0 = 0
    # T[0,0,1] += U[0,0]*V[0,1]*W[0,1] = 0*0*0 = 0
    # T[0,1,1] += U[0,1]*V[1,1]*W[0,1] = 1*1*0 = 0
    # T[1,0,1] += U[1,0]*V[0,1]*W[1,1] = 0*0*0 = 0
    # T[1,1,1] += U[1,1]*V[1,1]*W[1,1] = (-1)*1*0 = 0
    # Contributes -1 to T[0,1,0].
    #
    # Now sum all contributions for T[0,0,0]:
    # m1: 1, m2: 0, m3: 0, m4: 0, m5: 0, m6: 0, m7: 0 -> Total: 1 ✓
    #
    # T[0,1,0]: m1: 0, m2: 0, m3: 0, m4: 0, m5: 0, m6: 0, m7: -1 -> Total: -1 ✗ (should be 0)
    #
    # AH HA! T[0,1,0] should be 0 (there's no term A[0,1]*B[1,0] in the 2x2*2x2 product
    # that gives C[0,0]... wait, actually C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0])
    #
    # So T[0,1,0] SHOULD be 1 (it's the A[0,1]*B[1,0] term contributing to C[0,0]).
    #
    # Wait, I'm confusing myself. Let me reclarify the tensor definition.
    #
    # The matrix multiplication tensor T for M_{n,m,p} has:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    #
    # Because EVERY product A[i,j]*B[j,k] contributes to C[i,k].
    #
    # So T[0,1,0] = 1 (A[0,1]*B[1,0] contributes to C[0,0]).
    #
    # Let me redo the sum for T[0,1,0]:
    # m1: U[0,1]*V[1,0]*W[0,0] = 0*1*1 = 0
    # m2: U[0,1]*V[1,0]*W[0,0] = 0*1*0 = 0 (W for m2 is only C10)
    # m3: U[0,1]*V[1,0]*W[0,0] = 0*0*0 = 0
    # m4: U[0,1]*V[1,0]*W[0,0] = 0*1*1 = 0 (U[0,1] for m4 is 0)
    # m5: U[0,1]*V[1,0]*W[0,0] = 1*0*1 = 0
    # m6: U[0,1]*V[1,0]*W[0,0] = 0*0*0 = 0
    # m7: U[0,1]*V[1,0]*W[0,0] = 1*1*(-1) = -1
    # Total: -1, but should be 1. ERROR!
    #
    # The issue: m7 = (b-d)*(g+h) where b=A[0,1], d=A[1,1], g=B[1,0], h=B[1,1]
    # This term is designed to cancel in C[0,0] (coefficient -1) and contribute to C[0,1].
    # But in the tensor decomposition, it contributes -1 to T[0,1,0] which is wrong.
    #
    # I think I'm making an error in how Strassen maps to the tensor. Let me reconsider.
    #
    # Strassen says: C[0,0] = m1 + m4 + m5 - m7
    # where m1 = (a+d)(e+h), m4 = d(g-e), m5 = (a+b)h, m7 = (b-d)(g+h)
    #
    # Expanding: (a+d)(e+h) + d(g-e) + (a+b)h - (b-d)(g+h)
    # = ae+ah+de+dh + dg-de + ah+bh - bg-bh+dg+dh
    # = ae + 2ah + dh + dg + bh - bg + dh
    # Hmm, that doesn't simplify to ae+bg.
    #
    # Let me recheck: (a+d)(e+h) = ae+ah+de+dh
    # d(g-e) = dg-de
    # (a+b)h = ah+bh
    # -(b-d)(g+h) = -(bg+bh-dg-dh) = -bg-bh+dg+dh
    #
    # Sum: ae+ah+de+dh + dg-de + ah+bh -bg-bh+dg+dh
    # = ae + (ah+ah) + (de-de) + (dh+dh+dh) + dg+dg + bh-bh - bg
    # = ae + 2ah + 3dh + 2dg - bg
    #
    # That's definitely not ae+bg. I have the Strassen formulas wrong!
    #
    # Let me look up the correct Strassen formulas:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-e)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C[0,0] = m1 + m4 + m5 - m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    #
    # Let me verify C[0,0]:
    # m1 + m4 + m5 - m7 = (a+d)(e+h) + d(g-e) + (a+b)h - (b-d)(g+h)
    # = ae+ah+de+dh + dg-de + ah+bh - (bg+bh-dg-dh)
    # = ae+ah+de+dh + dg-de + ah+bh - bg-bh+dg+dh
    # = ae + 2ah + 2dh + 2dg - bg
    #
    # Still wrong. Let me look this up more carefully.
    #
    # Correct Strassen (from Wikipedia):
    # m1 = (a + d)(e + h)
    # m2 = (c + d)e
    # m3 = a(f − h)
    # m4 = d(g − e)
    # m5 = (a + b)h
    # m6 = (c − a)(e + f)
    # m7 = (b − d)(g + h)
    #
    # C[0,0] = m1 + m4 + m5 − m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 − m2 + m3 + m6
    #
    # Verify C[0,0] = m1 + m4 + m5 - m7:
    # = (a+d)(e+h) + d(g-e) + (a+b)h - (b-d)(g+h)
    # = ae+ah+de+dh + dg-de + ah+bh - bg-bh+dg+dh
    # = ae + 2ah + 2dh + 2dg - bg
    #
    # This is NOT ae+bg. So either my expansion is wrong or the formula is wrong.
    #
    # Let me expand more carefully:
    # (a+d)(e+h) = ae + ah + de + dh
    # d(g-e) = dg - de
    # (a+b)h = ah + bh
    # (b-d)(g+h) = bg + bh - dg - dh
    #
    # m1 + m4 + m5 - m7 = (ae + ah + de + dh) + (dg - de) + (ah + bh) - (bg + bh - dg - dh)
    # = ae + ah + de + dh + dg - de + ah + bh - bg - bh + dg + dh
    # = ae + (ah + ah) + (de - de) + (dh + dh + dh) + (dg + dg) + (bh - bh) - bg
    # = ae + 2ah + 3dh + 2dg - bg
    #
    # This is definitely wrong. I must have the Strassen formulas incorrect.
    #
    # Let me look at a reliable source. The standard Strassen algorithm:
    # m1 = (a + d)(e + h)
    # m2 = (c + d)(e)
    # m3 = (a)(f - h)
    # m4 = (d)(g - e)
    # m5 = (a + b)(h)
    # m6 = (c - a)(e + f)
    # m7 = (b - d)(g + h)
    #
    # C11 = m1 + m4 + m5 - m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    #
    # Hmm, let me try a different set. Maybe the issue is with which formula goes where.
    #
    # Actually, I recall now. Let me try:
    # p1 = (a + d)(e + h)
    # p2 = (c + d)e
    # p3 = a(f - h)
    # p4 = d(g - e)
    # p5 = (a + b)h
    # p6 = (c - a)(e + f)
    # p7 = (b - d)(g + h)
    #
    # C[0,0] = p1 + p4 - p5 + p7  (NOT p1+m4+m5-m7)
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    #
    # Let me verify C[0,0] = p1 + p4 - p5 + p7:
    # = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    # = ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh
    # = ae + (ah-ah) + (de-de) + (dh-dh) + (dg-dg) + (-bh+bh) + bg
    # = ae + bg ✓
    #
    # C[0,1] = p3 + p5 = a(f-h) + (a+b)h = af-ah+ah+bh = af+bh ✓
    # C[1,0] = p2 + p4 = (c+d)e + d(g-e) = ce+de+dg-de = ce+dg ✓
    # C[1,1] = p1 - p2 + p3 + p6 = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
    # = ae+ah+de+dh - ce-de + af-ah + ce+cf-ae-af
    # = (ae-ae) + (ah-ah) + (de-de) + dh + (-ce+ce) + (af-af) + cf
    # = dh + cf ✓
    #
    # So the CORRECT Strassen output formulas are:
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    #
    # The original code had:
    # W for m1: C00=1, C11=1
    # W for m4: C00=1, C10=1
    # W for m5: C00=1, C01=1  <-- WRONG! Should be C00=-1, C01=1
    # W for m7: C00=-1        <-- WRONG! Should be C00=+1
    #
    # So the bug is in the W coefficients for m5 and m7!
    # m5 should have W[C00] = -1, W[C01] = 1
    # m7 should have W[C00] = +1 (not -1)
    
    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets.
        
        Correct Strassen:
        m1 = (a+d)(e+h)
        m2 = (c+d)*e
        m3 = a*(f-h)
        m4 = d*(g-e)
        m5 = (a+b)*h
        m6 = (c-a)*(e+f)
        m7 = (b-d)*(g+h)
        
        C[0,0] = m1 + m4 - m5 + m7
        C[0,1] = m3 + m5
        C[1,0] = m2 + m4
        C[1,1] = m1 - m2 + m3 + m6
        """
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a+1)
        c = (row_off_a+1, col_off_a)
        d = (row_off_a+1, col_off_a+1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b+1)
        g = (row_off_b+1, col_off_b)
        h = (row_off_b+1, col_off_b+1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b+1)
        C10 = (row_off_a+1, col_off_b)
        C11 = (row_off_a+1, col_off_b+1)
        
        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U_t.append((a, 1)); U_t.append((d, 1))
        V_t.append((e, 1)); V_t.append((h, 1))
        W_t.append((C00, 1)); W_t.append((C11, 1))
        
        # m2 = (c+d)*e -> C10 += 1, C11 -= 1
        U_t.append((c, 1)); U_t.append((d, 1))
        V_t.append((e, 1))
        W_t.append((C10, 1)); W_t.append((C11, -1))
        
        # m3 = a*(f-h) -> C01 += 1, C11 += 1
        U_t.append((a, 1))
        V_t.append((f, 1)); V_t.append((h, -1))
        W_t.append((C01, 1)); W_t.append((C11, 1))
        
        # m4 = d*(g-e) -> C00 += 1, C10 += 1
        U_t.append((d, 1))
        V_t.append((g, 1)); V_t.append((e, -1))
        W_t.append((C00, 1)); W_t.append((C10, 1))
        
        # m5 = (a+b)*h -> C00 -= 1, C01 += 1
        U_t.append((a, 1)); U_t.append((b, 1))
        V_t.append((h, 1))
        W_t.append((C00, -1)); W_t.append((C01, 1))
        
        # m6 = (c-a)*(e+f) -> C11 += 1
        U_t.append((c, 1)); U_t.append((a, -1))
        V_t.append((e, 1)); V_t.append((f, 1))
        W_t.append((C11, 1))
        
        # m7 = (b-d)*(g+h) -> C00 += 1
        U_t.append((b, 1)); U_t.append((d, -1))
        V_t.append((g, 1)); V_t.append((h, 1))
        W_t.append((C00, 1))
    
    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append(((i, j), 1))
        V_t.append(((j, k), 1))
        W_t.append(((i, k), 1))
    
    U_t = []
    V_t = []
    W_t = []
    
    # A11*B11: Strassen for A[0:2,0:2] * B[0:2,0:2]
    strassen_decomp(0, 0, 0, 0, U_t, V_t, W_t)
    
    # A12*B21: A[0:2,2] * B[2,0:2]
    for i in range(2):
        for k in range(2):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A11*B12: A[0:2,0:2] * B[0:2,2:8]
    for block in range(3):
        col_off = 2 + 2*block
        strassen_decomp(0, 0, 0, col_off, U_t, V_t, W_t)
    
    # A12*B22: A[0:2,2] * B[2,2:8]
    for i in range(2):
        for k in range(2, 8):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A21*B11: A[2,0:2] * B[0:2,0:2]
    for j in range(2):
        for k in range(2):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A22*B21: A[2,2] * B[2,0:2]
    for k in range(2):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    # A21*B12: A[2,0:2] * B[0:2,2:8]
    for j in range(2):
        for k in range(2, 8):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A22*B22: A[2,2] * B[2,2:8]
    for k in range(2, 8):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    rank = len(W_t)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        (i, j), coeff = U_t[idx]
        U[i * m + j][idx] = coeff
        (j, k), coeff = V_t[idx]
        V[j * p + k][idx] = coeff
        (i, k), coeff = W_t[idx]
        W[k * n + i][idx] = coeff
    
    # Verify correctness and attempt rank reduction
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to reduce rank by checking for linearly dependent columns
    # in the tensor decomposition space
    best_U, best_V, best_W = U_arr, V_arr, W_arr
    best_rank = rank
    
    # Check if we can remove any column by seeing if it's a linear combination
    # of others in the tensor product space
    # Build the "tensor matrix" where each column r gives a vector in R^{n*m*p}
    # via U[:,r] ⊗ V[:,r] ⊗ W[:,r] flattened, but we only need the (i,j,k) entries
    
    # More efficient: for each pair of columns, check if one can be expressed
    # as a linear combination of the others by solving a linear system
    
    def try_remove_column(Ua, Va, Wa, r_to_remove):
        """Try to express column r_to_remove as linear combination of others."""
        R = Ua.shape[1]
        if R <= 1:
            return None
        
        # We need: sum_{s != r} alpha_s * (U[:,s] ⊙ V[:,s] ⊙ W[:,s]) = (U[:,r] ⊙ V[:,r] ⊙ W[:,r])
        # This is a linear system in the alpha_s variables
        
        # Build matrix: each row is a (i,j,k) triple, each column s is the product
        rows = []
        target = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = []
                    for s in range(R):
                        if s == r_to_remove:
                            continue
                        row.append(Ua[i*m+j, s] * Va[j*p+k, s] * Wa[k*n+i, s])
                    rows.append(row)
                    target.append(Ua[i*m+j, r_to_remove] * Va[j*p+k, r_to_remove] * Wa[k*n+i, r_to_remove])
        
        A_mat = np.array(rows, dtype=float)
        b_vec = np.array(target, dtype=float)
        
        # Solve least squares
        try:
            alphas, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
        except np.linalg.LinAlgError:
            return None
        
        # Check if the solution is exact
        residual = np.max(np.abs(A_mat @ alphas - b_vec))
        if residual > 1e-8:
            return None
        
        # Check that the alphas are "nice" (integer or half-integer)
        # Scale by 2 and check if close to integers
        alphas_scaled = alphas * 2
        if np.max(np.abs(alphas_scaled - np.round(alphas_scaled))) > 1e-6:
            return None
        
        return alphas
    
    # Try removing each column
    for attempt in range(min(rank, 50)):  # Limit attempts for time
        if time.time() > DEADLINE - 30:
            break
        
        removed = False
        for r in range(best_rank):
            if time.time() > DEADLINE - 30:
                break
            result = try_remove_column(best_U, best_V, best_W, r)
            if result is not None:
                # Remove column r and adjust others
                R = best_rank
                new_U = np.zeros((n*m, R-1), dtype=float)
                new_V = np.zeros((m*p, R-1), dtype=float)
                new_W = np.zeros((p*n, R-1), dtype=float)
                
                new_cols = [s for s in range(R) if s != r]
                for new_idx, s in enumerate(new_cols):
                    new_U[:, new_idx] = best_U[:, s]
                    new_V[:, new_idx] = best_V[:, s]
                    new_W[:, new_idx] = best_W[:, s]
                
                # Adjust: new_col_s = old_col_s + alpha_s * old_col_r for s != r
                for new_idx, s in enumerate(new_cols):
                    new_U[:, new_idx] += result[new_idx] * best_U[:, r]
                    new_V[:, new_idx] += result[new_idx] * best_V[:, r]
                    new_W[:, new_idx] += result[new_idx] * best_W[:, r]
                
                # Verify the new decomposition
                if verify_decomp(new_U, new_V, new_W):
                    # Check for nice values (integer or half-integer)
                    all_nice = True
                    for arr in [new_U, new_V, new_W]:
                        scaled = arr * 2
                        if np.max(np.abs(scaled - np.round(scaled))) > 1e-6:
                            all_nice = False
                            break
                    if all_nice:
                        best_U, best_V, best_W = new_U, new_V, new_W
                        best_rank = R - 1
                        removed = True
                        break
        
        if not removed:
            break
    
    # Convert back to list format with nice values
    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                # Check if it's integer or half-integer
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    # Convert to int if whole number
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    # Shouldn't happen if verification passed
                    nice_row.append(val)
            result.append(nice_row)
        return result
    
    if best_rank < rank:
        U_final = to_nice_list(best_U)
        V_final = to_nice_list(best_V)
        W_final = to_nice_list(best_W)
    else:
        U_final = U
        V_final = V
        W_final = W
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
