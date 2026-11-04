def construct():
    n, m, p = 2, 4, 5
    
    # We'll use the block decomposition:
    # A (2x4) = [A1 | A2] where A1, A2 are 2x2
    # B (4x5) = [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2
    #
    # For each 2x2 * 2x5 multiplication, use rank-18 decomposition
    # Total rank: 18 + 18 = 36
    
    # First, let's build the M_{2,2,5} rank-18 decomposition
    # Split the 5 columns into 2+2+1
    # Columns 0,1: Strassen (rank 7)
    # Columns 2,3: Strassen (rank 7)  
    # Column 4: schoolbook (rank 4)
    
    # For a 2x2 * 2x2 multiplication with Strassen:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # The 7 multiplications and their contributions to C:
    
    # Let me define the Strassen decomposition for 2x2*2x2:
    # Each "multiplication" r has:
    #   u_r: coefficients for A entries (a,b,c,d) -> indexed as (0,1,2,3) for (a,b,c,d)
    #   v_r: coefficients for B entries (e,f,g,h) -> indexed as (0,1,2,3) for (e,f,g,h)
    #   w_r: contributions to C entries (C11,C12,C21,C22) -> indexed as (0,1,2,3)
    
    strassen_2x2x2 = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # M1 = (a+d)(e+h) -> C11 += 1, C22 += 1
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M2 = (c+d)e -> C21 += 1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # M3 = a(f-h) -> C12 += 1, C22 += -1
        ([1,0,0,0], [0,1,0,-1], [0,1,0,-1]),
        # M4 = d(g-e) -> C11 += -1, C12 += 1
        ([0,0,0,1], [0,-1,1,0], [-1,1,0,0]),
        # M5 = (a+b)h -> C11 += 1, C21 += 1
        ([1,1,0,0], [0,0,0,1], [1,0,1,0]),
        # M6 = (c-a)(e+f) -> C22 += 1
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        # M7 = (b-d)(g+h) -> C11 += -1, C22 += -1
        ([0,1,0,-1], [0,0,1,1], [-1,0,0,-1]),
    ]
    
    # Verify: For C11: M1 - M4 + M5 - M7 = (a+d)(e+h) - d(g-e) + (a+b)h - (b-d)(g+h)
    # = ae + ah + de + dh - dg + de + ah + bh - bg - bh + dg + dh
    # Wait, let me redo this more carefully.
    # M1 = (a+d)(e+h) = ae + ah + de + dh
    # M4 = d(g-e) = dg - de
    # M5 = (a+b)h = ah + bh
    # M7 = (b-d)(g+h) = bg + bh - dg - dh
    # C11 = M1 - M4 + M5 - M7
    # = (ae + ah + de + dh) - (dg - de) + (ah + bh) - (bg + bh - dg - dh)
    # = ae + ah + de + dh - dg + de + ah + bh - bg - bh + dg + dh
    # = ae + 2ah + 2de + 2dh - bg
    # That's not right. Let me recheck the Strassen formulas.
    
    # Standard Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C11 = p1 - p4 + p5 - p7
    # C12 = p3 + p4
    # C21 = p2 + p5
    # C22 = p1 - p3 + p6 - p7
    
    # Let me verify C11:
    # p1 = (a+d)(e+h) = ae + ah + de + dh
    # p4 = d(g-e) = dg - de
    # p5 = (a+b)h = ah + bh
    # p7 = (b-d)(g+h) = bg + bh - dg - dh
    # C11 = p1 - p4 + p5 - p7
    # = (ae + ah + de + dh) - (dg - de) + (ah + bh) - (bg + bh - dg - dh)
    # = ae + ah + de + dh - dg + de + ah + bh - bg - bh + dg + dh
    # = ae + 2ah + 2de + 2dh - bg
    # Hmm, that gives extra terms. Let me recheck.
    
    # Oh wait, I think I'm making an error. Let me be very careful.
    # -p7 = -(bg + bh - dg - dh) = -bg - bh + dg + dh
    # So C11 = p1 - p4 + p5 - p7
    # = (ae + ah + de + dh) + (-dg + de) + (ah + bh) + (-bg - bh + dg + dh)
    # = ae + ah + de + dh - dg + de + ah + bh - bg - bh + dg + dh
    # Collect like terms:
    # ae: 1
    # ah: 1 + 1 = 2
    # de: 1 + 1 = 2
    # dh: 1 + 1 = 2
    # -dg: -1 + 1 = 0
    # bh: 1 - 1 = 0
    # -bg: -1
    # So C11 = ae + 2ah + 2de + 2dh - bg
    # But we want C11 = ae + bg. Something is wrong.
    
    # Let me look up the correct Strassen formulas again.
    # 
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e  
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C11 = p1 - p4 + p5 - p7
    # C12 = p3 + p4
    # C21 = p2 + p5
    # C22 = p1 - p3 + p6 - p7
    
    # Let me verify C11 = p1 - p4 + p5 - p7:
    # p1 = (a+d)(e+h) = ae + ah + de + dh
    # -p4 = -d(g-e) = -dg + de
    # p5 = (a+b)h = ah + bh
    # -p7 = -(b-d)(g+h) = -(bg + bh - dg - dh) = -bg - bh + dg + dh
    #
    # Sum: ae + ah + de + dh - dg + de + ah + bh - bg - bh + dg + dh
    # = ae + (ah + ah) + (de + de) + (dh + dh) + (-dg + dg) + (bh - bh) - bg
    # = ae + 2ah + 2de + 2dh - bg
    #
    # This is NOT equal to ae + bg. So either my Strassen formulas are wrong or I'm making an arithmetic error.
    
    # Let me try with specific values: a=1,b=0,c=0,d=0,e=0,f=0,g=1,h=0
    # A = [[1,0],[0,0]], B = [[0,0],[1,0]]
    # C = A*B = [[1*0+0*1, 1*0+0*0],[0*0+0*1, 0*0+0*0]] = [[0,0],[0,0]]
    # C11 should be 0.
    #
    # p1 = (1+0)(0+0) = 0
    # p4 = 0*(1-0) = 0
    # p5 = (1+0)*0 = 0
    # p7 = (0-0)(1+0) = 0
    # C11 = 0 - 0 + 0 - 0 = 0. OK.
    
    # Try a=1,b=0,c=0,d=0,e=1,f=0,g=0,h=0
    # A = [[1,0],[0,0]], B = [[1,0],[0,0]]
    # C = [[1,0],[0,0]], C11 = 1
    #
    # p1 = (1+0)(1+0) = 1
    # p4 = 0*(0-1) = 0
    # p5 = (1+0)*0 = 0
    # p7 = (0-0)(0+0) = 0
    # C11 = 1 - 0 + 0 - 0 = 1. OK.
    
    # Try a=0,b=1,c=0,d=0,e=0,f=0,g=1,h=0
    # A = [[0,1],[0,0]], B = [[0,0],[1,0]]
    # C = [[0*0+1*1, 0*0+1*0],[0*0+0*1, 0*0+0*0]] = [[1,0],[0,0]], C11 = 1
    #
    # p1 = (0+0)(0+0) = 0
    # p4 = 0*(1-0) = 0
    # p5 = (0+1)*0 = 0
    # p7 = (1-0)(1+0) = 1
    # C11 = 0 - 0 + 0 - 1 = -1. WRONG! Should be 1.
    
    # So my Strassen formulas are incorrect. Let me look them up properly.
    
    # Correct Strassen:
    # p1 = (a + d) * (e + h)
    # p2 = (c + d) * e
    # p3 = a * (f - h)
    # p4 = d * (g - e)
    # p5 = (a + b) * h
    # p6 = (c - a) * (e + f)
    # p7 = (b - d) * (g + h)
    #
    # C11 = p1 - p4 + p5 - p7
    # C12 = p3 + p4
    # C21 = p2 + p5
    # C22 = p1 - p3 + p6 - p7
    
    # Wait, that's what I had. Let me recheck the test case.
    # a=0,b=1,c=0,d=0,e=0,f=0,g=1,h=0
    # p1 = (0+0)(0+0) = 0
    # p2 = (0+0)*0 = 0
    # p3 = 0*(0-0) = 0
    # p4 = 0*(1-0) = 0
    # p5 = (0+1)*0 = 0
    # p6 = (0-0)*(0+0) = 0
    # p7 = (1-0)*(1+0) = 1
    # C11 = p1 - p4 + p5 - p7 = 0 - 0 + 0 - 1 = -1
    # But C11 should be ae + bg = 0*0 + 1*1 = 1.
    
    # I think the issue is that the standard Strassen formulas I'm recalling might have a different sign convention. Let me try:
    # C11 = p1 + p5 - p4 - p7
    # That's the same as what I have. Hmm.
    
    # Actually, I think I need to double-check the Strassen algorithm. Let me derive it.
    # We want to compute C = A*B with 7 multiplications.
    # 
    # The standard Strassen algorithm is:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C11 = m1 - m4 + m5 - m7
    # C12 = m3 + m4
    # C21 = m2 + m5
    # C22 = m1 - m3 + m6 - m7
    
    # Let me verify C12 = m3 + m4:
    # m3 = a(f-h) = af - ah
    # m4 = d(g-e) = dg - de
    # C12 = af - ah + dg - de
    # But C12 should be af + bh.
    # So C12 = af - ah + dg - de. For this to equal af + bh, we need -ah + dg - de = bh.
    # That's not generally true. So something is definitely wrong.
    
    # I think I'm confusing the Strassen formulas. Let me look this up from first principles.
    # 
    # Actually, I recall now. The correct formulas are:
    # C11 = m1 - m4 + m5 - m7
    # C12 = m3 + m4
    # C21 = m2 + m5  
    # C22 = m1 - m3 + m6 - m7
    #
    # But wait, let me verify C12 with the test case a=0,b=1,c=0,d=0,e=0,f=0,g=1,h=0:
    # m3 = 0*(0-0) = 0
    # m4 = 0*(1-0) = 0
    # C12 = 0 + 0 = 0
    # But C12 should be af + bh = 0*0 + 1*0 = 0. OK, that works.
    
    # Let me try another test: a=0,b=1,c=0,d=0,e=0,f=1,g=0,h=0
    # A = [[0,1],[0,0]], B = [[0,1],[0,0]]
    # C = [[0*0+1*0, 0*1+1*0],[0*0+0*0, 0*1+0*0]] = [[0,0],[0,0]]
    # C12 = 0
    # m3 = 0*(1-0) = 0
    # m4 = 0*(0-0) = 0
    # C12 = 0 + 0 = 0. OK.
    
    # Try a=1,b=0,c=0,d=0,e=0,f=1,g=0,h=0
    # A = [[1,0],[0,0]], B = [[0,1],[0,0]]
    # C = [[1*0+0*0, 1*1+0*0],[0*0+0*0, 0*1+0*0]] = [[0,1],[0,0]]
    # C12 = 1
    # m3 = 1*(1-0) = 1
    # m4 = 0*(0-0) = 0
    # C12 = 1 + 0 = 1. OK!
    
    # Try a=0,b=1,c=0,d=0,e=0,f=0,g=0,h=1
    # A = [[0,1],[0,0]], B = [[0,0],[0,1]]
    # C = [[0*0+1*0, 0*0+1*1],[0*0+0*0, 0*0+0*1]] = [[0,1],[0,0]]
    # C12 = 1
    # m3 = 0*(0-1) = 0
    # m4 = 0*(0-0) = 0
    # C12 = 0 + 0 = 0. WRONG! Should be 1.
    
    # So C12 = m3 + m4 doesn't work in this case. I must have the wrong formula.
    
    # Let me try to look up the correct Strassen formulas more carefully.
    # 
    # After more thought, I believe the correct Strassen formulas are:
    # m1 = (a + d) * (e + h)
    # m2 = (c + d) * e
    # m3 = a * (f - h)
    # m4 = d * (g - e)
    # m5 = (a + b) * h
    # m6 = (c - a) * (e + f)
    # m7 = (b - d) * (g + h)
    #
    # C11 = m1 - m4 + m5 - m7
    # C12 = m3 + m4
    # C21 = m2 + m5
    # C22 = m1 - m3 + m6 - m7
    
    # But this doesn't work. Let me try a completely different approach.
    # 
    # Actually, I just realized I might be misremembering. Let me try:
    # C11 = m1 + m5 - m4 - m7 (same as before)
    # C12 = m3 + m4 (doesn't work)
    # 
    # What if it's:
    # C12 = m3 + m4? No.
    # C12 = m3 + m5? Let's check:
    # For a=0,b=1,c=0,d=0,e=0,f=0,g=0,h=1:
    # m3 = 0, m5 = (0+1)*1 = 1
    # C12 = 0 + 1 = 1. OK!
    # For a=1,b=0,c=0,d=0,e=0,f=1,g=0,h=0:
    # m3 = 1, m5 = (1+0)*0 = 0
    # C12 = 1 + 0 = 1. OK!
    # For a=0,b=1,c=0,d=0,e=0,f=0,g=1,h=0:
    # m3 = 0, m5 = 0
    # C12 = 0. Should be 0. OK!
    
    # So maybe C12 = m3 + m5? Let me check more cases.
    # a=1,b=1,c=0,d=0,e=1,f=1,g=0,h=0
    # A = [[1,1],[0,0]], B = [[1,1],[0,0]]
    # C = [[1*1+1*0, 1*1+1*0],[0,0]] = [[1,1],[0,0]]
    # C12 = 1
    # m3 = 1*(1-0) = 1
    # m5 = (1+1)*0 = 0
    # C12 = 1 + 0 = 1. OK!
    
    # a=1,b=1,c=0,d=0,e=0,f=0,g=1,h=1
    # A = [[1,1],[0,0]], B = [[0,0],[1,1]]
    # C = [[1*0+1*1, 1*0+1*1],[0,0]] = [[1,1],[0,0]]
    # C12 = 1
    # m3 = 1*(0-1) = -1
    # m5 = (1+1)*1 = 2
    # C12 = -1 + 2 = 1. OK!
    
    # So it seems C12 = m3 + m5 might be correct? But that's not the standard formula I recall.
    
    # Let me just give up on deriving Strassen from memory and use a different approach.
    # 
    # Instead, let me just use the schoolbook decomposition but try to find linear dependencies to reduce the rank.
    # 
    # Actually, let me take a step back. The simplest approach that might work:
    # 1. Start with schoolbook rank 40
    # 2. Try to find if any columns are linearly dependent
    # 3. If so, eliminate them
    # 
    # But for the schoolbook decomposition, each column has a unique (i,j,k) triple, so they're all independent.
    # 
    # Let me try a different approach: use the fact that we can combine terms.
    # 
    # Actually, let me just implement the block decomposition with Strassen for the 2x2x2 parts.
    # I'll verify it numerically.
    
    # Let me try a completely different approach: just use the schoolbook and verify it works.
    # Then I'll try to reduce the rank by finding dependencies.
    
    # For now, let me just return the schoolbook to ensure correctness.
    
    rank = n * m * p  # 40
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
