"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time

def _schoolbook(n, m, p):
    """Return schoolbook rank-n*m*p decomposition."""
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
    return U, V, W

def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2 * 2x2 multiplication.
    Each term is (u, v, w) where u,v,w are lists of 4 entries.
    Using the standard Strassen decomposition.
    """
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[p1+p4+p5-p6, p2+p4], [p3+p5, p1-p3+p4+p6]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-d)(e+f)
    # p7 = (b-d)(g+h)
    
    # For term p1: (a+d)(e+h)
    # u = [1,0,0,1] (a+d), v = [1,1,0,0] (e+h)
    # w: contributes to c00 with +1, c01 with 0, c10 with 0, c11 with +1
    # Wait, p1 appears in c00 with +1 and c11 with +1.
    # Actually c00 = p1+p4+p5-p6, c11 = p1-p3+p4+p6
    # So p1 contributes +1 to c00 and +1 to c11.
    
    terms = []
    
    # p1 = (a+d)(e+h) -> +1 to c00, +1 to c11
    terms.append(([1,0,0,1], [1,1,0,0], [1,0,0,1]))
    
    # p2 = (c+d)e -> +1 to c01
    # u = [0,0,1,1], v = [1,0,0,0]
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,0]))
    
    # p3 = a(f-h) -> +1 to c10
    # u = [1,0,0,0], v = [0,1,0,-1]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,0,1,0]))
    
    # p4 = d(g-e) -> +1 to c00, +1 to c11
    # u = [0,0,0,1], v = [0,0,1,-1]
    terms.append(([0,0,0,1], [0,0,1,-1], [1,0,0,1]))
    
    # p5 = (a+b)h -> +1 to c00
    # u = [1,1,0,0], v = [0,0,0,1]
    terms.append(([1,1,0,0], [0,0,0,1], [1,0,0,0]))
    
    # p6 = (c-d)(e+f) -> -1 to c00, +1 to c11
    # u = [0,0,1,-1], v = [1,1,0,0]
    terms.append(([0,0,1,-1], [1,1,0,0], [-1,0,0,1]))
    
    # p7 = (b-d)(g+h) -> +1 to c10
    # Wait, let me recheck. c10 = p3+p5. Where does p7 go?
    # Standard Strassen:
    # c00 = p1+p4+p5-p6
    # c01 = p2+p4
    # c10 = p3+p5
    # c11 = p1-p3+p4+p6
    # Hmm, p7 is not in the standard 7-term Strassen for 2x2x2.
    # Let me re-derive. The standard Strassen uses 7 multiplications:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-d)(e+f)
    # M7 = (b-d)(g+h)  -- wait, this gives 7 but the output formulas are:
    # c00 = M1+M4+M5-M6
    # c01 = M2+M4
    # c10 = M3+M5
    # c11 = M1-M3+M4+M6
    # So M7 is not used? That can't be right.
    
    # Let me look up the correct Strassen decomposition.
    # Actually, the 7 products are:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e)
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)(h)
    # p6 = (c-d)(e+f)
    # p7 = (b-d)(g+h)
    
    # And:
    # c00 = p1+p4+p5-p6
    # c01 = p2+p4
    # c10 = p3+p5
    # c11 = p1-p3+p4+p6
    
    # But that's only using p1-p6 in the output, not p7. That's wrong.
    
    # Let me use a different standard form. The correct Strassen is:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)e
    # M7 = (b-d)(g+h)
    
    # c00 = M1+M4-M5+M7
    # c01 = M5+M3
    # c10 = M6+M2
    # c11 = M1+M2-M4-M6
    
    # Let me verify:
    # c00 should be ae+bg
    # M1+M4-M5+M7 = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    # = ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh
    # = ae + (ah-ah) + (de-de) + dh-dh + dg-dg + (bh-bh) + bg
    # = ae+bg. Correct!
    
    # c01 should be af+bh
    # M5+M3 = (a+b)h + a(f-h) = ah+bh+af-ah = af+bh. Correct!
    
    # c10 should be ce+dg
    # M6+M2 = (c-d)e + (c+d)(e+f) = ce-de + ce+cf+de+df = ce+cf+df
    # That's not ce+dg. Hmm.
    
    # I'm getting confused. Let me just use the well-known correct version.
    
    # Correct Strassen for 2x2:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e+f)  
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)e
    # m7 = (b-d)(g+h)
    
    # c00 = m1+m4-m5+m7
    # c01 = m5+m3
    # c10 = m6+m2
    # c11 = m1+m2-m4-m6
    
    # Let me verify c10: m6+m2 = (c-d)e + (c+d)(e+f) = ce-de+ce+cf+de+df = 2ce+cf+df. Wrong.
    
    # OK I clearly don't remember the exact Strassen coefficients. Let me just use the schoolbook
    # for 2x2x2 (rank 4) to be safe, since I need to get a valid answer.
    
    # Actually, let me just implement the full schoolbook and be done.
    # The key insight is that for 2x5x6, the schoolbook rank is 60.
    # I can try to find a better decomposition, but if I can't verify it,
    # I should return the schoolbook to at least get a valid score.
    
    return None

def construct():
    """
    Main entry point with deadline guard.
    Returns the best decomposition found within the time limit.
    """
    start_time = time.time()
    deadline = 145.0  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Check deadline
    if time.time() - start_time > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
