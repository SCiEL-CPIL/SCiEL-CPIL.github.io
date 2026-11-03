"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    # Rank 42 via Strassen on 2x2 blocks
    # Middle split into 2 blocks of 2, output split into 3 blocks of 2
    # 6 sub-problems, each rank 7 -> total rank 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Correct Strassen 2x2x2 decomposition
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B
    # C00=ae+bg, C01=af+bh, C10=ce+dg, C11=cf+dh
    #
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(g-h), M4=d(f-g)
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(f+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    #
    # Verify C00: (a+d)(e+h)+d(f-g)-(a+b)h+(b-d)(f+h)
    # = ae+ah+de+dh+df-dg-ah-bh+bf+bh-df-dh = ae+de-dg+bf
    # This should equal ae+bg... it doesn't match!
    #
    # I think the correct formulas are different. Let me use a verified set.
    # 
    # Actually, I recall now. The correct Strassen is:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Let me just verify with a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8
    # A=[[1,2],[3,4]], B=[[5,6],[7,8]]
    # C=[[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    #
    # M1=(1+4)(5+8)=5*13=65
    # M2=(3+4)*5=35
    # M3=1*(7-8)=-1
    # M4=4*(6-7)=-4
    # M5=(1+2)*8=24
    # M6=(3-1)*(5+6)=2*11=22
    # M7=(2-4)*(6+8)=(-2)*14=-28
    #
    # C00=65+(-4)-24+(-28)=65-4-24-28=9. Should be 19. WRONG.
    #
    # So my Strassen formula is incorrect. Let me find the right one.
    #
    # Correct Strassen (from Wikipedia):
    # p1 = (a + d)(e + h)
    # p2 = (c + d)e
    # p3 = a(g - h)
    # p4 = d(f - g)
    # p5 = (a + b)h
    # p6 = (c - a)(e + f)
    # p7 = (b - d)(f + h)
    #
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    #
    # With a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8:
    # p1=(1+4)(5+8)=65
    # p2=(3+4)*5=35
    # p3=1*(7-8)=-1
    # p4=4*(6-7)=-4
    # p5=(1+2)*8=24
    # p6=(3-1)*(5+6)=22
    # p7=(2-4)*(6+8)=-28
    #
    # C00=65+(-4)-24+(-28)=9. Still 9, not 19.
    #
    # I'm clearly getting the Strassen formula wrong. Let me derive it or use a different approach.
    #
    # Alternative: just use the known rank-7 decomposition directly.
    # The 7 bilinear forms for 2x2 matrix mult are:
    #
    # Actually, let me just hard-code a verified rank-7 decomposition.
    # One standard one:
    #
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(g-h)
    # m4 = d(f-g)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(f+h)
    #
    # C = [[m1+m4-m5+m7, m3+m5],
    #      [m2+m4, m1-m2+m3+m6]]
    #
    # This is what I had. Let me recheck my arithmetic.
    # a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8
    # m1 = (1+4)(5+8) = 5*13 = 65
    # m2 = (3+4)*5 = 35
    # m3 = 1*(7-8) = -1
    # m4 = 4*(6-7) = -4
    # m5 = (1+2)*8 = 24
    # m6 = (3-1)*(5+6) = 2*11 = 22
    # m7 = (2-4)*(6+8) = -2*14 = -28
    #
    # C00 = 65 + (-4) - 24 + (-28) = 65 - 4 - 24 - 28 = 9
    # But should be 1*5+2*7 = 5+14 = 19
    #
    # 9 != 19. So this formula is definitely wrong.
    #
    # Let me try yet another formulation. Perhaps:
    # C00 = m1 + m4 - m5 + m7 is wrong.
    #
    # Let me search my memory... I think the correct combination might be:
    # C00 = m1 + m4 - m5 + m7
    # but with different m definitions.
    #
    # OK let me just directly construct a rank-7 decomposition by solving.
    # Or better, let me use a completely different approach: I'll just use the
    # schoolbook for the 2x2 blocks (rank 8 each) giving total rank 48, but
    # that's not better.
    #
    # Actually, you know what, let me just verify the Strassen formula from first principles.
    # 
    # We need 7 bilinear forms m1..m7 such that:
    # C00 = ae+bg = linear combo of m's
    # C01 = af+bh = linear combo of m's
    # C10 = ce+dg = linear combo of m's
    # C11 = cf+dh = linear combo of m's
    #
    # Let me just use a known correct one. After more thought:
    #
    # m1 = (a+d)(e+h) = ae+ah+de+dh
    # m2 = (c+d)(e) = ce+de
    # m3 = a(g-h) = ag-ah
    # m4 = d(f-g) = df-dg
    # m5 = (a+b)(h) = ah+bh
    # m6 = (c-a)(e+f) = ce+cf-ae-af
    # m7 = (b-d)(f+h) = bf+bh-df-dh
    #
    # C00 = ae+bg
    # Let's see: m1+m4-m5+m7 = (ae+ah+de+dh)+(df-dg)-(ah+bh)+(bf+bh-df-dh)
    # = ae+ah+de+dh+df-dg-ah-bh+bf+bh-df-dh
    # = ae+de-dg+bf
    # 
    # For this to equal ae+bg, we need de-dg+bf = bg, i.e., d(e-g)+bf=bg
    # This is NOT true in general.
    #
    # I give up trying to recall Strassen. Let me just use a numerical approach:
    # I'll construct the decomposition and verify it, adjusting if needed.
    #
    # Actually, the simplest correct approach: use the fact that the 2x2x2 tensor
    # has rank 7, and I can find a rank-7 decomposition by solving a system.
    # But that's complex. 
    #
    # Alternative: just return the schoolbook rank-48 solution. It's correct and
    # will score 1.0. But the task says to improve.
    #
    # Let me try one more time with a different Strassen formulation I've seen:
    #
    # m1 = (a + d) * (e + h)
    # m2 = (c + d) * e
    # m3 = a * (g - h)
    # m4 = d * (f - g)
    # m5 = (a + b) * h
    # m6 = (c - a) * (e + f)
    # m7 = (b - d) * (f + h)
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    #
    # I've verified this gives C00=9 for my test case, which is wrong.
    #
    # Let me try: C00 = m1 + m4 - m5 + m7 is the issue.
    # What if C00 = m1 - m5 + m7 + m4? Same thing.
    #
    # What if the formula is actually:
    # C00 = m1 + m4 - m5 + m7
    # but with m4 = d*(f-g) should be m4 = d*(g-f)? Let me try:
    # m4 = d*(g-f) = 4*(7-6) = 4
    # C00 = 65 + 4 - 24 + (-28) = 17. Closer to 19 but not exact.
    #
    # What if m7 = (b-d)*(h-f)? m7 = (2-4)*(8-6) = -2*2 = -4
    # C00 = 65 + (-4) - 24 + (-4) = 33. No.
    #
    # I'm clearly misremembering Strassen. Let me just use a brute-force rank-7
    # decomposition that I can verify. I'll set up the problem and solve it.
    #
    # Actually, the cleanest approach: I'll just use the schoolbook (rank 48)
    # but that doesn't improve. OR, I can use a block approach with rank 8 per
    # 2x2 block (no Strassen) which also gives 48.
    #
    # Wait - maybe I should just try to find ANY rank < 48 decomposition.
    # 
    # For 2x4x6, the trivial bound is 48. Can we do better?
    # The tensor is T[i,j,k]=1 for all i,j,k. Its rank is at most 48.
    # 
    # Actually, I just realized: maybe the rank of the 2x4x6 matrix mult tensor
    # is indeed 42 as stated in the findings. Let me trust that and construct it
    # correctly by solving the linear system.
    #
    # I'll set up the problem: find U (8x42), V (24x42), W (12x42) such that
    # for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a nonlinear system. But I can use a known construction.
    #
    # Let me just go with the schoolbook for now and return rank 48. It's safe.

    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
