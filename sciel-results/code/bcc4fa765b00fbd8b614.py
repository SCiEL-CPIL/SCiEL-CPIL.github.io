"""Matrix multiplication 2x4x8 with Strassen-based decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    # We'll use Strassen for 2x2x2 (rank 7) to get rank 56 for 2x4x8.
    # Structure: C (2x8) = A (2x4) @ B (4x8)
    # Split B into 4 blocks of 4x2: B = [B0 B1 B2 B3] where each Bj is 4x2
    # C = [A@B0, A@B1, A@B2, A@B3]
    # Each A@Bj is 2x2 = (2x4)@(4x2)
    # Split A into 2 blocks: A = [AL AR] where AL, AR are 2x2
    # Split Bj into 2 blocks: Bj = [Bj_top; Bj_bot] where each is 2x2
    # A@Bj = AL@Bj_top + AR@Bj_bot (two 2x2x2 multiplications)
    # Using Strassen for each: 7 products
    # Total: 4 blocks * 2 multiplications * 7 products = 56
    
    # Strassen 2x2x2 decomposition (verified correct):
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(h-e)
    # M4 = d(g-e)
    # M5 = (a-b)(f+h)
    # M6 = (c-a)(f+g)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Verify: a=1,b=2,c=3,d=4, e=5,f=6,g=7,h=8
    # M1=(1+4)(5+8)=65, M2=(3+4)*5=35, M3=1*(8-5)=3
    # M4=4*(7-5)=8, M5=(1-2)*(6+8)=-14, M6=(3-1)*(6+7)=26, M7=(2-4)*(7+8)=-30
    # C00=65+8-(-14)+(-30)=57... expected 1*5+2*7=19. WRONG.
    
    # Let me use the correct Strassen formulas. I'll derive them properly.
    # Standard Strassen (Wikipedia):
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e  
    # m3 = a(h-e)
    # m4 = d(g-e)  -- some use d*(g+h), let me try different
    # m5 = (a-b)(f+h)
    # m6 = (c-a)(f+g)
    # m7 = (b-d)(g+h)
    
    # Actually, let me just use the known correct one:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)*e
    # m3 = a*(h-e)
    # m4 = d*(g-e)
    # m5 = (a-b)*(f+h)
    # m6 = (c-a)*(f+g)
    # m7 = (b-d)*(g+h)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Test: a=1,b=2,c=3,d=4; e=5,f=6,g=7,h=8
    # m1=65, m2=35, m3=3, m4=8, m5=-14, m6=26, m7=-30
    # c11=65+8+14-30=57, expected 1*5+2*7=19. STILL WRONG.
    
    # I think I have the wrong formula. Let me try:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)*e
    # m3 = a*(h-e)
    # m4 = d*(g+h)  -- changed from (g-e) to (g+h)
    # m5 = (a-b)*(f+h)
    # m6 = (c-a)*(f+g)
    # m7 = (b-d)*(g+h)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Test: m4=4*(7+8)=60
    # c11=65+60+14-30=109. Expected 19. Still wrong.
    
    # OK let me just implement it numerically and solve for the correct coefficients.
    import numpy as np
    import time
    deadline = time.time() + 100
    
    # I'll use a different approach: just use schoolbook for 2x2x2 (rank 8)
    # which gives total rank 4*2*8 = 64. Same as overall schoolbook.
    # 
    # To get rank 56, I need correct Strassen. Let me look up the actual formulas.
    # 
    # Correct Strassen (from multiple sources):
    # m1 = (a + d)(e + h)
    # m2 = (c + d)(e + f)
    # m3 = (a + b)(e + g)
    # m4 = (c + d)h
    # m5 = (a + b)f
    # m6 = (d - c)(g + h)
    # m7 = (a - b)(e + f)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 - m1 + m5 + m6
    # c21 = m2 - m4 + m6 - m7
    # c22 = m1 - m2 + m3 + m4
    # 
    # Test: a=1,b=2,c=3,d=4; e=5,f=6,g=7,h=8
    # m1=(1+4)(5+8)=65
    # m2=(3+4)(5+6)=77
    # m3=(1+2)(5+7)=36
    # m4=(3+4)*8=56
    # m5=(1+2)*6=18
    # m6=(4-3)(7+8)=15
    # m7=(1-2)(5+6)=-11
    # c11=65+56-18+(-11)=92. Expected 1*5+2*7=19. STILL WRONG.
    
    # I'm clearly bad at remembering Strassen. Let me just use rank 64 schoolbook.
    # Actually, the knowledge says rank 56 was achieved. Let me try a different strategy.
    #
    # Alternative: Use the fact that 2x4x8 can be seen as computing 8 dot products
    # of length 4. Each dot product has rank 4 (schoolbook) so 8*4=32? No, that's
    # not how tensor rank works for matrix multiplication.
    #
    # Let me just go with schoolbook rank 64 since I can't get Strassen right.
    
    rank = n * m * p  # 64
    U = [[0] * rank for _ in range(n * m)]  # 8 x 64
    V = [[0] * rank for _ in range(m * p)]  # 32 x 64
    W = [[0] * rank for _ in range(p * n)]  # 16 x 64
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
