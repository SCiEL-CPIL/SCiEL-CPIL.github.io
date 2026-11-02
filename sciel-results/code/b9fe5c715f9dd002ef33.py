import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n, m, p = 3, 4, 8  # A is n x m, B is m x p, C is n x p
    
    # We'll construct the tensor decomposition using a block approach.
    # The key is to find a low-rank decomposition for 2x2x2 (Strassen) and use it.
    #
    # Strassen's 7-product algorithm for 2x2 matrix multiplication:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # 
    # m1 = (a+b)*(e+f)
    # m2 = (c+d)*(e+f)  
    # m3 = a*(h-f)
    # m4 = d*(g-e)
    # m5 = (a+d)*(e+h)
    # m6 = (b+d)*(g-h)
    # m7 = (a+c)(f-h)
    #
    # Result:
    # [[a*e + b*g, a*f + b*h],
    #  [c*e + d*g, c*f + d*h]]
    #
    # = [[m1 + m4 - m5 + m7, m3 + m5],
    #    [m2 + m5 - m1 - m7, m2 + m4]]
    #
    # Wait, let me double-check the standard Strassen formulas:
    # m1 = (a+b)*(e+f) = ae+af+be+bf
    # m2 = (c+d)*(e+f) = ce+cf+de+df
    # m3 = a*(h-f) = ah-af
    # m4 = d*(g-e) = dg-de
    # m5 = (a+d)*(e+h) = ae+ah+de+dh
    # m6 = (b+d)*(g-h) = bg-bh+dg-dh
    # m7 = (a+c)*(f-h) = af-ah+cf-ch
    #
    # C11 = ae + bg = (m1-m2+m4+m7... no, let me look up the exact formula)
    #
    # Standard Strassen:
    # C11 = m1 - m2 + m4 + m7... 
    # Actually, the standard is:
    # C11 = m1 - m2 + m4 - m7  ... no
    #
    # Let me just verify directly.
    # C11 = ae + bg
    # C12 = af + bh
    # C21 = ce + dg
    # C22 = cf + dh
    #
    # m1 = ae+af+be+bf
    # m2 = ce+cf+de+df
    # m3 = ah-af
    # m4 = dg-de
    # m5 = ae+ah+de+dh
    # m6 = bg-bh+dg-dh
    # m7 = af-ah+cf-ch
    #
    # C11 = ae+bg. Can we express this?
    # m1 - m2 + m4 + m7 = ae+af+be+bf - ce-cf-de-df + dg-de + af-ah+cf-ch
    # That's getting messy. Let me use a known correct set.
    #
    # Actually, the well-known Strassen algorithm is:
    # p1 = (a + d)(e + h)
    # p2 = (c + d)(g + h)
    # p3 = (a + b)(e)
    # p4 = (d - b)(g + h)
    # p5 = (a + c)(e + f)
    # p6 = (a - c)(f - h)
    # p7 = (b - d)(e + f)
    #
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p6
    # C21 = p2 - p4
    # C22 = p1 - p2 + p3 + p5
    #
    # Let me verify:
    # p1 = ae+ah+de+dh
    # p2 = cg+ch+dg+dh
    # p3 = ae+be
    # p4 = dg+dh-bg-bh
    # p5 = ae+af+ce+cf
    # p6 = af-ah-cf+ch
    # p7 = be+bf-de-df
    #
    # C11 = p1+p4-p5+p7 = (ae+ah+de+dh) + (dg+dh-bg-bh) - (ae+af+ce+cf) + (be+bf-de-df)
    # = ae+ah+de+dh+dg+dh-bg-bh-ae-af-ce-cf+be+bf-de-df
    # = ah+2dh-bg-bh-af-ce-cf+be+bf-df
    # That doesn't look right. I'm confusing myself.
    #
    # Let me use the most standard reference formulation:
    # m1 = (a+b)(e+f)
    # m2 = (c+d)(e+f)
    # m3 = a(h-f)
    # m4 = d(g-e)
    # m5 = (a+d)(e+h)
    # m6 = (b+d)(g-h)
    # m7 = (a+c)(f-h)
    #
    # C11 = m1 - m2 + m4 + m7
    # C12 = m3 + m5
    # C21 = m2 - m5 + m6 + m7
    # C22 = m1 - m3 + m5 - m6
    #
    # Verify C11:
    # m1-m2+m4+m7 = (ae+af+be+bf) - (ce+cf+de+df) + (dg-de) + (af-ah+cf-ch)
    # = ae+af+be+bf-ce-cf-de-df+dg-de+af-ah+cf-ch
    # = ae+2af+be+bf-ce-de-df+dg-ah+cf-ch+... 
    # Hmm, this is getting complicated. Let me just hardcode a known correct rank-7 decomposition.
    #
    # I'll use the formulation from Baur-Strassen:
    # For A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # 
    # w1 = (a+b)(e+f)     -> U_A=[1,1,0,0], U_B=[1,1,0,0]
    # w2 = (c+d)(e+f)     -> U_A=[0,0,1,1], U_B=[1,1,0,0]
    # w3 = a(h-f)         -> U_A=[1,0,0,0], U_B=[0,-1,0,1]
    # w4 = d(g-e)         -> U_A=[0,0,0,1], U_B=[-1,0,1,0]
    # w5 = (a+d)(e+h)     -> U_A=[1,0,0,1], U_B=[1,0,0,1]
    # w6 = (b+d)(g-h)     -> U_A=[0,1,0,1], U_B=[0,0,1,-1]
    # w7 = (a+c)(f-h)     -> U_A=[1,0,1,0], U_B=[0,1,0,-1]
    #
    # C11 = w1 - w2 + w4 - w7
    # C12 = w3 + w5
    # C21 = w2 - w5 + w6 + w7
    # C22 = w1 - w3 + w5 - w6
    #
    # Let me verify C11 = ae+bg:
    # w1-w2+w4-w7 = (ae+af+be+bf) - (ce+cf+de+df) + (dg-de) - (af-ah+cf-ch)
    # = ae+af+be+bf-ce-cf-de-df+dg-de-af+ah-cf+ch
    # = ae+be+bf-ce-de-df+dg+ah-cf+ch
    # = ae+be+bf-ce-2de-df+dg+ah-cf+ch
    # That's not ae+bg. So this formulation is wrong.
    #
    # I think the issue is I'm mixing up different Strassen variants. Let me just use the one I can verify.
    #
    # Actually, for the purposes of this task, I'll use the schoolbook decomposition for everything.
    # The rank is 96, which gives score 1.0. It's guaranteed correct.
    #
    # If I want to try to improve, I could use a Kronecker product structure:
    # M_{3,4,8} = M_{3,2,4} ⊗ M_{1,2,2}
    # But M_{1,2,2} has rank 2 (schoolbook: 1*2*2=2, and rank can't be less than 2 since we need to compute 2 outputs from 2 inputs with 1 intermediate).
    # Actually M_{1,2,2}: A is 1x2, B is 2x2, C is 1x2.
    # C = A*B, so C[0,0] = A[0,0]*B[0,0]+A[0,1]*B[1,0], C[0,1] = A[0,0]*B[0,1]+A[0,1]*B[1,1].
    # Schoolbook rank = 4. Can we do better? The tensor is M_{1,2,2} which has rank at most 4.
    # Lower bound: it contains M_{1,1,1} (rank 1), so rank >= 1. But we need 4 independent terms? 
    # Actually, the border rank might be lower, but we need exact rank.
    # For M_{1,2,2}, I believe the rank is 4 (no Strassen-like shortcut for 1x2*2x2).
    #
    # So M_{3,2,4} ⊗ M_{1,2,2} would have rank rank(M_{3,2,4}) * 4.
    # rank(M_{3,2,4}) <= 3*2*4 = 24. So total <= 96. Not an improvement.
    #
    # What about M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}?
    # M_{1,1,2}: A is 1x1, B is 1x2, C is 1x2. C[0,0]=A*B[0,0], C[0,1]=A*B[0,1]. Rank 2.
    # rank(M_{3,4,4}) * 2 <= 48*2 = 96. Not an improvement.
    #
    # I'll stick with schoolbook rank 96.

    rank = n * m * p  # 96
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {s}")
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        raise TimeoutError("Too slow")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
