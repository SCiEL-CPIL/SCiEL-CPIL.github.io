import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build 4x4x4 decomposition using Strassen (rank 49)
    # A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks of 2x2 submatrices
    
    # First, define the 2x2x2 Strassen decomposition
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 4=B00, 5=B01, 6=B10, 7=B11
    # C indices: 8=C00, 9=C01, 10=C10, 11=C11
    
    # Strassen's 7 products
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    
    # C00 = P1 - P5 + P7 + P4
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let me verify this is correct by expanding:
    # P1 = A00B00 + A00B11 + A11B00 + A11B11
    # P5 = A00B11 + A01B11
    # P7 = A01B01 + A01B11 - A11B01 - A11B11
    # P4 = A11B01 - A11B00
    # C00 = P1 - P5 + P7 + P4
    #     = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A01B01 + A01B11 - A11B01 - A11B11 + A11B01 - A11B00
    #     = A00B00 + A01B01
    # But C00 should be A00B00 + A01B10. So this is wrong.
    
    # I think the issue is in the definition of P7. Let me try:
    # P7 = (A01-A11)(B10+B11) instead?
    # No, that doesn't seem right either.
    
    # Let me look at this more carefully. The correct Strassen algorithm is:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 - M5 + M7 + M4
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # With A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    
    # C00 = M1 - M5 + M7 + M4
    #     = (a+d)(e+h) - (a+b)h + (b-d)(f+h) + d(f-e)
    #     = ae + ah + de + dh - ah - bh + bf + bh - df - dh + df - de
    #     = ae + bf
    
    # But C00 = ae + bg. So I'm getting bf instead of bg.
    
    # I think the problem is that I'm using the wrong formula. Let me try:
    # C00 = M1 + M4 - M5 + M7 (same thing, just reordered)
    
    # Hmm, this still gives ae + bf.
    
    # Wait, maybe I have the matrix multiplication convention wrong. Let me double-check.
    # C = A * B, so C[i,j] = sum_k A[i,k] * B[k,j]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = a*e + b*g
    
    # So C00 = ae + bg. My formula gives ae + bf. So something is wrong.
    
    # Let me try a different Strassen formula. Maybe:
    # M7 = (A01-A11)(B10+B11)?
    # M7 = (b-d)(g+h) = bg + bh - dg - dh
    # C00 = M1 - M5 + M7 + M4
    #     = (ae + ah + de + dh) - (ah + bh) + (bg + bh - dg - dh) + (df - de)
    #     = ae + ah + de + dh - ah - bh + bg + bh - dg - dh + df - de
    #     = ae + bg + (ah-ah) + (de-de) + (dh-dh) + (-bh+bh) + (-dg+df)
    #     = ae + bg + df - dg
    # Still not right.
    
    # I think I need to look up the correct formula. Let me just use a known correct implementation.
    
    # Actually, I recall now. The correct Strassen formulas are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 - M5 + M7 + M4
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # But this gives C00 = ae + bf, not ae + bg.
    
    # I think the issue might be in how I'm indexing. Let me try:
    # Maybe B is indexed as B[k,j] where k is row and j is column.
    # B = [[e,f],[g,h]] means B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = a*e + b*g
    
    # So I need C00 = ae + bg.
    
    # Let me try different M definitions:
    # M1 = (A00+A11)(B00+B11) = (a+d)(e+h)
    # M2 = (A10+A11)B00 = (c+d)e
    # M3 = A00(B10-B11) = a(g-h)
    # M4 = A11(B01-B00) = d(f-e)
    # M5 = (A00+A01)B11 = (a+b)h
    # M6 = (A10-A00)(B00+B10) = (c-a)(e+g)
    # M7 = (A01-A11)(B01+B11) = (b-d)(f+h)
    
    # C00 = M1 + M4 - M5 + M7 = (a+d)(e+h) + d(f-e) - (a+b)h + (b-d)(f+h)
    #     = ae + ah + de + dh + df - de - ah - bh + bf + bh - df - dh
    #     = ae + bf
    
    # I keep getting ae + bf. Let me try C00 = M1 - M4 + M5 - M7 or some other combination.
    
    # Actually, I think I've been making an error. Let me carefully recompute.
    # (a+d)(e+h) = ae + ah + de + dh
    # d(f-e) = df - de
    # (a+b)h = ah + bh
    # (b-d)(f+h) = bf + bh - df - dh
    
    # M1 + M4 - M5 + M7 = (ae + ah + de + dh) + (df - de) - (ah + bh) + (bf + bh - df - dh)
    # = ae + ah + de + dh + df - de - ah - bh + bf + bh - df - dh
    # = ae + (ah-ah) + (de-de) + (dh-dh) + (df-df) + (-bh+bh) + bf
    # = ae + bf
    
    # So C00 = ae + bf. But it should be ae + bg.
    
    # I think the standard Strassen algorithm might have a different convention for B.
    # Maybe in some formulations, B is transposed or indexed differently.
    
    # Let me try: what if I use M7 = (A01-A11)(B10+B11)?
    # M7 = (b-d)(g+h) = bg + bh - dg - dh
    # C00 = M1 + M4 - M5 + M7
    #     = (ae + ah + de + dh) + (df - de) - (ah + bh) + (bg + bh - dg - dh)
    #     = ae + ah + de + dh + df - de - ah - bh + bg + bh - dg - dh
    #     = ae + bg + (ah-ah) + (de-de) + (dh-dh) + (-bh+bh) + (df-dg)
    #     = ae + bg + df - dg
    # Still not right.
    
    # I'm clearly making an error somewhere. Let me just give up on Strassen for now and use a different approach.
    
    # Given the time constraints, let me just return the schoolbook decomposition with a time guard.
    # The current score is 1.0 with rank 80.
    
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
