import time
import numpy as np
from itertools import product

def find_strassen():
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C[0,0] = ae+bg, C[0,1] = af+bh, C[1,0] = ce+dg, C[1,1] = cf+dh
    
    # Try different sets of 7 bilinear forms
    # Variables: a,b,c,d,e,f,g,h
    # Each Mj is a bilinear form: sum over pairs (x,y) where x in {a,b,c,d} and y in {e,f,g,h}
    
    # Let me define the 7 standard Strassen products and check if they span the right space
    # Actually, let me just try to verify the known formula works with a different C indexing
    
    # Maybe the issue is that C is indexed as C[j,i] instead of C[i,j]?
    # Or maybe B is transposed?
    
    # Let me try: what if C[0,0] = af+bg? No, that doesn't make sense for matrix mult.
    
    # I think I need to look at this from scratch. Let me define:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B, so C[i,j] = sum_k A[i,k]*B[k,j]
    # C[0,0] = A00*B00 + A01*B10
    # C[0,1] = A00*B01 + A01*B11
    # C[1,0] = A10*B00 + A11*B10
    # C[1,1] = A10*B01 + A11*B11
    
    # With A00=a, A01=b, A10=c, A11=d, B00=e, B01=f, B10=g, B11=h:
    # C[0,0] = ae + bg
    # C[0,1] = af + bh
    # C[1,0] = ce + dg
    # C[1,1] = cf + dh
    
    # Now let me define 7 products and see if I can find signs that work
    # I'll try the standard Strassen products but with different B indexing
    
    # What if I use B00=e, B01=g, B10=f, B11=h? (i.e., B is transposed)
    # Then C[0,0] = A00*B00 + A01*B10 = ae + bf
    # C[0,1] = A00*B01 + A01*B11 = ag + bh
    # C[1,0] = A10*B00 + A11*B10 = ce + df
    # C[1,1] = A10*B01 + A11*B11 = cg + dh
    
    # That doesn't help either.
    
    # Let me just try all 2^7 = 128 sign combinations for C[0,0] with the standard M's
    # and see if any give ae+bg
    
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M2 = (c+d)e = ce+de
    # M3 = a(g-h) = ag-ah
    # M4 = d(f-e) = df-de
    # M5 = (a+b)h = ah+bh
    # M6 = (c-a)(e+g) = ce+cg-ae-ag
    # M7 = (b-d)(f+h) = bf+bh-df-dh
    
    # For C[0,0] = ae+bg, I need:
    # Coeff of ae: 1, ah: 0, de: 0, dh: 0, ce: 0, ag: 0, df: 0, bh: 0, bf: 0, cg: 0, bg: 1
    # But bg doesn't appear in any Mj! So the standard Strassen products can't produce bg.
    
    # This means I have the wrong Strassen products. Let me try different ones.
    
    # What if M3 = (A00+A10)(B10-B11) = (a+c)(g-h) = ag-ah+cg-ch?
    # Still no bg.
    
    # I think the issue is that I need products that involve b and g together.
    # Let me try: M3 = A01*B10 = bg? But that's just one term, not a product of sums.
    
    # Actually, for a rank-7 decomposition, each term is a product of a linear form in A and a linear form in B.
    # So Mj = (linear form in a,b,c,d) * (linear form in e,f,g,h)
    
    # Let me try to find 7 such products that span the 4-dimensional space of C entries.
    
    # Actually, I think the correct Strassen algorithm uses:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # And the reconstruction is:
    # C00 = M1 - M5 + M7 + M4
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # But as I showed, C00 = ae+bf, not ae+bg.
    
    # I'm going to try a completely different approach: just use the schoolbook method with rank 80,
    # and then try to reduce the rank using search.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 5
    
    # Schoolbook decomposition: rank = n*m*p = 80
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
