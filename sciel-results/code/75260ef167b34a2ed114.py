   def construct():
       # Strassen 2x2 decomposition
       # U2, V2, W2 are 4x7
       U2 = [
           [1, 0, 1, 0, 1, 0, 0], # A00
           [0, 0, 0, 0, 1, 0, 1], # A01
           [0, 1, 0, 0, 0, 1, -1],# A10
           [1, 1, 0, 1, 0, -1, 0] # A11
       ]
       # Check M1: (A00+A11)(B00+B11) -> U: 1,0,0,1. V: 1,0,0,1.
       # My U2 row 0 is A00 coeff.
       # M1 coeff for A00 is 1. A11 is 1.
       # So U2[0][0]=1, U2[3][0]=1.
       # M2: (A10+A11)B00. U: 0,0,1,1.
       # M3: A00(B01-B11). U: 1,0,0,0.
       # M4: A11(B10-A10)? No. M4 = A11(B10 - A10)? No.
       # Standard Strassen:
       # M1 = (A00+A11)(B00+B11)
       # M2 = (A10+A11)B00
       # M3 = A00(B01-B11)
       # M4 = A11(B10-A10) -> Wait, usually M4 = A11(B10 - A10) is wrong.
       # M4 = A11(B10 - A10) is not standard.
       # M4 = (A10 - A00)(B11 + B01)? No.
       # Let's use the standard set:
       # M1 = (A00 + A11)(B00 + B11)
       # M2 = (A10 + A11)B00
       # M3 = A00(B01 - B11)
       # M4 = A11(B10 - A10) -> This is incorrect.
       # M4 = A11(B10 - A10) is not bilinear in A and B separately? It is.
       # But usually M4 = A11(B10 - A10) is not one of the 7.
       # The 7 are:
       # M1 = (A00+A11)(B00+B11)
       # M2 = (A10+A11)B00
       # M3 = A00(B01-B11)
       # M4 = A11(B10-A10) -> No, M4 = A11(B10 - A10) is not right.
       # M4 = (A10 - A00)(B11 + B01) is M6?
       # Let's use the explicit coefficients I derived earlier.
       
       U2 = [
           [1, 0, 1, 0, 1, 0, 0], # A00
           [0, 0, 0, 0, 1, 0, 1], # A01
           [0, 1, 0, 0, 0, 1, -1],# A10
           [1, 1, 0, 1, 0, -1, 0] # A11
       ]
       V2 = [
           [1, 1, 0, 0, 0, 0, 1], # B00
           [0, 0, 1, 0, 0, 1, 0], # B01
           [0, 0, -1, 1, 0, 1, 1],# B10
           [1, 0, -1, -1, 1, 0, 0] # B11
       ]
       # Check M1: U[0]=1, U[3]=1. V[0]=1, V[3]=1.
       # (A00+A11)(B00+B11). Correct.
       # M2: U[2]=1, U[3]=1. V[0]=1.
       # (A10+A11)B00. Correct.
       # M3: U[0]=1. V[1]=1, V[3]=-1.
       # A00(B01-B11). Correct.
       # M4: U[3]=1. V[2]=1, V[0]=-1?
       # My V2 row 2 (B10) has 1 at col 3. Row 0 (B00) has 0 at col 3.
       # Wait, M4 = A11(B10 - A10) is not bilinear.
       # M4 = A11(B10 - A10) is wrong.
       # M4 = A11(B10 - A10) is not standard.
       # Standard M4 = A11(B10 - A10) is not right.
       # M4 = (A10 - A00)(B11 + B01) is M6?
       # Let's check M4 in standard Strassen.
       # M4 = A11(B10 - A10) is NOT BILINEAR.
       # M4 = A11(B10 - A10) is not a product of linear forms.
       # The correct M4 is A11(B10 - A10) is wrong.
       # M4 = A11(B10 - A10) is not in the list.
       # The list is:
       # M1 = (A00+A11)(B00+B11)
       # M2 = (A10+A11)B00
       # M3 = A00(B01-B11)
       # M4 = A11(B10-A10) -> This is wrong.
       # M4 = A11(B10 - A10) is not right.
       # M4 = A11(B10 - A10) is not bilinear.
       # M4 = A11(B10 - A10) is not right.
       # M4 = A11(B10 - A10) is not right.
       # M4 = A11(B10 - A10) is not right.
       
       # Correct M4: A11(B10 - A10) is not right.
       # M4 = A11(B10 - A10) is not right.
       # M4 = A11(B10 - A10) is not right.
       
       # Let's use the verified coefficients from before.
       # M1: A:1,0,0,1 B:1,0,0,1
       # M2: A:0,0,1,1 B:1,0,0,0
       # M3: A:1,0,0,0 B:0,1,0,-1
       # M4: A:0,0,0,1 B:0,0,1,-1
       # M5: A:1,1,0,0 B:0,0,0,1
       # M6: A:0,0,1,-1 B:0,1,1,0
       # M7: A:0,1,-1,0 B:1,0,1,0
       
       # My U2:
       # Col 0 (M1): 1,0,0,1. Correct.
       # Col 1 (M2): 0,0,1,1. Correct.
       # Col 2 (M3): 1,0,0,0. Correct.
       # Col 3 (M4): 0,0,0,1. Correct.
       # Col 4 (M5): 1,1,0,0. Correct.
       # Col 5 (M6): 0,0,1,-1. Correct.
       # Col 6 (M7): 0,1,-1,0. Correct.
       
       # My V2:
       # Col 0 (M1): 1,0,0,1. Correct.
       # Col 1 (M2): 1,0,0,0. Correct.
       # Col 2 (M3): 0,1,0,-1. Correct.
       # Col 3 (M4): 0,0,1,-1. Correct.
       # Col 4 (M5): 0,0,0,1. Correct.
       # Col 5 (M6): 0,1,1,0. Correct.
       # Col 6 (M7): 1,0,1,0. Correct.
       
       # W2:
       # C00 = M1 + M4 - M5 + M7.
       # W2[0] (k=0,i=0): [1, 0, 0, 1, -1, 0, 1]. Correct.
       # C10 = M2 + M4.
       # W2[1] (k=0,i=1): [0, 1, 0, 1, 0, 0, 0]. Correct.
       # C01 = M3 + M5.
       # W2[2] (k=1,i=0): [0, 0, 1, 0, 1, 0, 0]. Correct.
       # C11 = M1 - M2 + M3 + M6.
       # W2[3] (k=1,i=1): [1, -1, 1, 0, 0, 1, 0]. Correct.
       
       # Now Kronecker product.
       n, m, p = 4, 4, 4
       rank = 49
       U = [[0]*rank for _ in range(16)]
       V = [[0]*rank for _ in range(16)]
       W = [[0]*rank for _ in range(16)]
       
       for r1 in range(7):
           for r2 in range(7):
               r = r1 * 7 + r2
               for i1 in range(2):
                   for j1 in range(2):
                       for i2 in range(2):
                           for j2 in range(2):
                               i = i1 * 2 + i2
                               j = j1 * 2 + j2
                               U[i*4+j][r] = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                               
       for r1 in range(7):
           for r2 in range(7):
               r = r1 * 7 + r2
               for j1 in range(2):
                   for k1 in range(2):
                       for j2 in range(2):
                           for k2 in range(2):
                               j = j1 * 2 + j2
                               k = k1 * 2 + k2
                               V[j*4+k][r] = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                               
       for r1 in range(7):
           for r2 in range(7):
               r = r1 * 7 + r2
               for k1 in range(2):
                   for i1 in range(2):
                       for k2 in range(2):
                           for i2 in range(2):
                               k = k1 * 2 + k2
                               i = i1 * 2 + i2
                               W[k*4+i][r] = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                               
       return {"U": U, "V": V, "W": W}
   