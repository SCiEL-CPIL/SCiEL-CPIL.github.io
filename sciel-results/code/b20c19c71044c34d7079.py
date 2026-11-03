import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    schoolbook_rank = n * m * p  # 18
    
    def get_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Known rank-17 decomposition for 2x3x3 matrix multiplication
    # This uses a combination of Strassen-like tricks for the overlapping parts
    # and schoolbook for the rest
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strategy: 
    # Split the 2x3x3 multiplication into blocks:
    # A is 2x3, B is 3x3, C is 2x3
    # 
    # Let's use a hybrid approach:
    # For the first 2 columns of B (and corresponding products), use a 
    # Strassen-like decomposition where possible, and schoolbook for the rest.
    #
    # Actually, let me use a known construction:
    # 
    # The tensor for 2x3x3 has been shown to have rank 17.
    # One construction:
    # - Use schoolbook for 14 of the 18 products
    # - Use 3 extra rank-1 terms to handle the remaining 4 products
    #   (since Strassen can do 2x2x2 with rank 7 instead of 8, saving 1,
    #    but we have a more complex structure)
    #
    # Let me try a different approach: explicit rank-17 construction
    #
    # Decomposition:
    # C = A * B where A is 2x3, B is 3x3
    #
    # Write A = [A1 A2 A3] where Ai is 2x1
    # Write B = [B1 B2 B3] where Bj is 3x1
    #
    # C = A1*B1 + A2*B2 + A3*B3
    #
    # Each Ai*Bj is a 2x1 * 3x1 = outer product, giving 2x3
    # 
    # For the 2x1 * 3x1 outer product, we need rank 6 (schoolbook) or...
    # Actually, a 2x3 outer product from 2x1 and 3x1 vectors has rank 1
    # as a tensor? No, wait.
    #
    # Let me think differently. The multiplication tensor T has entries T[i,j,k]
    # where i is row of A (0,1), j is col of A / row of B (0,1,2), k is col of B (0,1,2)
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # So the tensor is T[i,j,k] = delta (meaning it's 1 for each (i,j,k) triple)
    # and C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    #
    # This is a standard matrix multiplication tensor with dimensions 2,3,3.
    #
    # Known result: rank of 2x3x3 multiplication tensor is 17.
    #
    # Here's one explicit construction using the following idea:
    # Use 1 schoolbook term for each (i,j,k) except we can combine some.
    #
    # Actually, let me use a verified construction:
    # 
    # The 18 schoolbook terms are:
    # (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    #
    # We can save 1 rank by using a Strassen-like trick on a 2x2x2 sub-block.
    # Consider the sub-tensor for i in {0,1}, j in {0,1}, k in {0,1}:
    # This is a 2x2x2 multiplication tensor, which has rank 7 (Strassen) instead of 8.
    # The remaining terms: j=2 for all i,k (6 terms) plus we already handled j=0,1 for k=0,1
    # Wait, that's not quite right.
    
    # Let me just construct the rank-17 directly:
    # 
    # Terms 1-8: Strassen for the 2x2x2 sub-problem (i in {0,1}, j in {0,1}, k in {0,1})
    # Strassen's 7 terms for 2x2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    
    # In our indexing:
    # A[i,j], B[j,k], C[i,k]
    # For the sub-block: i in {0,1}, j in {0,1}, k in {0,1}
    # A00 = A[0,0], A01 = A[0,1], A10 = A[1,0], A11 = A[1,1]
    # B00 = B[0,0], B01 = B[0,1], B10 = B[1,0], B11 = B[1,1]
    # C00, C01, C10, C11
    
    # Strassen terms (rank 7):
    # Term 1: U: A[0,0]=1, A[1,1]=1; V: B[0,0]=1, B[1,1]=1; W: C[0,0]=1, C[0,1]=-1, C[1,0]=-1, C[1,1]=1
    # Wait, I need to be more careful.
    
    # Let me use a simpler approach. I'll implement the known rank-17 decomposition
    # from the literature.
    
    # Actually, for a 2x3x3 multiplication, here's a clean construction:
    # 
    # We have 18 schoolbook terms. We can reduce to 17 by:
    # Using the identity: (a+b)(c+d) - ac - bd = ad + bc
    # This allows us to compute 2 products with 3 rank-1 terms instead of 4.
    #
    # Let's pair up some terms:
    # Consider C[0,0] and C[1,1]:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # These don't share factors nicely.
    
    # Let me try a different pairing. Consider the 2x2x2 sub-tensor again.
    # For i in {0,1}, j in {0,1}, k in {0,1}, we have 8 schoolbook terms.
    # Using Strassen, we can do this with 7 terms.
    # The remaining 10 terms are for j=2 (6 terms) and... wait.
    #
    # Total terms: 18
    # Sub-block (i in {0,1}, j in {0,1}, k in {0,1}): 8 terms -> 7 with Strassen
    # Remaining: 18 - 8 = 10 terms
    # Total: 7 + 10 = 17. 
    
    # So the construction is:
    # 7 Strassen terms for the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1})
    # 10 schoolbook terms for the rest
    
    # Let me define the 10 schoolbook terms first:
    col = 0
    # Terms where j=2 (all i, k): 2*3 = 6 terms
    for i in range(2):
        for k in range(3):
            U[i * 3 + 2, col] = 1  # A[i, 2]
            V[2 * 3 + k, col] = 1  # B[2, k]
            W[k * 2 + i, col] = 1  # C[i, k]
            col += 1
    
    # Terms where j in {0,1} and k=2: 2*2 = 4 terms
    for i in range(2):
        for j in range(2):
            U[i * 3 + j, col] = 1  # A[i, j]
            V[j * 3 + 2, col] = 1  # B[j, 2]
            W[2 * 2 + i, col] = 1  # C[i, 2]
            col += 1
    
    # Now the 7 Strassen terms for the 2x2x2 sub-block
    # i in {0,1}, j in {0,1}, k in {0,1}
    
    # Strassen's 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    
    # And C is reconstructed as:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me verify:
    # C00 = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B01+B11)
    #     = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B01 + A01B11 - A11B01 - A11B11
    #     = A00B00 + A11B10 + A01B01
    # Hmm, that's not right. Let me re-derive Strassen.
    
    # Strassen for 2x2:
    # C = A * B
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # 
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M2 = (c+d)e = ce+de
    # M3 = a(f-h) = af-ah
    # M4 = d(g-e) = dg-de
    # M5 = (a+b)h = ah+bh
    # M6 = (c-a)(e+g) = ce+cg-ae-ag
    # M7 = (b-d)(f+h) = bf+bh-df-dh
    # 
    # C00 = M1 + M4 - M5 + M7 = (ae+ah+de+dh) + (dg-de) - (ah+bh) + (bf+bh-df-dh)
    #     = ae+ah+de+dh+dg-de-ah-bh+bf+bh-df-dh
    #     = ae + dg + bf - df
    # That's not right either. Let me look up the correct Strassen formulas.
    
    # Correct Strassen:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)(e + f)
    # M3 = (a + b)(g + h)
    # M4 = (c + b)(e + g)
    # M5 = (a + d)(f + h)
    # M6 = (c + a)(e + h)  -- wait, this isn't standard either.
    
    # Let me use the standard 7:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)(e+f)
    # P3 = (a+b)(g+h)
    # P4 = (c+b)(e+g)
    # P5 = (a+d)(f+h)
    # P6 = (c-a)(e+h)
    # P7 = (b-d)(f+h)
    # 
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5 - P1 - P2
    # C10 = P2 + P4 - P3 - P7
    # C11 = P1 + P3 - P2 - P6
    # 
    # Let me verify C00:
    # P1 + P4 - P5 + P7
    # = (a+d)(e+h) + (c+b)(e+g) - (a+d)(f+h) + (b-d)(f+h)
    # = ae+ah+de+dh + ce+cg+be+bg - af-ah-df-dh + bf+bh-df-dh
    # = ae + de + ce + cg + be + bg - af - df - bf - dh + bh - dh
    # Hmm, this is getting messy. Let me just use a verified set.
    
    # Actually, let me use a simpler approach. I'll just use the known fact that
    # the rank is 17 and construct it using a different method.
    
    # Alternative: use 17 schoolbook-like terms with one linear combination.
    
    # Actually, let me just return the schoolbook rank-18 for now, since I'm 
    # not confident in my Strassen construction for this specific case.
    # The inspirations mention rank 17 is achievable, but I need to get it right.
    
    # Let me try one more approach. Here's a clean rank-17 construction:
    
    # Reset
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    # 10 schoolbook terms as before
    for i in range(2):
        for k in range(3):
            U[i * 3 + 2, col] = 1
            V[2 * 3 + k, col] = 1
            W[k * 2 + i, col] = 1
            col += 1
    
    for i in range(2):
        for j in range(2):
            U[i * 3 + j, col] = 1
            V[j * 3 + 2, col] = 1
            W[2 * 2 + i, col] = 1
            col += 1
    
    # Now 7 terms for the 2x2x2 sub-block using Strassen
    # Let a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C[0,0]=ae+bg, C[0,1]=af+bh, C[1,0]=ce+dg, C[1,1]=cf+dh
    
    # Strassen terms:
    # T1: (a+d)(e+h) -> U: a=1,d=1; V: e=1,h=1; W: contributes to C
    # T2: (c+d)(e+f) -> U: c=1,d=1; V: e=1,f=1
    # T3: (a+b)(g+h) -> U: a=1,b=1; V: g=1,h=1
    # T4: (c+b)(e+g) -> U: c=1,b=1; V: e=1,g=1
    # T5: (a+d)(f+h) -> U: a=1,d=1; V: f=1,h=1
    # T6: (c-a)(e+h) -> U: c=1,a=-1; V: e=1,h=1
    # T7: (b-d)(f+h) -> U: b=1,d=-1; V: f=1,h=1
    
    # C[0,0] = T1 + T4 - T5 + T7
    # C[0,1] = T3 + T5 - T1 - T2  -- let me verify
    # C[1,0] = T2 + T4 - T3 - T7
    # C[1,1] = T1 + T3 - T2 - T6
    
    # Let me verify C[0,0] = T1 + T4 - T5 + T7:
    # T1 = (a+d)(e+h) = ae+ah+de+dh
    # T4 = (c+b)(e+g) = ce+cg+be+bg
    # T5 = (a+d)(f+h) = af+ah+df+dh
    # T7 = (b-d)(f+h) = bf+bh-df-dh
    # T1+T4-T5+T7 = ae+ah+de+dh+ce+cg+be+bg-af-ah-df-dh+bf+bh-df-dh
    # = ae + de + ce + cg + be + bg - af - df + bf + bh - 2dh + df
    # = ae + de + ce + cg + be + bg - af + bf + bh - 2dh
    # This doesn't simplify to ae+bg. So my Strassen formulas are wrong.
    
    # Let me use the correct ones from Strassen's paper:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = (a+b)(g+h)
    # M4 = (c+b)(e+g)
    # M5 = (a+d)(f+h)
    # M6 = (c-a)(e+h)
    # M7 = (b-d)(f+h)
    # 
    # C11 = M1 + M3 - M2 - M6
    # C12 = M4 + M5 - M3 - M7  (using 1-indexed, so C12 = C[0,1] in 0-indexed)
    # C21 = M2 + M4 - M1 - M5  (C21 = C[1,0])
    # C22 = M1 + M4 - M5 - M7  (C22 = C[1,1])
    
    # Wait, I'm getting confused with indexing. Let me just use a verified set.
    
    # Actually, I'll just return the schoolbook for now. The rank-17 construction
    # requires careful verification that I'm not confident in doing correctly here.
    
    return get_schoolbook()
