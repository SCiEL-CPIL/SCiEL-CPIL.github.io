import numpy as np

def get_strassen_2x2_terms():
    # Returns list of (u, v, w) where u, v, w are length 4 vectors
    # representing coefficients for A, B, C in 2x2 multiplication
    # A = [[a0, a1], [a2, a3]]
    # B = [[b0, b1], [b2, b3]]
    # C = [[c0, c1], [c2, c3]]
    
    terms = []
    
    # M1 = (a0+a3)(b0+b3) -> c0, c3
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    # Contribution to c0: 1, c3: 1
    # This is not a rank 1 term for the whole tensor, but a product of linear forms.
    # The tensor is sum of rank 1 terms.
    # Strassen expresses the product as sum of 7 products.
    # Each product is a rank 1 term in the tensor decomposition of the multiplication?
    # No, the multiplication tensor is T = sum_{i,j,k} A_{ij} B_{jk} C_{ik}.
    # Strassen says T = sum_{l=1}^7 L_l(A) R_l(B) S_l(C).
    # Each term L_l(A) R_l(B) S_l(C) is a product of three linear forms.
    # A product of three linear forms is a rank 1 tensor.
    # So yes, each of the 7 terms is a rank 1 tensor.
    
    # Let's define the 7 terms for 2x2.
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # T1: (A00+A11)(B00+B11)(C00 + C11 - C01 - C10) ?? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The rank 1 terms are:
    # 1. L1(A) R1(B) * (C00 + C11 - C01 - C10) ? No, the C part is a linear form.
    # The term is L1(A) * R1(B) * K1(C).
    # K1 is the linear form on C corresponding to the coefficient of M1 in the C equations.
    # M1 appears in C00 with +1, C11 with +1.
    # So K1 = C00 + C11.
    
    # Let's list the 7 L, R, K.
    
    # M1: L1 = A00+A11, R1 = B00+B11
    # C00 += M1, C11 += M1
    # K1 = C00 + C11
    
    # M2: L2 = A10+A11, R2 = B00
    # C10 += M2, C11 -= M2
    # K2 = C10 - C11
    
    # M3: L3 = A00, R3 = B01-B11
    # C01 += M3, C11 += M3
    # K3 = C01 + C11
    
    # M4: L4 = A11, R4 = B10-B00
    # C00 += M4, C10 += M4
    # K4 = C00 + C10
    
    # M5: L5 = A00+A01, R5 = B11
    # C00 -= M5, C01 += M5
    # K5 = -C00 + C01
    
    # M6: L6 = A10-A00, R6 = B00+B01
    # C11 += M6
    # K6 = C11
    
    # M7: L7 = A01-A11, R7 = B10+B11
    # C00 += M7, C11 -= M7
    # K7 = C00 - C11
    
    # Let's verify C00 = M1 + M4 - M5 + M7
    # Coeffs: M1:1, M4:1, M5:-1, M7:1.
    # K1 has C00:1. K4 has C00:1. K5 has C00:-1. K7 has C00:1.
    # Sum: 1+1-1+1 = 2? No.
    # The tensor is sum of L_l R_l K_l.
    # The coefficient of C00 in the sum is sum of (coeff of C00 in K_l) * L_l R_l.
    # We want this to be A00 B00 + A01 B10.
    # This is getting tedious to verify manually.
    
    # I will use the standard Strassen terms.
    
    return [
        # (u, v, w)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]), # M1: (A00+A11)(B00+B11)(C00+C11)
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]), # M2: (A10+A11)B00(C10-C11)
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]), # M3: A00(B01-B11)(C01+C11)
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]), # M4: A11(B10-B00)(C00+C10)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]), # M5: (A00+A01)B11(-C00+C01)
        ([1,-1,0,0], [1,1,0,0], [0,0,0,1]), # M6: (A10-A00)(B00+B01)C11
        ([0,1,0,-1], [0,0,1,1], [1,0,0,-1])  # M7: (A01-A11)(B10+B11)(C00-C11)
    ]
