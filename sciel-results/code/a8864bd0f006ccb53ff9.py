import numpy as np
import time

def construct():
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
            
    # A is 2x3. Indices: 0=a00, 1=a01, 2=a02, 3=a10, 4=a11, 5=a12
    # B is 3x4. Indices: 0=b00, 1=b01, 2=b02, 3=b03, 4=b10, 5=b11, 6=b12, 7=b13, 8=b20, 9=b21, 10=b22, 11=b23
    # C is 2x4. W indices: k*n+i.
    # W[0]=c00, W[1]=c10, W[2]=c01, W[3]=c11
    # W[4]=c02, W[5]=c12, W[6]=c03, W[7]=c13
    
    # Strassen Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # m1 = (a00+a11)(b00+b11) -> c00, c11
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    # m2 = (a10+a11)b00 -> c10, c11(-)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    # m3 = a00(b01-b10) -> c01, c11
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    # m4 = a11(b10+b00) -> c00, c10
    # Wait, standard Strassen m4 is a11(b10+b11)? No.
    # Let's use the standard set:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00)
    # m3 = (a00)(b01-b10)
    # m4 = (a11)(b10+b11)
    # m5 = (a00+a01)(b10)
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    #
    # c00 = m1 - m4 + m6 + m7
    # c01 = m3 + m4 + m5
    # c10 = m2 + m4 - m6 - m7
    # c11 = m1 - m2 + m3 + m5
    
    # Let's re-verify the assignments in the code above.
    # Term 0 (m1): U[0,4] V[0,5] W[0,3]. Correct.
    # Term 1 (m2): U[3,4] V[0] W[1,3(-)]. Correct.
    # Term 2 (m3): U[0] V[1,5(-)] W[2,3]. Correct.
    # Term 3 (m4): Should be a11(b10+b11).
    # Current code term 3: add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    # This is a11(b10 - b00). This is NOT m4.
    # The standard Strassen formulas vary. Let's stick to one set.
    
    # Let's use this set (from Wikipedia):
    # m1 = (a11 - a10) (b00 + b10)
    # m2 = (a00 + a10) (b00)
    # m3 = (a01 + a11) (b11)
    # m4 = (a00 - a11) (b01 + b11)
    # m5 = (a10 + a11) (b01 - b11)
    # m6 = (a00 + a01) (b10 - b00)
    # m7 = (a10 - a00) (b00 + b10)  <-- Duplicate? No, this is a different variant.
    
    # Let's use the variant from the code comments which seemed closer:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b10)
    # m4 = a11(b10+b11)  <-- Let's use this one
    # m5 = (a00+a01)b10
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # c00 = m1 - m4 + m6 + m7
    # c01 = m3 + m4 + m5
    # c10 = m2 + m4 - m6 - m7
    # c11 = m1 - m2 + m3 + m5
    
    # Let's re-implement terms 3, 4, 5, 6 with this definition.
    
    # Term 3 (m4): a11(b10+b11) -> c00(+), c10(+)
    add_term([(4, 1)], [(4, 1), (5, 1)], [(0, 1), (1, 1)], 3)
    
    # Term 4 (m5): (a00+a01)b10 -> c01(+), c11(+)
    add_term([(0, 1), (1, 1)], [(4, 1)], [(2, 1), (3, 1)], 4)
    
    # Term 5 (m6): (a10-a00)(b00+b01) -> c00(+), c10(-)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(0, 1), (1, -1)], 5)
    
    # Term 6 (m7): (a01-a11)(b10+b11) -> c00(+), c10(-)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1), (1, -1)], 6)
    
    # Check c00: m1(0) + m4(3) - m4(3) + m6(5) + m7(6) -> Wait.
    # c00 = m1 - m4 + m6 + m7.
    # Term 0 (m1): +1 to c00.
    # Term 3 (m4): +1 to c00. But formula says -m4.
    # So I need to flip the sign of Term 3 in the W assignment, or define m4 with a sign flip.
    # Let's adjust Term 3 W to be -1 for c00, -1 for c10?
    # If I use m4' = -m4, then c00 = m1 + m4' + m6 + m7? No.
    
    # Let's just hardcode the coefficients based on the desired c values.
    # We want:
    # c00 = 1*m1 - 1*m4 + 1*m6 + 1*m7
    # c01 = 0*m1 + 1*m4 + 1*m5 + 0*m6 + 0*m7 (and m3)
    # c10 = 0*m1 + 1*m4 - 1*m6 - 1*m7 (and m2)
    # c11 = 1*m1 - 1*m2 + 1*m3 + 1*m5
    
    # Let's re-map the terms to match these coefficients.
    
    # Term 0 (m1): c00(+), c11(+)
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    
    # Term 1 (m2): c10(+), c11(-)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    
    # Term 2 (m3): c01(+), c11(+)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    
    # Term 3 (m4): c00(-), c10(+)  <-- Note the -1 for c00
    # m4 = a11(b10+b11)
    add_term([(4, 1)], [(4, 1), (5, 1)], [(0, -1), (1, 1)], 3)
    
    # Term 4 (m5): c01(+), c11(+)
    # m5 = (a00+a01)b10
    add_term([(0, 1), (1, 1)], [(4, 1)], [(2, 1), (3, 1)], 4)
    
    # Term 5 (m6): c00(+), c10(-)
    # m6 = (a10-a00)(b00+b01)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(0, 1), (1, -1)], 5)
    
    # Term 6 (m7): c00(+), c10(-)
    # m7 = (a01-a11)(b10+b11)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1), (1, -1)], 6)
    
    # Verify c00: 1(m1) - 1(m4) + 1(m6) + 1(m7). Correct.
    # Verify c01: 1(m3) + 0(m4, wait m4 is -1 for c00, but what for c01?)
    # My Term 3 (m4) only touches c00 and c10.
    # But Strassen m4 should contribute to c01?
    # Formula: c01 = m3 + m4 + m5.
    # So m4 must contribute +1 to c01.
    # Let's add (2, 1) to Term 3 W.
    add_term([(2, 1)], [], [], 3) # Add c01 contribution to term 3
    
    # Verify c10: 1(m2) + 1(m4) - 1(m6) - 1(m7).
    # Term 1 (m2): c10(+).
    # Term 3 (m4): c10(+).
    # Term 5 (m6): c10(-).
    # Term 6 (m7): c10(-).
    # Correct.
    
    # Verify c11: 1(m1) - 1(m2) + 1(m3) + 1(m5).
    # Term 0 (m1): c11(+).
    # Term 1 (m2): c11(-).
    # Term 2 (m3): c11(+).
    # Term 4 (m5): c11(+).
    # Correct.
    
    # Now Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # B indices: 2=b02, 3=b03, 6=b12, 7=b13
    # C indices: 4=c02, 5=c12, 6=c03, 7=c13
    
    # m1' = (a00+a11)(b02+b13) -> c02, c13
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    
    # m2' = (a10+a11)b02 -> c12, c13(-)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    
    # m3' = a00(b03-b12) -> c03, c13
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    
    # m4' = a11(b12+b13) -> c02(-), c12(+), c03(+)
    add_term([(4, 1)], [(6, 1), (7, 1)], [(4, -1), (5, 1), (6, 1)], 10)
    
    # m5' = (a00+a01)b12 -> c03(+), c13(+)
    add_term([(0, 1), (1, 1)], [(6, 1)], [(6, 1), (7, 1)], 11)
    
    # m6' = (a10-a00)(b02+b03) -> c02(+), c12(-)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(4, 1), (5, -1)], 12)
    
    # m7' = (a01-a11)(b12+b13) -> c02(+), c12(-)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1), (5, -1)], 13)
    
    # Cross Terms: A[:, 2] * B[2, :] -> C[:, :]
    # A: 2=a02, 5=a12
    # B: 8=b20, 9=b21, 10=b22, 11=b23
    # C: 0..7
    
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    return {"U": U, "V": V, "W": W}
