import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Uses Strassen's 7-multiplication method for each of the four 2x2 block products.
    """
    t0 = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen patterns for 2x2 matrix multiplication
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Indices in 2x2 block: 0=a, 1=b, 2=c, 3=d
    # e=0, f=1, g=2, h=3
    
    strassen_patterns = [
        # (a_indices, a_coeffs, b_indices, b_coeffs, c_indices, c_coeffs)
        # M1
        ((0, 3), (1, 1), (0, 3), (1, 1), (0, 3), (1, 1)),
        # M2
        ((2, 3), (1, 1), (0,), (1,), (2,), (1,)),
        # M3
        ((0,), (1,), (1, 3), (1, -1), (1,), (1,)),
        # M4
        ((3,), (1,), (2, 0), (1, -1), (0, 2), (1, 1)),
        # M5
        ((0, 1), (1, 1), (3,), (1,), (0, 1), (-1, 1)),
        # M6
        ((2, 0), (1, -1), (0, 1), (1, 1), (3,), (1,)),
        # M7
        ((1, 3), (1, -1), (2, 3), (1, 1), (0, 3), (1, 1)),
    ]
    
    # Note: The C coefficients in the patterns above correspond to:
    # M1 -> C11, C22
    # M2 -> C21
    # M3 -> C12
    # M4 -> C11, C21
    # M5 -> C11, C12
    # M6 -> C22
    # M7 -> C11, C22
    
    # Wait, my previous derivation:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # Let's map these to the patterns.
    # Pattern 1 (M1): C coeffs (1, 1) for indices (0, 3). Index 0 is C11, 3 is C22.
    #   So M1 adds 1 to C11 and 1 to C22. Correct.
    # Pattern 2 (M2): C coeffs (1,) for index (2,). Index 2 is C21.
    #   So M2 adds 1 to C21. Correct.
    # Pattern 3 (M3): C coeffs (1,) for index (1,). Index 1 is C12.
    #   So M3 adds 1 to C12. Correct.
    # Pattern 4 (M4): C coeffs (1, 1) for indices (0, 2). Index 0 is C11, 2 is C21.
    #   So M4 adds 1 to C11 and 1 to C21. Correct.
    # Pattern 5 (M5): C coeffs (-1, 1) for indices (0, 1). Index 0 is C11, 1 is C12.
    #   So M5 adds -1 to C11 and 1 to C12. Correct.
    # Pattern 6 (M6): C coeffs (1,) for index (3,). Index 3 is C22.
    #   So M6 adds 1 to C22. 
    #   Wait, my formula for C22 was M1 - M2 + M3 + M7. It did NOT include M6.
    #   Let's re-verify C22 with M6.
    #   If C22 = M1 - M2 + M3 + M7, then M6 coefficient in C22 should be 0.
    #   But standard Strassen often uses M6 for C22?
    #   Let's check Wikipedia again.
    #   C22 = M1 - M2 + M3 - M7. (My previous failed formula)
    #   Let's check a different source.
    #   Some sources: C22 = M1 - M2 + M3 - M7.
    #   My test case A=I, B=I gave C22=1 with +M7.
    #   Let's re-run the test for C22 = M1 - M2 + M3 - M7 with A=I, B=I.
    #   M1=4, M2=1, M3=-1, M7=-1.
    #   4 - 1 + (-1) - (-1) = 3. Incorrect.
    #   So -M7 is wrong for this variant.
    #   
    #   What about M6?
    #   M6 = (c-a)(e+f). For I, I: (0-1)(1+0) = -1.
    #   If C22 included M6, e.g., C22 = ... + M6?
    #   If C22 = M1 - M2 + M3 + M7 + M6?
    #   4 - 1 - 1 - 1 - 1 = 0. Incorrect.
    #   
    #   It seems my derived formula C22 = M1 - M2 + M3 + M7 is correct for the given M definitions.
    #   And M6 is not used in C?
    #   That can't be right. M6 must be used.
    #   
    #   Let's check C21 = M2 + M4.
    #   Let's check C12 = M3 + M5.
    #   Let's check C11 = M1 + M4 - M5 + M7.
    #   
    #   Where is M6?
    #   Maybe my M definitions are slightly different from the standard "Wikipedia" set?
    #   Wikipedia:
    #   M1 = (a+d)(e+h)
    #   M2 = (c+d)(e)
    #   M3 = a(f-h)
    #   M4 = d(g-e)
    #   M5 = (a+b)(h)
    #   M6 = (c-a)(e+f)
    #   M7 = (b-d)(g+h)
    #   
    #   Reconstruction:
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 - M7
    #   
    #   I proved C22 = M1 - M2 + M3 - M7 is wrong for I, I.
    #   I proved C22 = M1 - M2 + M3 + M7 is correct for I, I.
    #   
    #   Is it possible that M7 definition in Wikipedia is (b-d)(g+h) but the sign in reconstruction is different?
    #   Or maybe M6 is used in C22?
    #   Let's check if C22 = M1 - M2 + M3 + M6?
    #   I, I: 4 - 1 - 1 + (-1) = 1. Correct.
    #   Let's check C22 = M1 - M2 + M3 + M6 with another case.
    #   A = [[1, 1], [1, 1]], B = [[1, 1], [1, 1]].
    #   M1=4, M2=2, M3=0, M6=0.
    #   C22 = 4 - 2 + 0 + 0 = 2. Correct.
    #   
    #   So C22 = M1 - M2 + M3 + M6?
    #   Let's check if M7 is used anywhere.
    #   C11 uses M7.
    #   If M6 is used in C22, and M7 in C11, then all M are used.
    #   
    #   Let's verify C22 = M1 - M2 + M3 + M6.
    #   Expansion:
    #   M1 = ae+ah+de+dh
    #   -M2 = -ce-de
    #   M3 = af-ah
    #   M6 = ce+cf-ae-af
    #   Sum:
    #   ae+ah+de+dh -ce-de +af-ah +ce+cf-ae-af
    #   = (ae-ae) + (ah-ah) + (de-de) + dh + (-ce+ce) + (af-af) + cf
    #   = dh + cf.
    #   Correct!
    #   
    #   So the correct reconstruction for this set of M is:
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #   
    #   This means my previous pattern for M6 was:
    #   ((2, 0), (1, -1), (0, 1), (1, 1), (3,), (1,))
    #   This adds 1 to C22 (index 3). This matches.
    #   
    #   And my previous pattern for M7 was:
    #   ((1, 3), (1, -1), (2, 3), (1, 1), (0, 3), (1, 1))
    #   This adds 1 to C11 (index 0) and 1 to C22 (index 3).
    #   But C22 should NOT have M7.
    #   C11 should have M7.
    #   So M7 pattern should be:
    #   ((1, 3), (1, -1), (2, 3), (1, 1), (0,), (1,))
    #   
    #   Let's update the patterns.

    strassen_patterns = [
        # M1: C11(+1), C22(+1)
        ((0, 3), (1, 1), (0, 3), (1, 1), (0, 3), (1, 1)),
        # M2: C21(+1)
        ((2, 3), (1, 1), (0,), (1,), (2,), (1,)),
        # M3: C12(+1)
        ((0,), (1,), (1, 3), (1, -1), (1,), (1,)),
        # M4: C11(+1), C21(+1)
        ((3,), (1,), (2, 0), (1, -1), (0, 2), (1, 1)),
        # M5: C11(-1), C12(+1)
        ((0, 1), (1, 1), (3,), (1,), (0, 1), (-1, 1)),
        # M6: C22(+1)
        ((2, 0), (1, -1), (0, 1), (1, 1), (3,), (1,)),
        # M7: C11(+1)
        ((1, 3), (1, -1), (2, 3), (1, 1), (0,), (1,)),
    ]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),   # A_L * B_TL -> C_L
        (0, 2, 2, 0, 0, 0),   # A_R * B_BL -> C_L
        (0, 0, 0, 2, 0, 2),   # A_L * B_TR -> C_R
        (0, 2, 2, 2, 0, 2),   # A_R * B_BR -> C_R
    ]
    
    col = 0
    for (a_ro, a_co, b_ro, b_co, c_ro, c_co) in blocks:
        for (a_idx, a_c, b_idx, b_c, c_idx, c_c) in strassen_patterns:
            for idx, coeff in zip(a_idx, a_c):
                lr = idx // 2
                lc = idx % 2
                gi = a_ro + lr
                gj = a_co + lc
                U[gi * m + gj, col] = coeff
            
            for idx, coeff in zip(b_idx, b_c):
                lr = idx // 2
                lc = idx % 2
                gj = b_ro + lr
                gk = b_co + lc
                V[gj * p + gk, col] = coeff
            
            for idx, coeff in zip(c_idx, c_c):
                lr = idx // 2
                lc = idx % 2
                gi = c_ro + lr
                gk = c_co + lc
                W[gk * n + gi, col] = coeff
            
            col += 1
            if time.time() - t0 > deadline - 5:
                break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
