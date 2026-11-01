import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 2, 2
    rank = 7

    # We will build the Strassen decomposition programmatically
    # using a set of 7 bilinear forms.
    # Each form is defined by a linear combination of A's entries and B's entries.
    # U, V, W are the factor matrices.

    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Define the 7 Strassen multiplications
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W index convention: W[k*n + i, r] where k is column index of C, i is row index of C?
    # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
    # Let's re-read the seed program comments:
    # c00 -> k=0, i=0 -> idx 0
    # c01 -> k=1, i=0 -> idx 2
    # c10 -> k=0, i=1 -> idx 1
    # c11 -> k=1, i=1 -> idx 3
    # This implies W index = col * n + row.
    # So W[0] is C[0,0], W[1] is C[1,0], W[2] is C[0,1], W[3] is C[1,1].

    # Let's define the 7 terms (M1 to M7) as linear combinations of A and B entries.
    # M1: (a00 + a11) * (b00 + b11)
    # M2: (a10 + a11) * b00
    # M3: a00 * (b01 - b11)
    # M4: a11 * (b10 - b00)
    # M5: (a00 + a01) * b11
    # M6: (a10 - a00) * (b00 + b01)
    # M7: (a01 - a11) * (b10 + b11)

    # We'll store the coefficients for A and B for each of the 7 terms.
    # terms[j] = (coeffs_A, coeffs_B) where coeffs_A is a dict {idx: coeff} and same for B.
    
    # Term 1
    coeffs_A_1 = {0: 1, 3: 1}
    coeffs_B_1 = {0: 1, 3: 1}
    # Term 2
    coeffs_A_2 = {2: 1, 3: 1}
    coeffs_B_2 = {0: 1}
    # Term 3
    coeffs_A_3 = {0: 1}
    coeffs_B_3 = {1: 1, 3: -1}
    # Term 4
    coeffs_A_4 = {3: 1}
    coeffs_B_4 = {2: 1, 0: -1}
    # Term 5
    coeffs_A_5 = {0: 1, 1: 1}
    coeffs_B_5 = {3: 1}
    # Term 6
    coeffs_A_6 = {2: 1, 0: -1}
    coeffs_B_6 = {0: 1, 1: 1}
    # Term 7
    coeffs_A_7 = {1: 1, 3: -1}
    coeffs_B_7 = {2: 1, 3: 1}

    terms = [
        (coeffs_A_1, coeffs_B_1),
        (coeffs_A_2, coeffs_B_2),
        (coeffs_A_3, coeffs_B_3),
        (coeffs_A_4, coeffs_B_4),
        (coeffs_A_5, coeffs_B_5),
        (coeffs_A_6, coeffs_B_6),
        (coeffs_A_7, coeffs_B_7)
    ]

    # Fill U and V
    for j, (coeffs_A, coeffs_B) in enumerate(terms):
        for idx_A, val in coeffs_A.items():
            U[idx_A][j] = val
        for idx_B, val in coeffs_B.items():
            V[idx_B][j] = val

    # Now we need to determine W.
    # The decomposition is: C = sum_j (sum_i U[i,j] A[i]) * (sum_k V[k,j] B[k]) * W[?,j]
    # Actually, the tensor is T[i,k,j] = sum_j U[i,j] V[k,j] W[l,j] where l corresponds to C[i,k]?
    # No, the standard decomposition is:
    # C[i,k] = sum_j (sum_a U[a,j] A[a]) * (sum_b V[b,j] B[b]) * W[i*k*n + i, j] ?
    # Let's check the indices.
    # A has indices 0..3 (row, col). B has indices 0..3 (row, col).
    # C has indices 0..3 (row, col).
    # The matrix multiplication tensor T is T[i, j, k] = A[i, l] B[l, k] summed over l.
    # In the flattened index:
    # A index a = i_A * m + l? No, A is n x m. A[i, l]. Index = i*m + l.
    # B index b = l * p + k? No, B is m x p. B[l, k]. Index = l*p + k.
    # C index c = i * p + k? No, C is n x p. C[i, k]. Index = i*p + k.
    # But the task says W[k*n+i]. This is unusual.
    # Let's re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # And the seed program comment:
    # c00 -> k=0, i=0 -> idx 0
    # c01 -> k=1, i=0 -> idx 2
    # c10 -> k=0, i=1 -> idx 1
    # c11 -> k=1, i=1 -> idx 3
    # This implies W index = k * n + i.
    # If C[i, k] is the element at row i, col k.
    # Then W index = k * n + i.
    # So W[0] corresponds to C[0,0].
    # W[1] corresponds to C[1,0] (row 1, col 0).
    # W[2] corresponds to C[0,1] (row 0, col 1).
    # W[3] corresponds to C[1,1] (row 1, col 1).
    
    # Let's verify the Strassen formulas with this convention.
    # C[0,0] = M1 - M4 - M5 + M7
    # C[1,0] = M2 + M4
    # C[0,1] = M3 + M5
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Let's check if my term definitions match standard Strassen.
    # Standard Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Let's verify C00:
    # M1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11
    # M4 = a11(b10-b00) = a11b10 - a11b00
    # M5 = (a00+a01)b11 = a00b11 + a01b11
    # M7 = (a01-a11)(b10+b11) = a01b10 + a01b11 - a11b10 - a11b11
    # M1 - M4 - M5 + M7
    # = (a00b00 + a00b11 + a11b00 + a11b11) - (a11b10 - a11b00) - (a00b11 + a01b11) + (a01b10 + a01b11 - a11b10 - a11b11)
    # = a00b00 + a00b11 + a11b00 + a11b11 - a11b10 + a11b00 - a00b11 - a01b11 + a01b10 + a01b11 - a11b10 - a11b11
    # = a00b00 + 2a11b00 - 2a11b10 + a01b10
    # This is not C00 = a00b00 + a01b10.
    # There is an error in my Strassen coefficients or the standard formula I recalled.
    
    # Let's re-derive or look up the correct Strassen coefficients.
    # Wikipedia:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 - M5 + M7
    # Let's re-evaluate the sum carefully.
    # M1: a00b00, a00b11, a11b00, a11b11
    # -M4: -a11b10, +a11b00
    # -M5: -a00b11, -a01b11
    # +M7: +a01b10, +a01b11, -a11b10, -a11b11
    
    # Sum:
    # a00b00: 1
    # a00b11: 1 - 1 = 0
    # a11b00: 1 + 1 = 2 ? Wait.
    # M1 has a11b00. -M4 has +a11b00. So 2a11b00.
    # a11b11: 1 - 1 = 0
    # a11b10: -1 - 1 = -2
    # a01b11: -1 + 1 = 0
    # a01b10: 1
    
    # Result: a00b00 + 2a11b00 - 2a11b10 + a01b10.
    # This is definitely not C00.
    
    # I must have the wrong Strassen formulas.
    # Let's try a different set or re-derive.
    # Actually, the standard Strassen is:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Let's check C10 = M2 + M4
    # M2 = (a10+a11)b00 = a10b00 + a11b00
    # M4 = a11(b10-b00) = a11b10 - a11b00
    # Sum = a10b00 + a11b10.
    # C10 = a10b00 + a11b10. This is correct!
    
    # Let's check C01 = M3 + M5
    # M3 = a00(b01-b11) = a00b01 - a00b11
    # M5 = (a00+a01)b11 = a00b11 + a01b11
    # Sum = a00b01 + a01b11.
    # C01 = a00b01 + a01b11. This is correct!
    
    # Let's check C11 = M1 - M2 + M3 + M6
    # M1: a00b00, a00b11, a11b00, a11b11
    # -M2: -a10b00, -a11b00
    # M3: a00b01, -a00b11
    # M6: (a10-a00)(b00+b01) = a10b00 + a10b01 - a00b00 - a00b01
    # Sum:
    # a00b00: 1 - 1 = 0
    # a00b11: 1 - 1 = 0
    # a11b00: 1 - 1 = 0
    # a11b11: 1
    # a10b00: -1 + 1 = 0
    # a00b01: 1 - 1 = 0
    # a10b01: 1
    # Result: a11b11 + a10b01.
    # C11 = a10b01 + a11b11. This is correct!
    
    # So why did C00 fail?
    # C00 = M1 - M4 - M5 + M7
    # Let's re-evaluate M1 - M4 - M5 + M7.
    # M1: a00b00, a00b11, a11b00, a11b11
    # -M4: -a11b10, +a11b00
    # -M5: -a00b11, -a01b11
    # +M7: +a01b10, +a01b11, -a11b10, -a11b11
    
    # Terms:
    # a00b00: 1
    # a00b11: 1 - 1 = 0
    # a11b00: 1 + 1 = 2
    # a11b11: 1 - 1 = 0
    # a11b10: -1 - 1 = -2
    # a01b11: -1 + 1 = 0
    # a01b10: 1
    
    # It seems I am making a mistake in the standard formula or my expansion.
    # Let's check a known correct Strassen implementation.
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Wait, is M7 correct?
    # M7 = (A01 - A11)(B10 + B11)
    # A01 is row 0, col 1. A11 is row 1, col 1.
    # B10 is row 1, col 0. B11 is row 1, col 1.
    
    # Let's try a different source.
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Maybe my expansion of M1 is wrong?
    # M1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11. Correct.
    # M4 = a11(b10-b00) = a11b10 - a11b00. Correct.
    # M5 = (a00+a01)b11 = a00b11 + a01b11. Correct.
    # M7 = (a01-a11)(b10+b11) = a01b10 + a01b11 - a11b10 - a11b11. Correct.
    
    # Sum:
    # a00b00: 1
    # a00b11: 1 - 1 = 0
    # a11b00: 1 - (-1) = 2? No, -M4 is -a11b10 + a11b00. So +a11b00.
    # So a11b00 coefficient is 1 (from M1) + 1 (from -M4) = 2.
    # a11b11: 1 - 1 = 0.
    # a11b10: -1 (from -M4) - 1 (from M7) = -2.
    # a01b11: -1 (from -M5) + 1 (from M7) = 0.
    # a01b10: 1 (from M7).
    
    # So C00 = a00b00 + 2a11b00 - 2a11b10 + a01b10.
    # This is not C00 = a00b00 + a01b10.
    
    # There must be a different set of 7 multiplications.
    # Let's look at the seed program again.
    # M1: (a00 + a11) * (b00 + b11)
    # M2: (a10 + a11) * b00
    # M3: a00 * (b01 - b11)
    # M4: a11 * (b10 - b00)
    # M5: (a00 + a01) * b11
    # M6: (a10 - a00) * (b00 + b01)
    # M7: (a01 - a11) * (b10 + b11)
    
    # And the W coefficients in the seed program:
    # W[0][0] = 1; W[3][0] = 1  (M1 contributes to C00 and C11)
    # W[1][1] = 1; W[3][1] = -1 (M2 contributes to C10 and -C11)
    # W[2][2] = 1; W[3][2] = 1  (M3 contributes to C01 and C11)
    # W[0][3] = 1; W[1][3] = 1  (M4 contributes to C00 and C10)
    # W[0][4] = -1; W[2][4] = 1 (M5 contributes to -C00 and C01)
    # W[3][5] = 1               (M6 contributes to C11)
    # W[0][6] = 1               (M7 contributes to C00)
    
    # Let's check C00 from seed:
    # M1 - M5 + M4 + M7
    # = (a00b00 + a00b11 + a11b00 + a11b11) - (a00b11 + a01b11) + (a11b10 - a11b00) + (a01b10 + a01b11 - a11b10 - a11b11)
    # = a00b00 + a00b11 + a11b00 + a11b11 - a00b11 - a01b11 + a11b10 - a11b00 + a01b10 + a01b11 - a11b10 - a11b11
    # = a00b00 + a01b10.
    # This is correct!
    
    # So the seed program uses:
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Let's verify C10:
    # M2 + M4 = (a10b00 + a11b00) + (a11b10 - a11b00) = a10b00 + a11b10. Correct.
    
    # Let's verify C01:
    # M3 + M5 = (a00b01 - a00b11) + (a00b11 + a01b11) = a00b01 + a01b11. Correct.
    
    # Let's verify C11:
    # M1 - M2 + M3 + M6
    # = (a00b00 + a00b11 + a11b00 + a11b11) - (a10b00 + a11b00) + (a00b01 - a00b11) + (a10b00 + a10b01 - a00b00 - a00b01)
    # = a00b00 + a00b11 + a11b00 + a11b11 - a10b00 - a11b00 + a00b01 - a00b11 + a10b00 + a10b01 - a00b00 - a00b01
    # = a11b11 + a10b01. Correct.
    
    # So the W coefficients are:
    # C00 (idx 0): M1(1), M4(1), M5(-1), M7(1)
    # C10 (idx 1): M2(1), M4(1)
    # C01 (idx 2): M3(1), M5(1)
    # C11 (idx 3): M1(1), M2(-1), M3(1), M6(1)
    
    # Now I can fill W.
    W[0][0] = 1  # M1
    W[0][3] = 1  # M4
    W[0][4] = -1 # M5
    W[0][6] = 1  # M7
    
    W[1][1] = 1  # M2
    W[1][3] = 1  # M4
    
    W[2][2] = 1  # M3
    W[2][4] = 1  # M5
    
    W[3][0] = 1  # M1
    W[3][1] = -1 # M2
    W[3][2] = 1  # M3
    W[3][5] = 1  # M6

    return {"U": U, "V": V, "W": W}
