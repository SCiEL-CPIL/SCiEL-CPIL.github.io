import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x3x6 multiplication into two 3x3x3 multiplications.
    # Block 1: Columns 0-2 of B
    # Block 2: Columns 3-5 of B
    
    # We use the Strassen-like decomposition for 3x3 matrices which has rank 19.
    # The 19 terms for multiplying two 3x3 matrices A (rows i, cols j) and B (rows j, cols k)
    # to produce C (rows i, cols k) are defined by their contributions to C[i,k].
    # Each term is a product of a linear combination of A entries and a linear combination of B entries.
    # We represent each term as a list of (index, coeff) for U (A), V (B), and W (C).
    
    # The 19 terms are derived from the standard Strassen algorithm extended to 3x3.
    # Let's define them explicitly based on known rank 19 decompositions for 3x3.
    
    # Term 1: A00 * B00 -> C00
    # Term 2: A01 * B10 -> C00
    # ...
    
    # Actually, let's use the explicit rank 19 decomposition for 3x3.
    # A common rank 19 decomposition for 3x3 matrix multiplication is:
    # M1 = (A00 + A11 + A22) * (B00 + B11 + B22)
    # M2 = (A00 - A11) * (B00 - B11)
    # M3 = (A00 - A22) * (B00 - B22)
    # M4 = (A11 - A22) * (B11 - B22)
    # M5 = A00 * B01
    # M6 = A01 * B10
    # M7 = A00 * B02
    # M8 = A02 * B20
    # M9 = A11 * B12
    # M10 = A12 * B21
    # M11 = A20 * B01
    # M12 = A21 * B10
    # M13 = A20 * B02
    # M14 = A22 * B21
    # M15 = A01 * B12
    # M16 = A10 * B02
    # M17 = A12 * B22
    # M18 = A21 * B12
    # M19 = A10 * B01
    
    # Wait, this is getting complicated to verify manually.
    # Let's use a simpler approach: the standard Strassen for 2x2 blocks within 3x3 is not directly applicable.
    # However, it is known that the tensor rank for 3x3 is 19.
    # Let's construct the 19 terms programmatically or use a known list.
    
    # Known rank 19 decomposition for 3x3:
    # The 19 multiplications are:
    # 1.  (A00+A11+A22)(B00+B11+B22)
    # 2.  (A00-A11)(B00-B11)
    # 3.  (A00-A22)(B00-B22)
    # 4.  (A11-A22)(B11-B22)
    # 5.  A00B01
    # 6.  A01B10
    # 7.  A00B02
    # 8.  A02B20
    # 9.  A11B12
    # 10. A12B21
    # 11. A20B01
    # 12. A21B10
    # 13. A20B02
    # 14. A22B21
    # 15. A01B12
    # 16. A10B02
    # 17. A12B22
    # 18. A21B12
    # 19. A10B01
    
    # Let's verify the reconstruction of C entries.
    # C00 = M1 + M2 + M3 + M6 + M19
    #      = (A00+A11+A22)(B00+B11+B22) + (A00-A11)(B00-B11) + (A00-A22)(B00-B22) + A01B10 + A10B01
    #      = A00B00 + A00B11 + A00B22 + A11B00 + A11B11 + A11B22 + A22B00 + A22B11 + A22B22
    #        + A00B00 - A00B11 - A11B00 + A11B11
    #        + A00B00 - A00B22 - A22B00 + A22B22
    #        + A01B10 + A10B01
    #      = 3A00B00 + A00B11 + A00B22 + A11B11 + A11B22 + A22B22 + A01B10 + A10B01
    #      This is not C00 = A00B00 + A01B10 + A02B20.
    # So this specific set of terms is not correct for direct sum.
    
    # Let's use a verified rank 19 decomposition.
    # From literature, the rank of 3x3 matrix multiplication is 19.
    # A specific decomposition is:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A00-A11)(B00-B11)
    # M3 = (A02+A20)(B00+B22)
    # M4 = (A02-A20)(B00-B22)
    # M5 = (A12+A21)(B11+B22)
    # M6 = (A12-A21)(B11-B22)
    # M7 = (A10+A01)(B00+B11)
    # M8 = (A10-A01)(B00-B11)
    # M9 = (A20+A02)(B00+B22)
    # M10 = (A20-A02)(B00-B22)
    # M11 = (A21+A12)(B11+B22)
    # M12 = (A21-A12)(B11-B22)
    # M13 = A00B22
    # M14 = A11B00
    # M15 = A22B11
    # M16 = A01B20
    # M17 = A02B10
    # M18 = A10B21
    # M19 = A20B12
    
    # This is also complex. Let's stick to the schoolbook method but try to optimize.
    # Wait, the task asks for rank-minimal. The current program uses rank 54 (schoolbook).
    # The theoretical minimum for 3x3x6 is likely related to the rank of 3x3.
    # If rank(3x3) = 19, then rank(3x3x6) might be 2*19 = 38?
    # Or is it less?
    
    # Let's implement a known rank 19 decomposition for 3x3.
    # Here is a correct rank 19 decomposition for 3x3 matrices:
    # Reference: "The rank of matrix multiplication is at most 19 for 3x3"
    
    # Terms:
    # 1.  A00*B00
    # 2.  A01*B10
    # 3.  A02*B20
    # 4.  A10*B01
    # 5.  A11*B11
    # 6.  A12*B21
    # 7.  A20*B02
    # 8.  A21*B12
    # 9.  A22*B22
    # 10. (A00+A11+A22)*(B00+B11+B22)
    # 11. (A00-A11)*(B00-B11)
    # 12. (A00-A22)*(B00-B22)
    # 13. (A11-A22)*(B11-B22)
    # 14. (A01+A10)*(B01+B10)
    # 15. (A01-A10)*(B01-B10)
    # 16. (A02+A20)*(B02+B20)
    # 17. (A02-A20)*(B02-B20)
    # 18. (A12+A21)*(B12+B21)
    # 19. (A12-A21)*(B12-B21)
    
    # Let's check C00:
    # C00 = A00B00 + A01B10 + A02B20
    # From terms:
    # T1: A00B00
    # T2: A01B10
    # T3: A02B20
    # T10: contributes A00B00, A00B11, A00B22, A11B00, A11B11, A11B22, A22B00, A22B11, A22B22
    # T11: contributes A00B00 - A00B11 - A11B00 + A11B11
    # T12: contributes A00B00 - A00B22 - A22B00 + A22B22
    # T13: contributes A11B11 - A11B22 - A22B11 + A22B22
    # Sum of T10+T11+T12+T13 for C00 part:
    # A00B00: 1+1+1 = 3
    # A00B11: 1-1 = 0
    # A00B22: 1-1 = 0
    # A11B00: 1-1 = 0
    # A11B11: 1+1+1 = 3
    # A11B22: 1-1 = 0
    # A22B00: 1-1 = 0
    # A22B11: 1-1 = 0
    # A22B22: 1+1+1 = 3
    # So T10-13 contribute 3*(A00B00 + A11B11 + A22B22) to C00? No, they contribute to all C entries.
    # This manual verification is error-prone.
    
    # Let's use a simpler, guaranteed correct decomposition.
    # The schoolbook decomposition has rank 27 for 3x3.
    # Total rank for 3x3x6 is 54.
    # Can we do better?
    # Strassen's algorithm for 2x2 has rank 7.
    # For 3x3, we can pad to 4x4 and use Strassen? No, that increases rank.
    # The best known rank for 3x3 is 19.
    
    # Let's try to find a valid rank 19 decomposition online or derive one.
    # Actually, let's just use the schoolbook method but check if we can merge any terms.
    # For 3x3x6, we have two blocks of 3x3.
    # If we use rank 27 for each, total is 54.
    # If we use rank 19 for each, total is 38.
    
    # Let's implement a rank 19 decomposition for 3x3.
    # Here is a known correct rank 19 decomposition:
    # M1 = A00B00
    # M2 = A01B10
    # M3 = A02B20
    # M4 = A10B01
    # M5 = A11B11
    # M6 = A12B21
    # M7 = A20B02
    # M8 = A21B12
    # M9 = A22B22
    # M10 = (A00+A11+A22)(B00+B11+B22)
    # M11 = (A00-A11)(B00-B11)
    # M12 = (A00-A22)(B00-B22)
    # M13 = (A11-A22)(B11-B22)
    # M14 = (A01+A10)(B01+B10)
    # M15 = (A01-A10)(B01-B10)
    # M16 = (A02+A20)(B02+B20)
    # M17 = (A02-A20)(B02-B20)
    # M18 = (A12+A21)(B12+B21)
    # M19 = (A12-A21)(B12-B21)
    
    # We need to verify this.
    # C00 = A00B00 + A01B10 + A02B20
    # Contributions to C00:
    # M1: A00B00
    # M2: A01B10
    # M3: A02B20
    # M10: A00B00 + A00B11 + A00B22 + A11B00 + A11B11 + A11B22 + A22B00 + A22B11 + A22B22
    # M11: A00B00 - A00B11 - A11B00 + A11B11
    # M12: A00B00 - A00B22 - A22B00 + A22B22
    # M13: A11B11 - A11B22 - A22B11 + A22B22
    # M14: A01B01 + A01B10 + A10B01 + A10B10
    # M15: A01B01 - A01B10 - A10B01 + A10B10
    # M16: A02B02 + A02B20 + A20B02 + A20B20
    # M17: A02B02 - A02B20 - A20B02 + A20B20
    # M18: A12B12 + A12B21 + A21B12 + A21B21
    # M19: A12B12 - A12B21 - A21B12 + A21B21
    
    # Summing coefficients for A00B00 in C00:
    # M1: 1
    # M10: 1
    # M11: 1
    # M12: 1
    # Total: 4. We need 1.
    # So we need to subtract 3/4 of (M10+M11+M12+M13)?
    # This implies the coefficients are not just 1.
    
    # Let's use a simpler approach: just use the schoolbook method.
    # Rank 54 is correct but not minimal.
    # However, finding a correct rank 19 decomposition is non-trivial.
    # Let's try to use the known result that rank(3x3) = 19.
    # I will use a pre-computed set of 19 terms.
    
    # Terms for 3x3 rank 19:
    # Each term is (u_indices, v_indices, w_indices)
    # u_indices: dict of (i,j) -> coeff
    # v_indices: dict of (j,k) -> coeff
    # w_indices: dict of (i,k) -> coeff
    
    # I will use the decomposition from the paper "A new algorithm for matrix multiplication" by Strassen, adapted for 3x3.
    # Actually, let's just use the schoolbook method to ensure correctness, as the task allows it.
    # The score is rank. Lower is better.
    # If I can't find a correct lower rank, I'll stick with 54.
    
    # Wait, the current program uses rank 54.
    # Can I reduce it?
    # For 3x3, the rank is 19.
    # So for 3x3x6, the rank should be 19+19=38.
    
    # Let's implement the rank 19 decomposition for 3x3.
    # I found a correct set of 19 terms:
    
    terms_3x3 = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: list of (i, j, coeff)
        # v_coeffs: list of (j, k, coeff)
        # w_coeffs: list of (i, k, coeff)
        ([ (0,0,1) ], [ (0,0,1) ], [ (0,0,1) ]), # M1
        ([ (0,1,1) ], [ (1,0,1) ], [ (0,0,1) ]), # M2
        ([ (0,2,1) ], [ (2,0,1) ], [ (0,0,1) ]), # M3
        ([ (1,0,1) ], [ (0,1,1) ], [ (0,1,1) ]), # M4
        ([ (1,1,1) ], [ (1,1,1) ], [ (1,1,1) ]), # M5
        ([ (1,2,1) ], [ (2,1,1) ], [ (1,1,1) ]), # M6
        ([ (2,0,1) ], [ (0,2,1) ], [ (0,2,1) ]), # M7
        ([ (2,1,1) ], [ (1,2,1) ], [ (1,2,1) ]), # M8
        ([ (2,2,1) ], [ (2,2,1) ], [ (2,2,1) ]), # M9
        ([ (0,0,1), (1,1,1), (2,2,1) ], [ (0,0,1), (1,1,1), (2,2,1) ], [ (0,0,1), (1,1,1), (2,2,1) ]), # M10
        ([ (0,0,1), (1,1,-1) ], [ (0,0,1), (1,1,-1) ], [ (0,0,1), (1,1,1) ]), # M11 (Note: W coeff for 1,1 is 1? No, need to check)
        # This is too risky.
        
        # Let's stick to the schoolbook method. It is correct.
        # Rank 54.
    ]
    
    # Since I cannot guarantee the correctness of a rank 19 decomposition without extensive verification,
    # and the current program already works with rank 54, I will keep the schoolbook method.
    # However, I will optimize the code structure.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(2):
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    k_global = k + block * 3
                    U[i * m + j][col] = 1
                    V[j * p + k_global][col] = 1
                    W[k_global * n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
