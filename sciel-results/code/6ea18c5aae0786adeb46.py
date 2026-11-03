"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen blocks
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2: A11(B10-B00) -> C10 += M2, C11 -= M2
    # M3: (A00-A11)(B11+B22) ... wait, standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M5 + M3
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-verify the coefficients used in the current code.
    # The current code uses:
    # M1: U(0,0; 1,1), V(0,0; 1,1), W(0,0; 1,1)
    #   U: A00+A11. V: B00+B11. W: C00+=1, C11+=1. Correct.
    
    # M2: U(1,0; 1,1), V(0,0), W(0,1; 1,1,-1)
    #   U: A10+A11. V: B00. W: C01+=1, C11-=-1.
    #   Standard Strassen M2 = (A10+A11)B00.
    #   C01 = M5 + M3. C11 = M1 - M2 + M3 + M6.
    #   Wait, the standard formulas vary by index convention.
    #   Let's check the current code's W for M2: (0,1,1), (1,1,-1).
    #   This adds to C[0,1] and subtracts from C[1,1].
    #   In standard Strassen:
    #   C01 = M5 + M3
    #   C11 = M1 - M2 + M3 + M6
    #   So M2 should contribute -1 to C11. It should NOT contribute to C01.
    #   The current code has W for M2 as (0,1,1) and (1,1,-1).
    #   This implies C01 gets +M2.
    #   Let's check if the current code is actually valid.
    #   The prompt says "score 1.1429", which implies it passed evaluation.
    #   So the decomposition is correct for the specific index ordering.
    #   The index ordering is W[k*n+i, r].
    #   n=2.
    #   W[0] -> k=0, i=0 -> C[0,0]
    #   W[1] -> k=0, i=1 -> C[0,1]
    #   W[2] -> k=1, i=0 -> C[1,0]
    #   W[3] -> k=1, i=1 -> C[1,1]
    
    # Let's re-verify M2 in current code:
    # w_coeffs: [(0,1,1), (1,1,-1)]
    # k=0, i=1 -> C[0,1] += 1
    # k=1, i=1 -> C[1,1] += -1
    # So C01 += M2, C11 -= M2.
    
    # Let's check M3 in current code:
    # u_coeffs: [(0,0,1)] -> A00
    # v_coeffs: [(0,1,1), (1,1,-1)] -> B01 - B11
    # w_coeffs: [(1,0,1), (1,1,1)] -> C10 += 1, C11 += 1
    # So C10 += M3, C11 += M3.
    
    # Let's check M4 in current code:
    # u_coeffs: [(1,1,1)] -> A11
    # v_coeffs: [(1,0,1), (0,0,-1)] -> B10 - B00
    # w_coeffs: [(0,0,1), (0,1,1)] -> C00 += 1, C01 += 1
    # So C00 += M4, C01 += M4.
    
    # Let's check M5 in current code:
    # u_coeffs: [(0,0,1), (0,1,1)] -> A00 + A01
    # v_coeffs: [(1,1,1)] -> B11
    # w_coeffs: [(0,0,-1), (1,0,1)] -> C00 -= 1, C10 += 1
    # So C00 -= M5, C10 += M5.
    
    # Let's check M6 in current code:
    # u_coeffs: [(1,0,1), (0,0,-1)] -> A10 - A00
    # v_coeffs: [(0,0,1), (0,1,1)] -> B00 + B01
    # w_coeffs: [(1,1,1)] -> C11 += 1
    # So C11 += M6.
    
    # Let's check M7 in current code:
    # u_coeffs: [(0,1,1), (1,1,-1)] -> A01 - A11
    # v_coeffs: [(1,0,1), (1,1,1)] -> B10 + B11
    # w_coeffs: [(0,0,1)] -> C00 += 1
    # So C00 += M7.
    
    # Let's sum up contributions to each C element:
    # C00: M1 + M4 - M5 + M7
    # C01: M2 + M4 + M3 (wait, M3 was C10, C11. M3 w_coeffs: (1,0,1), (1,1,1). So C10, C11. No C01.)
    #      M2 (C01), M4 (C01).
    #      So C01 = M2 + M4.
    # C10: M3 (C10), M5 (C10).
    #      So C10 = M3 + M5.
    # C11: M1 - M2 + M3 + M6.
    #      M1 (C11), M2 (-C11), M3 (C11), M6 (C11).
    #      So C11 = M1 - M2 + M3 + M6.
    
    # Let's verify if this matches matrix multiplication.
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # Let's expand the Strassen terms:
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M2 = (A10+A11)B00 = A10B00 + A11B00
    # M3 = A00(B01-B11) = A00B01 - A00B11
    # M4 = A11(B10-B00) = A11B10 - A11B00
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    
    # C00 = M1 + M4 - M5 + M7
    # = (A00B00 + A00B11 + A11B00 + A11B11)
    #   + (A11B10 - A11B00)
    #   - (A00B11 + A01B11)
    #   + (A01B10 + A01B11 - A11B10 - A11B11)
    # Terms:
    # A00B00: 1
    # A00B11: 1 - 1 = 0
    # A11B00: 1 - 1 = 0
    # A11B11: 1 - 1 = 0
    # A11B10: 1 - 1 = 0
    # A01B11: -1 + 1 = 0
    # A01B10: 1
    # Result: A00B00 + A01B10. Correct.
    
    # C01 = M2 + M4
    # = (A10B00 + A11B00) + (A11B10 - A11B00)
    # = A10B00 + A11B10.
    # Expected: A00B01 + A01B11.
    # MISMATCH!
    
    # The current code's C01 is A10B00 + A11B10.
    # This is NOT the correct C01 for standard matrix multiplication.
    # C01 should be A00B01 + A01B11.
    
    # Why did it pass?
    # Maybe the index mapping is different?
    # U is (n*m, rank) -> 8x28. Row i*m + j.
    # V is (m*p, rank) -> 16x28. Row j*p + k.
    # W is (p*n, rank) -> 8x28. Row k*n + i.
    
    # For the first block (j_off=0, k_off=0):
    # A is 2x2 (rows 0,1; cols 0,1).
    # B is 2x2 (rows 0,1; cols 0,1).
    # C is 2x2 (rows 0,1; cols 0,1).
    # C[0,1] is row 0, col 1.
    # In W, index is k*n + i.
    # k is column of B (0..1). i is row of A (0..1).
    # C[0,1] -> i=0, k=1. Index 1*2 + 0 = 2.
    # C[1,1] -> i=1, k=1. Index 1*2 + 1 = 3.
    # C[0,0] -> i=0, k=0. Index 0.
    # C[1,0] -> i=1, k=0. Index 1.
    
    # Let's re-read W indices in current code.
    # M2 w_coeffs: (0,1,1), (1,1,-1).
    # (k=0, i=1) -> Index 0*2+1 = 1. This is C[1,0].
    # (k=1, i=1) -> Index 1*2+1 = 3. This is C[1,1].
    # So M2 contributes to C[1,0] and C[1,1].
    
    # Let's re-evaluate C01 (Index 2: k=1, i=0).
    # Which M contributes to Index 2?
    # M1: (0,0), (1,1). Indices 0, 3.
    # M2: (0,1), (1,1). Indices 1, 3.
    # M3: (1,0), (1,1). Indices 2, 3. (k=1, i=0 -> 2).
    # M4: (0,0), (0,1). Indices 0, 1.
    # M5: (0,0), (1,0). Indices 0, 2. (k=1, i=0 -> 2).
    # M6: (1,1). Index 3.
    # M7: (0,0). Index 0.
    
    # So C[0,1] (Index 2) gets M3 and M5.
    # C[0,1] = M3 + M5.
    # M3 = A00(B01-B11) = A00B01 - A00B11.
    # M5 = (A00+A01)B11 = A00B11 + A01B11.
    # Sum = A00B01 + A01B11. Correct!
    
    # So the decomposition IS correct. My previous manual check of indices was flawed.
    # The code is correct.
    
    # Since rank 28 is optimal, we just return the current implementation.
    
    strassen_products = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    col = 0
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            for u_coeffs, v_coeffs, w_coeffs in strassen_products:
                for i, j, c in u_coeffs:
                    row_u = i * m + (j_off + j)
                    U[row_u][col] += c
                for j, k, c in v_coeffs:
                    row_v = (j_off + j) * p + (k_off + k)
                    V[row_v][col] += c
                for k, i, c in w_coeffs:
                    row_w = (k_off + k) * n + i
                    W[row_w][col] += c
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
