"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = 2, 4, 4
    
    # Try rank 25 decomposition using shared Strassen structure
    # Partition: j into blocks of 2 (j0, j1), k into blocks of 2 (k0, k1)
    # A: (2,4) -> A0=(2,2) [cols 0,1], A1=(2,2) [cols 2,3]
    # B: (4,4) -> B00=(2,2), B01=(2,2), B10=(2,2), B11=(2,2)
    # C: (4,2) -> C0=(2,2) [rows 0,1], C1=(2,2) [rows 2,3]
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    
    # Strassen for 2x2*2x2 (7 products):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For C0 = A0*B00 + A1*B10:
    # Let's define combined matrices:
    # P1 = A00 + A11  (from A0)
    # P2 = A10 + A11  (from A0)  
    # P3 = A00        (from A0)
    # P4 = A11        (from A0)
    # P5 = A00 + A01  (from A0)
    # P6 = A10 - A00  (from A0)
    # P7 = A01 - A11  (from A0)
    # P8 = A00 + A11  (from A1) -- same as P1!
    # P9 = A10 + A11  (from A1) -- same as P2!
    # P10 = A00       (from A1) -- same as P3!
    # P11 = A11       (from A1) -- same as P4!
    # P12 = A00 + A01 (from A1) -- same as P5!
    # P13 = A10 - A00 (from A1) -- same as P6!
    # P14 = A01 - A11 (from A1) -- same as P7!
    
    # So all 14 products reduce to 7 unique A-side factors!
    # But the B-side factors are different:
    # For C0: Q1=B00+B10, Q2=B00, Q3=B10-B00, Q4=B00+B10, Q5=B10, Q6=B00+B10, Q7=B10+B00
    # Wait, let me be more careful.
    
    # C0 = A0*B00 + A1*B10
    # Strassen on A0*B00:
    #   S1 = (A00+A11)(B00_00+B00_11)
    #   S2 = (A10+A11)B00_00
    #   S3 = A00(B00_01-B00_11)
    #   S4 = A11(B00_10-B00_00)
    #   S5 = (A00+A01)B00_11
    #   S6 = (A10-A00)(B00_00+B00_01)
    #   S7 = (A01-A11)(B00_10+B00_11)
    # Strassen on A1*B10:
    #   T1 = (A1_00+A1_11)(B10_00+B10_11)
    #   T2 = (A1_10+A1_11)B10_00
    #   T3 = A1_00(B10_01-B10_11)
    #   T4 = A1_11(B10_10-B10_00)
    #   T5 = (A1_00+A1_01)B10_11
    #   T6 = (A1_10-A1_00)(B10_00+B10_01)
    #   T7 = (A1_01-A1_11)(B10_10+B10_11)
    
    # A-side factors (using A0 elements a00,a01,a10,a11 and A1 elements b00,b01,b10,b11):
    # From A0*B00: a00+a11, a10+a11, a00, a11, a00+a01, a10-a00, a01-a11
    # From A1*B10: b00+b11, b10+b11, b00, b11, b00+b01, b10-b00, b01-b11
    # These are 14 distinct A-side factors (no overlap between A0 and A1 factors)
    
    # B-side factors:
    # From A0*B00: B00_00+B00_11, B00_00, B00_01-B00_11, B00_10-B00_00, B00_11, B00_00+B00_01, B00_10+B00_11
    # From A1*B10: B10_00+B10_11, B10_00, B10_01-B10_11, B10_10-B10_00, B10_11, B10_00+B10_01, B10_10+B10_11
    # These are 14 distinct B-side factors
    
    # Total: 14 products for C0, 14 for C1 = 28. No savings.
    
    # Let me try a different approach: use the full 2x4 x 4x2 Strassen-like decomposition
    # The tensor [2,4,4] has rank at most 28 (4x Strassen).
    # Lower bound: max(2*4, 4*4, 4*2) = 16, but Strassen gives 28.
    # Actually, for [2,4,4], the rank is known to be 28? Let me verify.
    
    # Actually, let me just use the rank 28 Strassen decomposition which is verified.
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    # Strassen coefficients for 2x2 * 2x2
    u_coeffs = [
        [[1, 0], [0, 1]],   # M1
        [[0, 0], [1, 1]],   # M2
        [[1, 0], [0, 0]],   # M3
        [[0, 0], [0, 1]],   # M4
        [[1, 1], [0, 0]],   # M5
        [[-1, 0], [1, 0]],  # M6
        [[0, 1], [0, -1]],  # M7
    ]
    v_coeffs = [
        [[1, 0], [0, 1]],   # M1
        [[1, 0], [0, 0]],   # M2
        [[0, 1], [0, -1]],  # M3
        [[-1, 0], [1, 0]],  # M4
        [[0, 0], [0, 1]],   # M5
        [[1, 1], [0, 0]],   # M6
        [[0, 0], [1, 1]],   # M7
    ]
    w_coeffs = [
        [[1, 0], [0, 1]],   # M1
        [[0, 1], [0, -1]],  # M2
        [[0, 0], [1, 1]],   # M3
        [[1, 1], [0, 0]],   # M4
        [[-1, 1], [0, 0]],  # M5
        [[0, 0], [0, 1]],   # M6
        [[1, 0], [0, 0]],   # M7
    ]
    
    for s in range(4):
        j_block = s // 2
        k_block = s % 2
        for f in range(7):
            col = s * 7 + f
            for pi in range(2):
                for pj in range(2):
                    val = u_coeffs[f][pi][pj]
                    if val != 0:
                        j = 2 * j_block + pj
                        U[pi * m + j][col] += val
            for pj in range(2):
                for qk in range(2):
                    val = v_coeffs[f][pj][qk]
                    if val != 0:
                        j = 2 * j_block + pj
                        k = 2 * k_block + qk
                        V[j * p + k][col] += val
            for qk in range(2):
                for pi in range(2):
                    val = w_coeffs[f][qk][pi]
                    if val != 0:
                        k = 2 * k_block + qk
                        W[k * n + pi][col] += val
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
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
# EVOLVE-BLOCK-END
