import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 0,1,3,4
    # B indices: 0,1,5,6
    # W indices: 0,1,2,3
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1; W[1][col] = -1; W[2][col] = -1 
    # Wait, Strassen reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's just store the multiplication factors in U, V and the linear combination in W.
    # The "term" added is the product of the linear combination in U and V.
    # So for M1, U has coeffs for A00, A11. V has coeffs for B00, B11.
    # W should have coeffs for C00, C01, C10, C11 corresponding to M1's contribution.
    # M1 contributes +1 to C00 and +1 to C11.
    
    # Reset col logic: I will just fill the columns directly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_col(c, u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: list of (index, value)
        # v_coeffs: list of (index, value)
        # w_coeffs: list of (index, value)
        for idx, val in u_coeffs:
            U[idx][c] = val
        for idx, val in v_coeffs:
            V[idx][c] = val
        for idx, val in w_coeffs:
            W[idx][c] = val
            
    c = 0
    
    # Part 1: Strassen A[0:2,0:2] B[0:2,0:2]
    # A idx: 0,1,3,4. B idx: 0,1,5,6. W idx: 0,1,2,3.
    
    # M1: (A0+A3)(B0+B6) -> C00 += 1, C11 += 1
    set_col(c, [(0,1),(4,1)], [(0,1),(6,1)], [(0,1),(3,1)]); c+=1
    # M2: (A3+A4)B0 -> C10 += 1, C11 -= 1
    set_col(c, [(3,1),(4,1)], [(0,1)], [(2,1),(3,-1)]); c+=1
    # M3: A0(B1-B6) -> C01 += 1, C11 += 1
    set_col(c, [(0,1)], [(1,1),(6,-1)], [(1,1),(3,1)]); c+=1
    # M4: A4(B5-B0) -> C00 += 1, C10 += 1
    set_col(c, [(4,1)], [(5,1),(0,-1)], [(0,1),(2,1)]); c+=1
    # M5: (A0+A1)B6 -> C00 -= 1, C01 += 1
    set_col(c, [(0,1),(1,1)], [(6,1)], [(0,-1),(1,1)]); c+=1
    # M6: (A3-A0)(B0+B1) -> C11 += 1
    set_col(c, [(3,1),(0,-1)], [(0,1),(1,1)], [(3,1)]); c+=1
    # M7: (A1-A4)(B5+B6) -> C00 += 1
    set_col(c, [(1,1),(4,-1)], [(5,1),(6,1)], [(0,1)]); c+=1
    
    # Part 2: Strassen A[0:2,0:2] B[0:2,2:4]
    # A idx: 0,1,3,4. B idx: 2,3,7,8. W idx: 4,5,6,7.
    
    # M1: (A0+A3)(B2+B8) -> C02 += 1, C13 += 1
    set_col(c, [(0,1),(4,1)], [(2,1),(8,1)], [(4,1),(7,1)]); c+=1
    # M2: (A3+A4)B2 -> C12 += 1, C13 -= 1
    set_col(c, [(3,1),(4,1)], [(2,1)], [(6,1),(7,-1)]); c+=1
    # M3: A0(B3-B8) -> C03 += 1, C13 += 1
    set_col(c, [(0,1)], [(3,1),(8,-1)], [(5,1),(7,1)]); c+=1
    # M4: A4(B7-B2) -> C02 += 1, C12 += 1
    set_col(c, [(4,1)], [(7,1),(2,-1)], [(4,1),(6,1)]); c+=1
    # M5: (A0+A1)B8 -> C02 -= 1, C03 += 1
    set_col(c, [(0,1),(1,1)], [(8,1)], [(4,-1),(5,1)]); c+=1
    # M6: (A3-A0)(B2+B3) -> C13 += 1
    set_col(c, [(3,1),(0,-1)], [(2,1),(3,1)], [(7,1)]); c+=1
    # M7: (A1-A4)(B7+B8) -> C02 += 1
    set_col(c, [(1,1),(4,-1)], [(7,1),(8,1)], [(4,1)]); c+=1
    
    # Part 3: Schoolbook A[0:2,0:2] B[0:2,4]
    # A idx: 0,1,3,4. B idx: 4,9. W idx: 8,9.
    
    # M1: A0 B4 -> C04 += 1
    set_col(c, [(0,1)], [(4,1)], [(8,1)]); c+=1
    # M2: A1 B9 -> C04 += 1
    set_col(c, [(1,1)], [(9,1)], [(8,1)]); c+=1
    # M3: A3 B4 -> C14 += 1
    set_col(c, [(3,1)], [(4,1)], [(9,1)]); c+=1
    # M4: A4 B9 -> C14 += 1
    set_col(c, [(4,1)], [(9,1)], [(9,1)]); c+=1
    
    # Part 4: Schoolbook A[0:2,2] B[2,0:5]
    # A idx: 2,5. B idx: 10,11,12,13,14. W idx: 0..9.
    
    # A02 (idx 2) * B20 (idx 10) -> C00 (idx 0)
    set_col(c, [(2,1)], [(10,1)], [(0,1)]); c+=1
    # A02 (idx 2) * B21 (idx 11) -> C01 (idx 1)
    set_col(c, [(2,1)], [(11,1)], [(1,1)]); c+=1
    # A02 (idx 2) * B22 (idx 12) -> C02 (idx 4)
    set_col(c, [(2,1)], [(12,1)], [(4,1)]); c+=1
    # A02 (idx 2) * B23 (idx 13) -> C03 (idx 5)
    set_col(c, [(2,1)], [(13,1)], [(5,1)]); c+=1
    # A02 (idx 2) * B24 (idx 14) -> C04 (idx 8)
    set_col(c, [(2,1)], [(14,1)], [(8,1)]); c+=1
    
    # A12 (idx 5) * B20 (idx 10) -> C10 (idx 2)
    set_col(c, [(5,1)], [(10,1)], [(2,1)]); c+=1
    # A12 (idx 5) * B21 (idx 11) -> C11 (idx 3)
    set_col(c, [(5,1)], [(11,1)], [(3,1)]); c+=1
    # A12 (idx 5) * B22 (idx 12) -> C12 (idx 6)
    set_col(c, [(5,1)], [(12,1)], [(6,1)]); c+=1
    # A12 (idx 5) * B23 (idx 13) -> C13 (idx 7)
    set_col(c, [(5,1)], [(13,1)], [(7,1)]); c+=1
    # A12 (idx 5) * B24 (idx 14) -> C14 (idx 9)
    set_col(c, [(5,1)], [(14,1)], [(9,1)]); c+=1
    
    return {"U": U, "V": V, "W": W}
