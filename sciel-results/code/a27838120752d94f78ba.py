import time

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 26 decomposition
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_idx: list of (row, coeff)
    # v_idx: list of (row, coeff)
    # w_idx: list of (row, coeff)
    def add_term(col, u_idx, v_idx, w_idx):
        for r, c in u_idx:
            U[r][col] += c
        for r, c in v_idx:
            V[r][col] += c
        for r, c in w_idx:
            W[r][col] += c
            
    col = 0
    
    # 1. Strassen for A1 B1 (7 terms)
    # A1: rows 0,1 cols 0,1. B1: rows 0,1 cols 0,1.
    # U indices: i*3+j. V indices: j*3+k. W indices: k*3+i.
    
    # M1 = (A00+A11)(B00+B11)
    # U: (0,1), (4,1)
    # V: (0,1), (4,1)
    # W: M00(0,1), M01(3,1), M10(1,1), M11(4,1)
    add_term(col, [(0,1), (4,1)], [(0,1), (4,1)], [(0,1), (3,1), (1,1), (4,1)])
    col += 1
    
    # M2 = (A10+A11)B00
    # U: (3,1), (4,1)
    # V: (0,1)
    # W: M10(1,1), M11(4,-1)
    add_term(col, [(3,1), (4,1)], [(0,1)], [(1,1), (4,-1)])
    col += 1
    
    # M3 = A00(B01-B11)
    # U: (0,1)
    # V: (1,1), (4,-1)
    # W: M01(3,1), M11(4,1)
    add_term(col, [(0,1)], [(1,1), (4,-1)], [(3,1), (4,1)])
    col += 1
    
    # M4 = A11(B10-B00)
    # U: (4,1)
    # V: (3,1), (0,-1)
    # W: M00(0,1), M10(1,1)
    add_term(col, [(4,1)], [(3,1), (0,-1)], [(0,1), (1,1)])
    col += 1
    
    # M5 = (A00+A01)B11
    # U: (0,1), (1,1)
    # V: (4,1)
    # W: M00(0,-1), M01(3,1)
    add_term(col, [(0,1), (1,1)], [(4,1)], [(0,-1), (3,1)])
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    # U: (3,1), (0,-1)
    # V: (0,1), (1,1)
    # W: M11(4,1)
    add_term(col, [(3,1), (0,-1)], [(0,1), (1,1)], [(4,1)])
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    # U: (1,1), (4,-1)
    # V: (3,1), (4,1)
    # W: M00(0,1)
    add_term(col, [(1,1), (4,-1)], [(3,1), (4,1)], [(0,1)])
    col += 1
    
    # 2. a e^T (4 terms)
    # a: A02, A12. e: B20, B21.
    # U: A02(2), A12(5). V: B20(6), B21(7).
    # M00(0), M01(3), M10(1), M11(4).
    
    # T1: A02 B20 -> M00
    add_term(col, [(2,1)], [(6,1)], [(0,1)])
    col += 1
    
    # T2: A02 B21 -> M01
    add_term(col, [(2,1)], [(7,1)], [(3,1)])
    col += 1
    
    # T3: A12 B20 -> M10
    add_term(col, [(5,1)], [(6,1)], [(1,1)])
    col += 1
    
    # T4: A12 B21 -> M11
    add_term(col, [(5,1)], [(7,1)], [(4,1)])
    col += 1
    
    # 3. A1 d (4 terms)
    # A1: A00(0), A01(1), A10(3), A11(4). d: B02(2), B12(5).
    # M02(6), M12(7).
    
    # T5: A00 B02 -> M02
    add_term(col, [(0,1)], [(2,1)], [(6,1)])
    col += 1
    
    # T6: A01 B12 -> M02
    add_term(col, [(1,1)], [(5,1)], [(6,1)])
    col += 1
    
    # T7: A10 B02 -> M12
    add_term(col, [(3,1)], [(2,1)], [(7,1)])
    col += 1
    
    # T8: A11 B12 -> M12
    add_term(col, [(4,1)], [(5,1)], [(7,1)])
    col += 1
    
    # 4. a f (2 terms)
    # a: A02(2), A12(5). f: B22(8).
    # M02(6), M12(7).
    
    # T9: A02 B22 -> M02
    add_term(col, [(2,1)], [(8,1)], [(6,1)])
    col += 1
    
    # T10: A12 B22 -> M12
    add_term(col, [(5,1)], [(8,1)], [(7,1)])
    col += 1
    
    # 5. b^T B1 (4 terms)
    # b: A20(6), A21(7). B1: B00(0), B01(1), B10(3), B11(4).
    # M20(2), M21(5).
    
    # T11: A20 B00 -> M20
    add_term(col, [(6,1)], [(0,1)], [(2,1)])
    col += 1
    
    # T12: A21 B10 -> M20
    add_term(col, [(7,1)], [(3,1)], [(2,1)])
    col += 1
    
    # T13: A20 B01 -> M21
    add_term(col, [(6,1)], [(1,1)], [(5,1)])
    col += 1
    
    # T14: A21 B11 -> M21
    add_term(col, [(7,1)], [(4,1)], [(5,1)])
    col += 1
    
    # 6. c e^T (2 terms)
    # c: A22(8). e: B20(6), B21(7).
    # M20(2), M21(5).
    
    # T15: A22 B20 -> M20
    add_term(col, [(8,1)], [(6,1)], [(2,1)])
    col += 1
    
    # T16: A22 B21 -> M21
    add_term(col, [(8,1)], [(7,1)], [(5,1)])
    col += 1
    
    # 7. b^T d (2 terms)
    # b: A20(6), A21(7). d: B02(2), B12(5).
    # M22(8).
    
    # T17: A20 B02 -> M22
    add_term(col, [(6,1)], [(2,1)], [(8,1)])
    col += 1
    
    # T18: A21 B12 -> M22
    add_term(col, [(7,1)], [(5,1)], [(8,1)])
    col += 1
    
    # 8. c f (1 term)
    # c: A22(8). f: B22(8).
    # M22(8).
    
    # T19: A22 B22 -> M22
    add_term(col, [(8,1)], [(8,1)], [(8,1)])
    col += 1
    
    return {"U": U, "V": V, "W": W}
