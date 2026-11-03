import numpy as np

def construct():
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, coeff=1):
        nonlocal col
        U[i * m + j][col] = coeff
        V[j * p + k][col] = coeff
        W[k * n + i][col] = coeff
        col += 1
        
    # Block 1: A_left (2x1) * B_top (1x5) -> Rank 10
    # A indices: i in {0,1}, j=0
    # B indices: j=0, k in {0,1,2,3,4}
    for i in range(2):
        for k in range(5):
            add_term(i, 0, k)
            
    # Block 2a: A_right (2x2) * B_bot1 (2x2) -> Strassen Rank 7
    # A indices: i in {0,1}, j in {1,2}
    # B indices: j in {1,2}, k in {0,1}
    # Local A: A_00 -> A[0,1], A_01 -> A[0,2], A_10 -> A[1,1], A_11 -> A[1,2]
    # Local B: B_00 -> B[1,0], B_01 -> B[1,1], B_10 -> B[2,0], B_11 -> B[2,1]
    # Local C: C_00 -> C[0,0], C_01 -> C[0,1], C_10 -> C[1,0], C_11 -> C[1,1]
    
    def add_strassen(i_base, j_base, k_base, A_map, B_map, C_map):
        # A_map: local (li, lj) -> global (gi, gj)
        # B_map: local (lj, lk) -> global (gj, gk)
        # C_map: local (li, lk) -> global (gi, gk)
        nonlocal col
        
        # M1 = (A00 + A11)(B00 + B11)
        def add_m1():
            nonlocal col
            # U
            for li, lj in [(0,0), (1,1)]:
                gi, gj = A_map[li, lj]
                U[gi * m + gj][col] = 1
            # V
            for lj, lk in [(0,0), (1,1)]:
                gj, gk = B_map[lj, lk]
                V[gj * p + gk][col] = 1
            # W
            for li, lk in [(0,0), (0,1), (1,0), (1,1)]:
                gi, gk = C_map[li, lk]
                W[gk * n + gi][col] = 1
            col += 1

        # M2 = (A10 + A11)B00
        def add_m2():
            nonlocal col
            for li, lj in [(1,0), (1,1)]:
                gi, gj = A_map[li, lj]
                U[gi * m + gj][col] = 1
            lj, lk = (0,0)
            gj, gk = B_map[lj, lk]
            V[gj * p + gk][col] = 1
            for li, lk in [(1,0), (1,1)]:
                gi, gk = C_map[li, lk]
                coeff = 1 if li == 1 and lk == 0 else -1
                W[gk * n + gi][col] = coeff
            col += 1

        # M3 = A00 B10
        def add_m3():
            nonlocal col
            li, lj = (0,0)
            gi, gj = A_map[li, lj]
            U[gi * m + gj][col] = 1
            lj, lk = (1,0)
            gj, gk = B_map[lj, lk]
            V[gj * p + gk][col] = 1
            for li, lk in [(0,1), (1,1)]:
                gi, gk = C_map[li, lk]
                W[gk * n + gi][col] = 1
            col += 1

        # M4 = A11 B11
        def add_m4():
            nonlocal col
            li, lj = (1,1)
            gi, gj = A_map[li, lj]
            U[gi * m + gj][col] = 1
            lj, lk = (1,1)
            gj, gk = B_map[lj, lk]
            V[gj * p + gk][col] = 1
            for li, lk in [(0,0), (1,0), (1,1)]:
                gi, gk = C_map[li, lk]
                coeff = -1 if li == 0 and lk == 0 else 1
                W[gk * n + gi][col] = coeff
            col += 1

        # M5 = (A00 + A01)B11
        def add_m5():
            nonlocal col
            for li, lj in [(0,0), (0,1)]:
                gi, gj = A_map[li, lj]
                U[gi * m + gj][col] = 1
            lj, lk = (1,1)
            gj, gk = B_map[lj, lk]
            V[gj * p + gk][col] = 1
            for li, lk in [(0,0), (0,1)]:
                gi, gk = C_map[li, lk]
                coeff = -1 if li == 0 and lk == 0 else 1
                W[gk * n + gi][col] = coeff
            col += 1

        # M6 = (A10 - A00)B01
        def add_m6():
            nonlocal col
            for li, lj in [(1,0), (0,0)]:
                gi, gj = A_map[li, lj]
                coeff = 1 if li == 1 else -1
                U[gi * m + gj][col] = coeff
            lj, lk = (0,1)
            gj, gk = B_map[lj, lk]
            V[gj * p + gk][col] = 1
            li, lk = (1,1)
            gi, gk = C_map[li, lk]
            W[gk * n + gi][col] = 1
            col += 1

        # M7 = (A01 - A11)B10
        def add_m7():
            nonlocal col
            for li, lj in [(0,1), (1,1)]:
                gi, gj = A_map[li, lj]
                coeff = 1 if li == 0 else -1
                U[gi * m + gj][col] = coeff
            lj, lk = (1,0)
            gj, gk = B_map[lj, lk]
            V[gj * p + gk][col] = 1
            for li, lk in [(0,0), (1,1)]:
                gi, gk = C_map[li, lk]
                W[gk * n + gi][col] = 1
            col += 1

        add_m1()
        add_m2()
        add_m3()
        add_m4()
        add_m5()
        add_m6()
        add_m7()

    # Define maps for Block 2a
    # A: local (0,0)->(0,1), (0,1)->(0,2), (1,0)->(1,1), (1,1)->(1,2)
    A_map_2a = {(0,0): (0,1), (0,1): (0,2), (1,0): (1,1), (1,1): (1,2)}
    # B: local (0,0)->(1,0), (0,1)->(1,1), (1,0)->(2,0), (1,1)->(2,1)
    B_map_2a = {(0,0): (1,0), (0,1): (1,1), (1,0): (2,0), (1,1): (2,1)}
    # C: local (0,0)->(0,0), (0,1)->(0,1), (1,0)->(1,0), (1,1)->(1,1)
    C_map_2a = {(0,0): (0,0), (0,1): (0,1), (1,0): (1,0), (1,1): (1,1)}
    
    add_strassen(0, 1, 0, A_map_2a, B_map_2a, C_map_2a)

    # Block 2b1: A_right (2x2) * B_bot21 (2x1) -> Rank 4
    # A indices: i in {0,1}, j in {1,2}
    # B indices: j in {1,2}, k=2
    # Terms:
    # A[0,1] B[1,2] -> C[0,2]
    # A[0,2] B[2,2] -> C[0,2]
    # A[1,1] B[1,2] -> C[1,2]
    # A[1,2] B[2,2] -> C[1,2]
    
    add_term(0, 1, 2)
    add_term(0, 2, 2)
    add_term(1, 1, 2)
    add_term(1, 2, 2)

    # Block 2b2: A_right (2x2) * B_bot22 (2x2) -> Strassen Rank 7
    # A indices: i in {0,1}, j in {1,2}
    # B indices: j in {1,2}, k in {3,4}
    
    A_map_2b2 = {(0,0): (0,1), (0,1): (0,2), (1,0): (1,1), (1,1): (1,2)}
    B_map_2b2 = {(0,0): (1,3), (0,1): (1,4), (1,0): (2,3), (1,1): (2,4)}
    C_map_2b2 = {(0,0): (0,3), (0,1): (0,4), (1,0): (1,3), (1,1): (1,4)}
    
    add_strassen(0, 1, 3, A_map_2b2, B_map_2b2, C_map_2b2)

    return {"U": U, "V": V, "W": W}
