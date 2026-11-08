def construct():
    import time
    start_time = time.time()
    
    n, m, p = 2, 3, 4
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1
        
    # Part 1: A_L B_T (Strassen)
    # A_L indices: 0,1,3,4
    # B_T indices: 0,1,4,5
    # C_LT indices: 0,1,2,3
    
    # M1
    add_term({0: 1, 4: 1}, {0: 1, 5: 1}, {0: 1, 3: 1})
    # M2
    add_term({3: 1, 4: 1}, {0: 1}, {1: 1})
    # M3
    add_term({0: 1}, {1: 1, 5: -1}, {2: 1, 3: -1})
    # M4
    add_term({4: 1}, {4: 1, 0: -1}, {0: 1})
    # M5
    add_term({0: 1, 3: 1}, {5: 1}, {0: 1})
    # M6
    add_term({4: 1, 1: -1}, {0: 1, 4: 1}, {1: 1, 3: 1})
    # M7
    add_term({1: 1, 4: -1}, {5: 1, 1: 1}, {2: 1, 3: 1})

    # Part 2: A_L B_B (Strassen)
    # A_L indices: 0,1,3,4
    # B_B indices: 8,9,12,13
    # C_LB indices: 4,5,6,7
    
    # M1
    add_term({0: 1, 4: 1}, {8: 1, 13: 1}, {4: 1, 7: 1})
    # M2
    add_term({3: 1, 4: 1}, {8: 1}, {5: 1})
    # M3
    add_term({0: 1}, {9: 1, 13: -1}, {6: 1, 7: -1})
    # M4
    add_term({4: 1}, {12: 1, 8: -1}, {4: 1})
    # M5
    add_term({0: 1, 3: 1}, {13: 1}, {4: 1})
    # M6
    add_term({4: 1, 1: -1}, {8: 1, 12: 1}, {5: 1, 7: 1})
    # M7
    add_term({1: 1, 4: -1}, {13: 1, 9: 1}, {6: 1, 7: 1})

    # Part 3: A_R B_T
    # A_R indices: 2,5
    # B_T indices: 0,1,4,5
    # C indices: 0,1,2,3
    
    # Term for C_00, C_10
    add_term({2: 1, 5: 1}, {0: 1, 4: 1}, {0: 1, 1: 1})
    # Term for C_01, C_11
    add_term({2: 1, 5: 1}, {1: 1, 5: 1}, {2: 1, 3: 1})

    # Part 4: A_R B_B
    # A_R indices: 2,5
    # B_B indices: 8,9,12,13
    # C indices: 4,5,6,7
    
    # Term for C_02, C_12
    add_term({2: 1, 5: 1}, {8: 1, 12: 1}, {4: 1, 5: 1})
    # Term for C_03, C_13
    add_term({2: 1, 5: 1}, {9: 1, 13: 1}, {6: 1, 7: 1})

    if time.time() - start_time > 100:
        return {'U': U, 'V': V, 'W': W}
        
    return {'U': U, 'V': V, 'W': W}
