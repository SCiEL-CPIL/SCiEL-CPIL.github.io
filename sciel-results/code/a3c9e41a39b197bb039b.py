import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank component
    # u_indices: list of (i, j)
    # v_indices: list of (j, k)
    # w_indices: list of (k, i)
    def add_component(c, u_list, v_list, w_list):
        for i, j in u_list:
            U[i * m + j][c] = 1
        for j, k in v_list:
            V[j * p + k][c] = 1
        for k, i in w_list:
            W[k * n + i][c] = 1

    # Strassen Part (Rank 7) for B columns 0-3
    # Blocks:
    # A11: i 0-1, j 0-1
    # A12: i 0-1, j 2-3
    # A21: i 2-3, j 0-1
    # A22: i 2-3, j 2-3
    # B11: j 0-1, k 0-1
    # B12: j 0-1, k 2-3
    # B21: j 2-3, k 0-1
    # B22: j 2-3, k 2-3
    
    # M1 = A11(B11 + B22)
    u_m1 = [(i, j) for i in [0, 1] for j in [0, 1]]
    v_m1 = [(j, k) for j in [0, 1] for k in [0, 1]] + [(j, k) for j in [2, 3] for k in [2, 3]]
    w_m1 = [(k, i) for k in [0, 1] for i in [0, 1]] + [(k, i) for k in [2, 3] for i in [2, 3]]
    add_component(0, u_m1, v_m1, w_m1)
    
    # M2 = (A21 + A22)B11
    u_m2 = [(i, j) for i in [2, 3] for j in [0, 3]]
    v_m2 = [(j, k) for j in [0, 1] for k in [0, 1]]
    w_m2 = [(k, i) for k in [0, 1] for i in [2, 3]] + [(k, i) for k in [2, 3] for i in [2, 3]]
    # Wait, M2 contributes to C21 and C22.
    # C21: i 2-3, k 0-1.
    # C22: i 2-3, k 2-3.
    w_m2 = [(k, i) for k in [0, 1] for i in [2, 3]] + [(k, i) for k in [2, 3] for i in [2, 3]]
    add_component(1, u_m2, v_m2, w_m2)
    
    # M3 = A12(B21 + B22)
    u_m3 = [(i, j) for i in [0, 1] for j in [2, 3]]
    v_m3 = [(j, k) for j in [2, 3] for k in [0, 1]] + [(j, k) for j in [2, 3] for k in [2, 3]]
    w_m3 = [(k, i) for k in [2, 3] for i in [0, 1]] + [(k, i) for k in [2, 3] for i in [2, 3]]
    # M3 contributes to C12 and C22.
    # C12: i 0-1, k 2-3.
    # C22: i 2-3, k 2-3.
    add_component(2, u_m3, v_m3, w_m3)
    
    # M4 = A22(B22 - B12)
    u_m4 = [(i, j) for i in [2, 3] for j in [2, 3]]
    v_m4 = [(j, k) for j in [2, 3] for k in [2, 3]] + [(j, k) for j in [0, 1] for k in [2, 3]]
    # Need to handle subtraction in V?
    # The tensor decomposition allows negative entries.
    # V entries for B12 should be -1.
    # My add_component sets 1. I need to modify it or handle signs.
    pass
