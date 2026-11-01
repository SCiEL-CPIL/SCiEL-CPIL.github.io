def construct():
    n, m, p = 3, 3, 4
    nm = n * m
    mp = m * p
    pn = p * n
    
    rank = 34

    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        U[u_idx][r] += val
        V[v_idx][r] += val
        W[w_idx][r] += val

    # S1: A22 * B22 (r=0..6)
    # U indices: 0,1,3,4
    # V indices: 0,1,4,5
    # W indices: 0,1,3,4
    
    r = 0
    # M1: (x00+x11)(y00+y11)
    add_term(0, 0, 0, 1)
    add_term(4, 0, 0, 1)
    add_term(0, 5, 0, 1)
    add_term(4, 5, 0, 1)
    # This logic is flawed for Strassen because U and V have sums.
    # We must define U and V vectors for each rank-1 term.
    
    # Let's redefine using explicit U, V, W vectors for each r.
    
    # Reset
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    def set_term(r, u_dict, v_dict, w_dict):
        for u_idx, val in u_dict.items():
            U[u_idx][r] = val
        for v_idx, val in v_dict.items():
            V[v_idx][r] = val
        for w_idx, val in w_dict.items():
            W[w_idx][r] = val

    # S1: A22 * B22
    # x00=0, x01=1, x10=3, x11=4
    # y00=0, y01=1, y10=4, y11=5
    # z00=0, z01=1, z10=3, z11=4
    
    # M1
    set_term(0, {0:1, 4:1}, {0:1, 5:1}, {0:1, 4:1})
    # M2
    set_term(1, {3:1, 4:1}, {0:1}, {3:1, 4:-1})
    # M3
    set_term(2, {0:1}, {4:1, 5:-1}, {1:1, 4:1})
    # M4
    set_term(3, {4:1}, {1:1, 0:-1}, {0:1, 3:1})
    # M5
    set_term(4, {0:1, 1:1}, {5:1}, {0:-1, 1:1})
    # M6
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {4:1})
    # M7
    set_term(6, {1:1, 4:-1}, {4:1, 5:1}, {0:1})

    # S2: A22 * b24
    # x00=0, x01=1, x10=3, x11=4
    # y00=2, y01=3, y10=6, y11=7
    # z00=6, z01=7, z10=9, z11=10
    
    # M1
    set_term(7, {0:1, 4:1}, {2:1, 7:1}, {6:1, 10:1})
    # M2
    set_term(8, {3:1, 4:1}, {2:1}, {9:1, 10:-1})
    # M3
    set_term(9, {0:1}, {6:1, 7:-1}, {7:1, 10:1})
    # M4
    set_term(10, {4:1}, {3:1, 2:-1}, {6:1, 9:1})
    # M5
    set_term(11, {0:1, 1:1}, {7:1}, {6:-1, 7:1})
    # M6
    set_term(12, {3:1, 0:-1}, {2:1, 3:1}, {10:1})
    # M7
    set_term(13, {1:1, 4:-1}, {6:1, 7:1}, {6:1})

    # T1: a21 * b12^T
    # x0=2, x1=5
    # y0=8, y1=9
    # z00=0, z01=1, z10=3, z11=4
    
    set_term(14, {2:1}, {8:1}, {0:1})
    set_term(15, {2:1}, {9:1}, {1:1})
    set_term(16, {5:1}, {8:1}, {3:1})
    set_term(17, {5:1}, {9:1}, {4:1})

    # T2: a21 * b14
    # x0=2, x1=5
    # y0=10, y1=11
    # z00=6, z01=7, z10=9, z11=10
    
    set_term(18, {2:1}, {10:1}, {6:1})
    set_term(19, {2:1}, {11:1}, {7:1})
    set_term(20, {5:1}, {10:1}, {9:1})
    set_term(21, {5:1}, {11:1}, {10:1})

    # T3: a12^T * B22
    # x0=6, x1=7
    # y00=0, y01=1, y10=4, y11=5
    # z0=2, z1=5
    
    set_term(22, {6:1}, {0:1}, {2:1})
    set_term(23, {6:1}, {1:1}, {5:1})
    set_term(24, {7:1}, {4:1}, {2:1})
    set_term(25, {7:1}, {5:1}, {5:1})

    # T4: a12^T * b24
    # x0=6, x1=7
    # y00=2, y01=3, y10=6, y11=7
    # z0=8, z1=11
    
    set_term(26, {6:1}, {2:1}, {8:1})
    set_term(27, {6:1}, {3:1}, {11:1})
    set_term(28, {7:1}, {6:1}, {8:1})
    set_term(29, {7:1}, {7:1}, {11:1})

    # T5: a11 * b12^T
    # x=8
    # y0=8, y1=9
    # z0=2, z1=5
    
    set_term(30, {8:1}, {8:1}, {2:1})
    set_term(31, {8:1}, {9:1}, {5:1})

    # T6: a11 * b14
    # x=8
    # y0=10, y1=11
    # z0=8, z1=11
    
    set_term(32, {8:1}, {10:1}, {8:1})
    set_term(33, {8:1}, {11:1}, {11:1})

    return {"U": U, "V": V, "W": W}
