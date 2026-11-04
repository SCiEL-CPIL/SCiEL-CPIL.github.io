import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    return _construct_impl()

def _construct_impl():
    n, m, p = 4, 4, 5
    total_rank = 65
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def _add_strassen_2x2x2(U, V, W, col, a_map, b_map, c_map):
        def add_term(u_coeffs, v_coeffs, w_coeffs, sign):
            nonlocal col
            for idx, coeff in u_coeffs:
                U[idx][col] += coeff * sign
            for idx, coeff in v_coeffs:
                V[idx][col] += coeff * sign
            for idx, coeff in w_coeffs:
                W[idx][col] += coeff * sign
            col += 1

        def get_u(r, c):
            return a_map[(r,c)]
        def get_v(r, c):
            return b_map[(r,c)]
        def get_w(r, c):
            return c_map[(r,c)]

        # m1
        u1 = get_u(0,0) + get_u(1,1)
        v1 = get_v(0,0) + get_v(1,1)
        w1 = get_w(0,0) + get_w(1,1)
        add_term(u1, v1, w1, 1)

        # m2
        u2 = get_u(1,0) + get_u(1,1)
        v2 = get_v(0,0)
        w2 = get_w(1,0)
        add_term(u2, v2, w2, 1)

        # m3
        u3 = get_u(0,0)
        v3 = get_v(0,1) - get_v(1,1)
        w3 = get_w(0,1) - get_w(1,1)
        add_term(u3, v3, w3, 1)

        # m4
        u4 = get_u(1,1)
        v4 = get_v(1,0) - get_v(0,0)
        w4 = -get_w(0,0) + get_w(1,0)
        add_term(u4, v4, w4, 1)

        # m5
        u5 = get_u(0,0) + get_u(0,1)
        v5 = get_v(1,1)
        w5 = get_w(0,1)
        add_term(u5, v5, w5, 1)

        # m6
        u6 = get_u(1,0) - get_u(0,0)
        v6 = get_v(0,0) + get_v(0,1)
        w6 = get_w(0,0)
        add_term(u6, v6, w6, 1)

        # m7
        u7 = get_u(0,1) - get_u(1,1)
        v7 = get_v(1,0) + get_v(1,1)
        w7 = get_w(1,1)
        add_term(u7, v7, w7, 1)

        return col

    def a_idx(r, c): return r * 4 + c
    def b_idx(r, c): return r * 5 + c
    def w_idx(r, c): return c * 4 + r
    
    # Step 1
    a_map_1 = {
        (0,0): [(a_idx(0,0), 1), (a_idx(2,2), 1)],
        (0,1): [(a_idx(0,1), 1), (a_idx(2,3), 1)],
        (1,0): [(a_idx(1,0), 1), (a_idx(3,2), 1)],
        (1,1): [(a_idx(1,1), 1), (a_idx(3,3), 1)]
    }
    b_map_1 = {
        (0,0): [(b_idx(0,0), 1), (b_idx(2,2), 1)],
        (0,1): [(b_idx(0,1), 1), (b_idx(2,3), 1)],
        (1,0): [(b_idx(1,0), 1), (b_idx(3,2), 1)],
        (1,1): [(b_idx(1,1), 1), (b_idx(3,3), 1)]
    }
    c_map_1 = {
        (0,0): [(w_idx(0,0), 1), (w_idx(2,2), 1)],
        (0,1): [(w_idx(0,1), 1), (w_idx(2,3), 1)],
        (1,0): [(w_idx(1,0), 1), (w_idx(3,2), 1)],
        (1,1): [(w_idx(1,1), 1), (w_idx(3,3), 1)]
    }
    col = _add_strassen_2x2x2(U, V, W, 0, a_map_1, b_map_1, c_map_1)

    # Step 2
    a_map_2 = {
        (0,0): [(a_idx(2,0), 1), (a_idx(2,2), 1)],
        (0,1): [(a_idx(2,1), 1), (a_idx(2,3), 1)],
        (1,0): [(a_idx(3,0), 1), (a_idx(3,2), 1)],
        (1,1): [(a_idx(3,1), 1), (a_idx(3,3), 1)]
    }
    b_map_2 = {
        (0,0): [(b_idx(0,0), 1)],
        (0,1): [(b_idx(0,1), 1)],
        (1,0): [(b_idx(1,0), 1)],
        (1,1): [(b_idx(1,1), 1)]
    }
    c_map_2 = {
        (0,0): [(w_idx(2,0), 1)],
        (0,1): [(w_idx(2,1), 1)],
        (1,0): [(w_idx(3,0), 1)],
        (1,1): [(w_idx(3,1), 1)]
    }
    col = _add_strassen_2x2x2(U, V, W, col, a_map_2, b_map_2, c_map_2)

    # Step 3
    a_map_3 = {
        (0,0): [(a_idx(0,0), 1)],
        (0,1): [(a_idx(0,1), 1)],
        (1,0): [(a_idx(1,0), 1)],
        (1,1): [(a_idx(1,1), 1)]
    }
    b_map_3 = {
        (0,0): [(b_idx(0,1), 1), (b_idx(2,1), -1)],
        (0,1): [(b_idx(0,2), 1), (b_idx(2,2), -1)],
        (1,0): [(b_idx(1,1), 1), (b_idx(3,1), -1)],
        (1,1): [(b_idx(1,2), 1), (b_idx(3,2), -1)]
    }
    c_map_3 = {
        (0,0): [(w_idx(0,2), 1), (w_idx(2,2), -1)],
        (0,1): [(w_idx(0,3), 1), (w_idx(2,3), -1)],
        (1,0): [(w_idx(1,2), 1), (w_idx(3,2), -1)],
        (1,1): [(w_idx(1,3), 1), (w_idx(3,3), -1)]
    }
    col = _add_strassen_2x2x2(U, V, W, col, a_map_3, b_map_3, c_map_3)

    # Step 4
    a_map_4 = {
        (0,0): [(a_idx(2,2), 1)],
        (0,1): [(a_idx(2,3), 1)],
        (1,0): [(a_idx(3,2), 1)],
        (1,1): [(a_idx(3,3), 1)]
    }
    b_map_4 = {
        (0,0): [(b_idx(2,0), 1), (b_idx(0,0), -1)],
        (0,1): [(b_idx(2,1), 1), (b_idx(0,1), -1)],
        (1,0): [(b_idx(3,0), 1), (b_idx(1,0), -1)],
        (1,1): [(b_idx(3,1), 1), (b_idx(1,1), -1)]
    }
    c_map_4 = {
        (0,0): [(w_idx(0,0), -1), (w_idx(2,0), 1)],
        (0,1): [(w_idx(0,1), -1), (w_idx(2,1), 1)],
        (1,0): [(w_idx(1,0), -1), (w_idx(3,0), 1)],
        (1,1): [(w_idx(1,1), -1), (w_idx(3,1), 1)]
    }
    col = _add_strassen_2x2x2(U, V, W, col, a_map_4, b_map_4, c_map_4)

    # Step 5
    a_map_5 = {
        (0,0): [(a_idx(0,0), 1), (a_idx(0,2), 1)],
        (0,1): [(a_idx(0,1), 1), (a_idx(0,3), 1)],
        (1,0): [(a_idx(1,0), 1), (a_idx(1,2), 1)],
        (1,1): [(a_idx(1,1), 1), (a_idx(1,3), 1)]
    }
    b_map_5 = {
        (0,0): [(b_idx(2,2), 1)],
        (0,1): [(b_idx(2,3), 1)],
        (1,0): [(b_idx(3,2), 1)],
        (1,1): [(b_idx(3,3), 1)]
    }
    c_map_5 = {
        (0,0): [(w_idx(0,2), 1)],
        (0,1): [(w_idx(0,3), 1)],
        (1,0): [(w_idx(1,2), 1)],
        (1,1): [(w_idx(1,3), 1)]
    }
    col = _add_strassen_2x2x2(U, V, W, col, a_map_5, b_map_5, c_map_5)

    # Step 6
    a_map_6 = {
        (0,0): [(a_idx(2,0), 1), (a_idx(0,0), -1)],
        (0,1): [(a_idx(2,1), 1), (a_idx(0,1), -1)],
        (1,0): [(a_idx(3,0), 1), (a_idx(1,0), -1)],
        (1,1): [(a_idx(3,1), 1), (a_idx(1,1), -1)]
    }
    b_map_6 = {
        (0,0): [(b_idx(0,0), 1), (b_idx(0,2), 1)],
        (0,1): [(b_idx(0,1), 1), (b_idx(0,3), 1)],
        (1,0): [(b_idx(1,0), 1), (b_idx(1,2), 1)],
        (1,1): [(b_idx(1,1), 1), (b_idx(1,3), 1)]
    }
    c_map_6 = {
        (0,0): [(w_idx(0,0), 1)],
        (0,1): [(w_idx(0,1), 1)],
        (1,0): [(w_idx(1,0), 1)],
        (1,1): [(w_idx(1,1), 1)]
    }
    col = _add_strassen_2x2x2(U, V, W, col, a_map_6, b_map_6, c_map_6)

    # Step 7
    a_map_7 = {
        (0,0): [(a_idx(0,2), 1), (a_idx(2,2), -1)],
        (0,1): [(a_idx(0,3), 1), (a_idx(2,3), -1)],
        (1,0): [(a_idx(1,2), 1), (a_idx(3,2), -1)],
        (1,1): [(a_idx(1,3), 1), (a_idx(3,3), -1)]
    }
    b_map_7 = {
        (0,0): [(b_idx(2,0), 1), (b_idx(2,2), 1)],
        (0,1): [(b_idx(2,1), 1), (b_idx(2,3), 1)],
        (1,0): [(b_idx(3,0), 1), (b_idx(3,2), 1)],
        (1,1): [(b_idx(3,1), 1), (b_idx(3,3), 1)]
    }
    c_map_7 = {
        (0,0): [(w_idx(2,2), 1)],
        (0,1): [(w_idx(2,3), 1)],
        (1,0): [(w_idx(3,2), 1)],
        (1,1): [(w_idx(3,3), 1)]
    }
    col = _add_strassen_2x2x2(U, V, W, col, a_map_7, b_map_7, c_map_7)
    
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
            
    assert col == total_rank
    return {"U": U, "V": V, "W": W}
