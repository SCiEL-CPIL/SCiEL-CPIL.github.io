import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1

    def idx_a(i, j):
        return i * m + j

    def idx_b(j, k):
        return j * p + k

    def idx_w(k, i):
        return k * n + i

    def add_strassen_2x2(a_i, a_j, b_j, b_k, c_i, c_k):
        # M1 = (A00+A11)(B00+B11)
        u = {idx_a(a_i, a_j): 1, idx_a(a_i+1, a_j+1): 1}
        v = {idx_b(b_j, b_k): 1, idx_b(b_j+1, b_k+1): 1}
        w = {idx_w(c_k, c_i): 1, idx_w(c_k+1, c_i+1): 1}
        add_rank1_term(u, v, w)

        # M2 = (A10+A11)B00
        u = {idx_a(a_i+1, a_j): 1, idx_a(a_i+1, a_j+1): 1}
        v = {idx_b(b_j, b_k): 1}
        w = {idx_w(c_k, c_i+1): 1, idx_w(c_k+1, c_i+1): -1}
        add_rank1_term(u, v, w)

        # M3 = A00(B01-B11)
        u = {idx_a(a_i, a_j): 1}
        v = {idx_b(b_j, b_k+1): 1, idx_b(b_j+1, b_k+1): -1}
        w = {idx_w(c_k+1, c_i): 1, idx_w(c_k+1, c_i+1): 1}
        add_rank1_term(u, v, w)

        # M4 = A11(B10-B00)
        u = {idx_a(a_i+1, a_j+1): 1}
        v = {idx_b(b_j+1, b_k): 1, idx_b(b_j, b_k): -1}
        w = {idx_w(c_k, c_i): 1, idx_w(c_k, c_i+1): 1}
        add_rank1_term(u, v, w)

        # M5 = (A00+A01)B11
        u = {idx_a(a_i, a_j): 1, idx_a(a_i, a_j+1): 1}
        v = {idx_b(b_j+1, b_k+1): 1}
        w = {idx_w(c_k, c_i): -1, idx_w(c_k+1, c_i): 1}
        add_rank1_term(u, v, w)

        # M6 = (A10-A00)(B00+B01)
        u = {idx_a(a_i+1, a_j): 1, idx_a(a_i, a_j): -1}
        v = {idx_b(b_j, b_k): 1, idx_b(b_j, b_k+1): 1}
        w = {idx_w(c_k+1, c_i+1): 1}
        add_rank1_term(u, v, w)

        # M7 = (A01-A11)(B10+B11)
        u = {idx_a(a_i, a_j+1): 1, idx_a(a_i+1, a_j+1): -1}
        v = {idx_b(b_j+1, b_k): 1, idx_b(b_j+1, b_k+1): 1}
        w = {idx_w(c_k, c_i): 1}
        add_rank1_term(u, v, w)

    def add_schoolbook_2x2_1(a_i, a_j, b_j, b_k, c_i, c_k):
        # C00 = A00*B00 + A01*B10
        u = {idx_a(a_i, a_j): 1}
        v = {idx_b(b_j, b_k): 1}
        w = {idx_w(c_k, c_i): 1}
        add_rank1_term(u, v, w)
        
        # C00 += A01*B10
        u = {idx_a(a_i, a_j+1): 1}
        v = {idx_b(b_j+1, b_k): 1}
        w = {idx_w(c_k, c_i): 1}
        add_rank1_term(u, v, w)
        
        # C10 = A10*B00 + A11*B10
        u = {idx_a(a_i+1, a_j): 1}
        v = {idx_b(b_j, b_k): 1}
        w = {idx_w(c_k, c_i+1): 1}
        add_rank1_term(u, v, w)
        
        # C10 += A11*B10
        u = {idx_a(a_i+1, a_j+1): 1}
        v = {idx_b(b_j+1, b_k): 1}
        w = {idx_w(c_k, c_i+1): 1}
        add_rank1_term(u, v, w)

    # Block 1: A1 (cols 0-1) * B1 (rows 0-1)
    # B1 has 5 cols: 0-1, 2-3, 4
    add_strassen_2x2(0, 0, 0, 0, 0, 0)
    add_strassen_2x2(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)

    # Block 2: A2 (cols 2-3) * B2 (rows 2-3)
    # B2 has 5 cols: 0-1, 2-3, 4
    add_strassen_2x2(0, 2, 2, 0, 0, 0)
    add_strassen_2x2(0, 2, 2, 2, 0, 2)
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)

    return {"U": U, "V": V, "W": W}
