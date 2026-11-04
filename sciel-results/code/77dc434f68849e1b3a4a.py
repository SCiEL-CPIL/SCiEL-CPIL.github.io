"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses a 2+2+1 block partition with Strassen for the 2x2x4 blocks and schoolbook for the rest.
This achieves a rank of 117.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    terms = []
    
    def add_term(U_dict, V_dict, W_dict):
        if time.time() > deadline:
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))
    
    # Helper to add schoolbook terms for a block
    def add_schoolbook(r_start, r_end, c_start, c_end, k_start, k_end):
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                for k in range(k_start, k_end):
                    u_idx = r * m + c
                    v_idx = c * p + k
                    w_idx = k * n + r
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Helper to add Strassen terms for a 2x2x2 block
    def add_strassen_2x2(r_start, c_start, k_start):
        A00 = r_start * m + c_start
        A01 = r_start * m + (c_start + 1)
        A10 = (r_start + 1) * m + c_start
        A11 = (r_start + 1) * m + (c_start + 1)
        
        B00 = c_start * p + k_start
        B01 = c_start * p + (k_start + 1)
        B10 = (c_start + 1) * p + k_start
        B11 = (c_start + 1) * p + (k_start + 1)
        
        C00 = k_start * n + r_start
        C01 = (k_start + 1) * n + r_start
        C10 = k_start * n + (r_start + 1)
        C11 = (k_start + 1) * n + (r_start + 1)
        
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1})
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1})
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01: 1, C11: 1})
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1})
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1})
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1})
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1})
        
    def _build_result(terms, n, m, p):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for r in range(rank):
            U_dict, V_dict, W_dict = terms[r]
            for idx, val in U_dict.items():
                U[idx][r] = val
            for idx, val in V_dict.items():
                V[idx][r] = val
            for idx, val in W_dict.items():
                W[idx][r] = val
        
        return {"U": U, "V": V, "W": W}

    blocks = [(0, 2), (2, 4), (4, 5)]
    
    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline:
                    return _build_result(terms, n, m, p)
                
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                
                if ar == 2 and ac == 2 and bc == 2:
                    add_strassen_2x2(r1, c1, k1)
                else:
                    add_schoolbook(r1, r2, c1, c2, k1, k2)
    
    return _build_result(terms, n, m, p)
# EVOLVE-BLOCK-END
