import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = 145  # seconds, safety margin before 150s limit

    n, m, p = DIMENSIONS  # 4, 4, 9

    # U: i*m + j (16 entries)
    # V: j*p + k (36 entries)  
    # W: k*n + i (36 entries)

    rank = 108
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    term_idx = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal term_idx
        if term_idx >= rank:
            return
        for idx, coeff in u_coeffs.items():
            if coeff != 0:
                U[idx][term_idx] += coeff
        for idx, coeff in v_coeffs.items():
            if coeff != 0:
                V[idx][term_idx] += coeff
        for idx, coeff in w_coeffs.items():
            if coeff != 0:
                W[idx][term_idx] += coeff
        term_idx += 1

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_lin_comb(d1, c1, d2, c2):
        res = {}
        for k, v in d1.items():
            res[k] = res.get(k, 0) + c1 * v
        for k, v in d2.items():
            res[k] = res.get(k, 0) + c2 * v
        return res

    # Part 1: A (4x4) x B1 (4x4) using Strassen
    # A blocks (2x2 each):
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    A11 = {u_idx(0,0): 1, u_idx(0,1): 1, u_idx(1,0): 1, u_idx(1,1): 1}
    A12 = {u_idx(0,2): 1, u_idx(0,3): 1, u_idx(1,2): 1, u_idx(1,3): 1}
    A21 = {u_idx(2,0): 1, u_idx(2,1): 1, u_idx(3,0): 1, u_idx(3,1): 1}
    A22 = {u_idx(2,2): 1, u_idx(2,3): 1, u_idx(3,2): 1, u_idx(3,3): 1}
    
    # B1 blocks (2x2 each) - columns 0-3 of B:
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    B11 = {v_idx(0,0): 1, v_idx(0,1): 1, v_idx(1,0): 1, v_idx(1,1): 1}
    B12 = {v_idx(0,2): 1, v_idx(0,3): 1, v_idx(1,2): 1, v_idx(1,3): 1}
    B21 = {v_idx(2,0): 1, v_idx(2,1): 1, v_idx(3,0): 1, v_idx(3,1): 1}
    B22 = {v_idx(2,2): 1, v_idx(2,3): 1, v_idx(3,2): 1, v_idx(3,3): 1}
    
    # Strassen products for 4x4 x 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mi is 2x2, computed with schoolbook (4 terms)
    # C1 = A*B1 has blocks:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # C11: rows 0-1, cols 0-1 -> w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)
    # C12: rows 0-1, cols 2-3 -> w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1)
    # C21: rows 2-3, cols 0-1 -> w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3)
    # C22: rows 2-3, cols 2-3 -> w_idx(2,2), w_idx(2,3), w_idx(3,2), w_idx(3,3)
    
    # Define W indices for each C block
    C11_w = {0: w_idx(0,0), 1: w_idx(0,1), 4: w_idx(1,0), 5: w_idx(1,1)}  # (local_row, local_col) -> global
    C12_w = {0: w_idx(2,0), 1: w_idx(2,1), 4: w_idx(3,0), 5: w_idx(3,1)}
    C21_w = {0: w_idx(0,2), 1: w_idx(0,3), 4: w_idx(1,2), 5: w_idx(1,3)}
    C22_w = {0: w_idx(2,2), 1: w_idx(2,3), 4: w_idx(3,2), 5: w_idx(3,3)}
    
    # Helper: add a 2x2 schoolbook product to specified C blocks with signs
    def add_2x2_schoolbook(A_entries, B_entries, contributions):
        """
        A_entries: list of 4 dicts [A00, A01, A10, A11]
        B_entries: list of 4 dicts [B00, B01, B10, B11]
        contributions: list of (c_block_w_dict, sign)
        """
        # M00 = A00*B00 + A01*B10
        # M01 = A00*B01 + A01*B11
        # M10 = A10*B00 + A11*B10
        # M11 = A10*B01 + A11*B11
        
        # Term 1: A00 * B00 -> contributes to M00
        u = A_entries[0]
        v = B_entries[0]
        for c_block, sign in contributions:
            w = {0: sign}  # local (0,0)
            # Map local (0,0) to global via c_block
            global_w = {c_block[0]: sign}
            add_term(u, v, global_w)
        
        # Term 2: A01 * B10 -> contributes to M00
        u = A_entries[1]
        v = B_entries[2]
        for c_block, sign in contributions:
            global_w = {c_block[0]: sign}
            add_term(u, v, global_w)
        
        # Term 3: A00 * B01 -> contributes to M01
        u = A_entries[0]
        v = B_entries[1]
        for c_block, sign in contributions:
            global_w = {c_block[1]: sign}
            add_term(u, v, global_w)
        
        # Term 4: A01 * B11 -> contributes to M01
        u = A_entries[1]
        v = B_entries[3]
        for c_block, sign in contributions:
            global_w = {c_block[1]: sign}
            add_term(u, v, global_w)
        
        # Term 5: A10 * B00 -> contributes to M10
        u = A_entries[2]
        v = B_entries[0]
        for c_block, sign in contributions:
            global_w = {c_block[4]: sign}
            add_term(u, v, global_w)
        
        # Term 6: A11 * B10 -> contributes to M10
        u = A_entries[3]
        v = B_entries[2]
        for c_block, sign in contributions:
            global_w = {c_block[4]: sign}
            add_term(u, v, global_w)
        
        # Term 7: A10 * B01 -> contributes to M11
        u = A_entries[2]
        v = B_entries[1]
        for c_block, sign in contributions:
            global_w = {c_block[5]: sign}
            add_term(u, v, global_w)
        
        # Term 8: A11 * B11 -> contributes to M11
        u = A_entries[3]
        v = B_entries[3]
        for c_block, sign in contributions:
            global_w = {c_block[5]: sign}
            add_term(u, v, global_w)
    
    # Now compute each Mi and add to appropriate C blocks
    
    # M1 = (A11+A22)(B11+B22)
    A_M1_00 = add_lin_comb({u_idx(0,0):1}, 1, {u_idx(2,2):1}, 1)
    A_M1_01 = add_lin_comb({u_idx(0,1):1}, 1, {u_idx(2,3):1}, 1)
    A_M1_10 = add_lin_comb({u_idx(1,0):1}, 1, {u_idx(3,2):1}, 1)
    A_M1_11 = add_lin_comb({u_idx(1,1):1}, 1, {u_idx(3,3):1}, 1)
    B_M1_00 = add_lin_comb({v_idx(0,0):1}, 1, {v_idx(2,2):1}, 1)
    B_M1_01 = add_lin_comb({v_idx(0,1):1}, 1, {v_idx(2,3):1}, 1)
    B_M1_10 = add_lin_comb({v_idx(1,0):1}, 1, {v_idx(3,2):1}, 1)
    B_M1_11 = add_lin_comb({v_idx(1,1):1}, 1, {v_idx(3,3):1}, 1)
    
    # M1 contributes to C11 (+1) and C22 (+1)
    add_2x2_schoolbook([A_M1_00, A_M1_01, A_M1_10, A_M1_11],
                       [B_M1_00, B_M1_01, B_M1_10, B_M1_11],
                       [(C11_w, 1), (C22_w, 1)])
    
    # M2 = (A21+A22)(B11)
    A_M2_00 = add_lin_comb({u_idx(2,0):1}, 1, {u_idx(2,2):1}, 1)
    A_M2_01 = add_lin_comb({u_idx(2,1):1}, 1, {u_idx(2,3):1}, 1)
    A_M2_10 = add_lin_comb({u_idx(3,0):1}, 1, {u_idx(3,2):1}, 1)
    A_M2_11 = add_lin_comb({u_idx(3,1):1}, 1, {u_idx(3,3):1}, 1)
    B_M2_00 = {v_idx(0,0):1}
    B_M2_01 = {v_idx(0,1):1}
    B_M2_10 = {v_idx(1,0):1}
    B_M2_11 = {v_idx(1,1):1}
    
    # M2 contributes to C21 (+1) and C22 (-1)
    add_2x2_schoolbook([A_M2_00, A_M2_01, A_M2_10, A_M2_11],
                       [B_M2_00, B_M2_01, B_M2_10, B_M2_11],
                       [(C21_w, 1), (C22_w, -1)])
    
    # M3 = (A11)(B12-B22)
    A_M3_00 = {u_idx(0,0):1}
    A_M3_01 = {u_idx(0,1):1}
    A_M3_10 = {u_idx(1,0):1}
    A_M3_11 = {u_idx(1,1):1}
    B_M3_00 = add_lin_comb({v_idx(0,2):1}, 1, {v_idx(2,2):1}, -1)
    B_M3_01 = add_lin_comb({v_idx(0,3):1}, 1, {v_idx(2,3):1}, -1)
    B_M3_10 = add_lin_comb({v_idx(1,2):1}, 1, {v_idx(3,2):1}, -1)
    B_M3_11 = add_lin_comb({v_idx(1,3):1}, 1, {v_idx(3,3):1}, -1)
    
    # M3 contributes to C12 (+1) and C22 (+1)
    add_2x2_schoolbook([A_M3_00, A_M3_01, A_M3_10, A_M3_11],
                       [B_M3_00, B_M3_01, B_M3_10, B_M3_11],
                       [(C12_w, 1), (C22_w, 1)])
    
    # M4 = (A22)(B21-B11)
    A_M4_00 = {u_idx(2,2):1}
    A_M4_01 = {u_idx(2,3):1}
    A_M4_10 = {u_idx(3,2):1}
    A_M4_11 = {u_idx(3,3):1}
    B_M4_00 = add_lin_comb({v_idx(2,0):1}, 1, {v_idx(0,0):1}, -1)
    B_M4_01 = add_lin_comb({v_idx(2,1):1}, 1, {v_idx(0,1):1}, -1)
    B_M4_10 = add_lin_comb({v_idx(3,0):1}, 1, {v_idx(1,0):1}, -1)
    B_M4_11 = add_lin_comb({v_idx(3,1):1}, 1, {v_idx(1,1):1}, -1)
    
    # M4 contributes to C11 (+1) and C21 (+1)
    add_2x2_schoolbook([A_M4_00, A_M4_01, A_M4_10, A_M4_11],
                       [B_M4_00, B_M4_01, B_M4_10, B_M4_11],
                       [(C11_w, 1), (C21_w, 1)])
    
    # M5 = (A11+A12)(B22)
    A_M5_00 = add_lin_comb({u_idx(0,0):1}, 1, {u_idx(0,2):1}, 1)
    A_M5_01 = add_lin_comb({u_idx(0,1):1}, 1, {u_idx(0,3):1}, 1)
    A_M5_10 = add_lin_comb({u_idx(1,0):1}, 1, {u_idx(1,2):1}, 1)
    A_M5_11 = add_lin_comb({u_idx(1,1):1}, 1, {u_idx(1,3):1}, 1)
    B_M5_00 = {v_idx(2,2):1}
    B_M5_01 = {v_idx(2,3):1}
    B_M5_10 = {v_idx(3,2):1}
    B_M5_11 = {v_idx(3,3):1}
    
    # M5 contributes to C11 (-1) and C12 (+1)
    add_2x2_schoolbook([A_M5_00, A_M5_01, A_M5_10, A_M5_11],
                       [B_M5_00, B_M5_01, B_M5_10, B_M5_11],
                       [(C11_w, -1), (C12_w, 1)])
    
    # M6 = (A21-A11)(B11+B12)
    A_M6_00 = add_lin_comb({u_idx(2,0):1}, 1, {u_idx(0,0):1}, -1)
    A_M6_01 = add_lin_comb({u_idx(2,1):1}, 1, {u_idx(0,1):1}, -1)
    A_M6_10 = add_lin_comb({u_idx(3,0):1}, 1, {u_idx(1,0):1}, -1)
    A_M6_11 = add_lin_comb({u_idx(3,1):1}, 1, {u_idx(1,1):1}, -1)
    B_M6_00 = add_lin_comb({v_idx(0,0):1}, 1, {v_idx(0,2):1}, 1)
    B_M6_01 = add_lin_comb({v_idx(0,1):1}, 1, {v_idx(0,3):1}, 1)
    B_M6_10 = add_lin_comb({v_idx(1,0):1}, 1, {v_idx(1,2):1}, 1)
    B_M6_11 = add_lin_comb({v_idx(1,1):1}, 1, {v_idx(1,3):1}, 1)
    
    # M6 contributes to C22 (+1)
    add_2x2_schoolbook([A_M6_00, A_M6_01, A_M6_10, A_M6_11],
                       [B_M6_00, B_M6_01, B_M6_10, B_M6_11],
                       [(C22_w, 1)])
    
    # M7 = (A12-A22)(B21+B22)
    A_M7_00 = add_lin_comb({u_idx(0,2):1}, 1, {u_idx(2,2):1}, -1)
    A_M7_01 = add_lin_comb({u_idx(0,3):1}, 1, {u_idx(2,3):1}, -1)
    A_M7_10 = add_lin_comb({u_idx(1,2):1}, 1, {u_idx(3,2):1}, -1)
    A_M7_11 = add_lin_comb({u_idx(1,3):1}, 1, {u_idx(3,3):1}, -1)
    B_M7_00 = add_lin_comb({v_idx(2,0):1}, 1, {v_idx(2,2):1}, 1)
    B_M7_01 = add_lin_comb({v_idx(2,1):1}, 1, {v_idx(2,3):1}, 1)
    B_M7_10 = add_lin_comb({v_idx(3,0):1}, 1, {v_idx(3,2):1}, 1)
    B_M7_11 = add_lin_comb({v_idx(3,1):1}, 1, {v_idx(3,3):1}, 1)
    
    # M7 contributes to C11 (+1)
    add_2x2_schoolbook([A_M7_00, A_M7_01, A_M7_10, A_M7_11],
                       [B_M7_00, B_M7_01, B_M7_10, B_M7_11],
                       [(C11_w, 1)])
    
    # Part 2: A (4x4) x B2 (4x5) using schoolbook
    # B2 is columns 4-8 of B
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                u = {u_idx(i, j): 1}
                v = {v_idx(j, k): 1}
                w = {w_idx(k, i): 1}
                add_term(u, v, w)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
