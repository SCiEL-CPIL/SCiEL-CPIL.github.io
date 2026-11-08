import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    def add_forms(f1, f2):
        d = {}
        for idx, c in f1.items():
            d[idx] = d.get(idx, 0) + c
        for idx, c in f2.items():
            d[idx] = d.get(idx, 0) + c
        return {k: v for k, v in d.items() if v != 0}
        
    def sub_forms(f1, f2):
        d = {}
        for idx, c in f1.items():
            d[idx] = d.get(idx, 0) + c
        for idx, c in f2.items():
            d[idx] = d.get(idx, 0) - c
        return {k: v for k, v in d.items() if v != 0}
        
    # Define block entries as forms
    A00 = [
        [{0: 1}, {1: 1}],
        [{4: 1}, {5: 1}]
    ]
    A01 = [
        [{2: 1}, {3: 1}],
        [{6: 1}, {7: 1}]
    ]
    A10 = [
        [{8: 1}, {9: 1}],
        [{12: 1}, {13: 1}]
    ]
    A11 = [
        [{10: 1}, {11: 1}],
        [{14: 1}, {15: 1}]
    ]
    
    B00 = [
        [{0: 1}, {1: 1}],
        [{4: 1}, {5: 1}]
    ]
    B01 = [
        [{2: 1}, {3: 1}],
        [{6: 1}, {7: 1}]
    ]
    B10 = [
        [{8: 1}, {9: 1}],
        [{12: 1}, {13: 1}]
    ]
    B11 = [
        [{10: 1}, {11: 1}],
        [{14: 1}, {15: 1}]
    ]
    
    def add_blocks(B1, B2):
        return [[add_forms(B1[r][c], B2[r][c]) for c in range(2)] for r in range(2)]
        
    def sub_blocks(B1, B2):
        return [[sub_forms(B1[r][c], B2[r][c]) for c in range(2)] for r in range(2)]

    # Outer Strassen Combinations
    L0 = add_blocks(A00, A11)
    R0 = add_blocks(B00, B11)
    
    L1 = add_blocks(A10, A11)
    R1 = B00
    
    L2 = A00
    R2 = sub_blocks(B01, B11)
    
    L3 = A11
    R3 = sub_blocks(B10, B00)
    
    L4 = add_blocks(A00, A01)
    R4 = B11
    
    L5 = sub_blocks(A10, A00)
    R5 = add_blocks(B00, B01)
    
    L6 = sub_blocks(A01, A11)
    R6 = add_blocks(B10, B11)
    
    outer_combos = [
        (L0, R0),
        (L1, R1),
        (L2, R2),
        (L3, R3),
        (L4, R4),
        (L5, R5),
        (L6, R6)
    ]
    
    # Coefficients of P in M entries
    coeff_P_in_M = {
        0: {0: 1, 3: 1, 5: -1, 6: 1}, # m00
        1: {4: 1, 5: 1},               # m01
        2: {1: 1, 2: 1},               # m10
        3: {0: 1, 1: -1, 2: 1, 6: 1}   # m11
    }
    
    # Coefficients of M blocks in C blocks
    coeff_M_in_C = {
        0: {0: 1, 3: 1, 5: -1, 6: 1}, # C00
        1: {4: 1, 5: 1},               # C01
        2: {1: 1, 2: 1},               # C10
        3: {0: 1, 1: -1, 2: 1, 6: 1}   # C11
    }
    
    # Now iterate over t
    for t in range(7):
        if time.time() > deadline - 10:
            break
            
        L_t, R_t = outer_combos[t]
        
        l00, l01 = L_t[0][0], L_t[0][1]
        l10, l11 = L_t[1][0], L_t[1][1]
        
        r00, r01 = R_t[0][0], R_t[0][1]
        r10, r11 = R_t[1][0], R_t[1][1]
        
        # Compute the 7 P forms (left and right)
        P_forms = []
        
        # P0
        P_forms.append((add_forms(l00, l11), add_forms(r00, r11)))
        # P1
        P_forms.append((add_forms(l10, l11), r00))
        # P2
        P_forms.append((l00, sub_forms(r01, r11)))
        # P3
        P_forms.append((l11, sub_forms(r10, r00)))
        # P4
        P_forms.append((add_forms(l00, l01), r11))
        # P5
        P_forms.append((sub_forms(l10, l00), add_forms(r00, r01)))
        # P6
        P_forms.append((sub_forms(l01, l11), add_forms(r10, r11)))
        
        for k in range(7):
            rank_idx = t * 7 + k
            left_form, right_form = P_forms[k]
            
            # Fill U and V
            for idx, coeff in left_form.items():
                U[idx, rank_idx] = coeff
            for idx, coeff in right_form.items():
                V[idx, rank_idx] = coeff
            
            # Fill W
            for c_block_idx in range(4):
                b_i = c_block_idx // 2
                b_j = c_block_idx % 2
                
                coeff_M = coeff_M_in_C[c_block_idx].get(t, 0)
                if coeff_M == 0:
                    continue
                
                for m_entry_idx in range(4):
                    u = m_entry_idx // 2
                    v = m_entry_idx % 2
                    
                    coeff_P = coeff_P_in_M[m_entry_idx].get(k, 0)
                    if coeff_P == 0:
                        continue
                    
                    total_coeff = coeff_M * coeff_P
                    if total_coeff != 0:
                        i_global = 2 * b_i + u
                        j_global = 2 * b_j + v
                        w_idx = j_global * n + i_global
                        W[w_idx, rank_idx] = total_coeff
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
