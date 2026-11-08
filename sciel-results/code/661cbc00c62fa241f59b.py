import numpy as np

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((16, rank), dtype=np.int64)
    W = np.zeros((16, rank), dtype=np.int64)
    
    def u_idx(i, j): return i * 4 + j
    def v_idx(i, j): return i * 4 + j
    def w_idx(i, j): return j * 4 + i  # Trace convention: W[k*n+i, r]

    def get_a_indices(bi, bj):
        return [
            u_idx(2*bi, 2*bj),
            u_idx(2*bi, 2*bj+1),
            u_idx(2*bi+1, 2*bj),
            u_idx(2*bi+1, 2*bj+1)
        ]
        
    def get_b_indices(bi, bj):
        return [
            v_idx(2*bi, 2*bj),
            v_idx(2*bi, 2*bj+1),
            v_idx(2*bi+1, 2*bj),
            v_idx(2*bi+1, 2*bj+1)
        ]
    
    def add_combos(c1, c2):
        d = {}
        for idx, c in c1:
            d[idx] = d.get(idx, 0) + c
        for idx, c in c2:
            d[idx] = d.get(idx, 0) + c
        return [(k, v) for k, v in d.items() if v != 0]
        
    def sub_combos(c1, c2):
        d = {}
        for idx, c in c1:
            d[idx] = d.get(idx, 0) + c
        for idx, c in c2:
            d[idx] = d.get(idx, 0) - c
        return [(k, v) for k, v in d.items() if v != 0]
    
    A00_idx = get_a_indices(0, 0)
    A01_idx = get_a_indices(0, 1)
    A10_idx = get_a_indices(1, 0)
    A11_idx = get_a_indices(1, 1)
    
    B00_idx = get_b_indices(0, 0)
    B01_idx = get_b_indices(0, 1)
    B10_idx = get_b_indices(1, 0)
    B11_idx = get_b_indices(1, 1)
    
    def to_combo(indices):
        return [(i, 1) for i in indices]
        
    A00 = to_combo(A00_idx)
    A01 = to_combo(A01_idx)
    A10 = to_combo(A10_idx)
    A11 = to_combo(A11_idx)
    
    B00 = to_combo(B00_idx)
    B01 = to_combo(B01_idx)
    B10 = to_combo(B10_idx)
    B11 = to_combo(B11_idx)
    
    L_terms = []
    R_terms = []
    
    # T0: (A00+A11) * (B00+B11)
    L0 = [
        add_combos(A00[0:1], A11[0:1]),
        add_combos(A00[1:2], A11[1:2]),
        add_combos(A00[2:3], A11[2:3]),
        add_combos(A00[3:4], A11[3:4])
    ]
    R0 = [
        add_combos(B00[0:1], B11[0:1]),
        add_combos(B00[1:2], B11[1:2]),
        add_combos(B00[2:3], B11[2:3]),
        add_combos(B00[3:4], B11[3:4])
    ]
    L_terms.append(L0)
    R_terms.append(R0)
    
    # T1: (A10+A11) * B00
    L1 = [
        add_combos(A10[0:1], A11[0:1]),
        add_combos(A10[1:2], A11[1:2]),
        add_combos(A10[2:3], A11[2:3]),
        add_combos(A10[3:4], A11[3:4])
    ]
    R1 = [
        B00[0:1], B00[1:2], B00[2:3], B00[3:4]
    ]
    L_terms.append(L1)
    R_terms.append(R1)
    
    # T2: A00 * (B01-B11)
    L2 = [
        A00[0:1], A00[1:2], A00[2:3], A00[3:4]
    ]
    R2 = [
        sub_combos(B01[0:1], B11[0:1]),
        sub_combos(B01[1:2], B11[1:2]),
        sub_combos(B01[2:3], B11[2:3]),
        sub_combos(B01[3:4], B11[3:4])
    ]
    L_terms.append(L2)
    R_terms.append(R2)
    
    # T3: A11 * (B10-B00)
    L3 = [
        A11[0:1], A11[1:2], A11[2:3], A11[3:4]
    ]
    R3 = [
        sub_combos(B10[0:1], B00[0:1]),
        sub_combos(B10[1:2], B00[1:2]),
        sub_combos(B10[2:3], B00[2:3]),
        sub_combos(B10[3:4], B00[3:4])
    ]
    L_terms.append(L3)
    R_terms.append(R3)
    
    # T4: (A00+A01) * B11
    L4 = [
        add_combos(A00[0:1], A01[0:1]),
        add_combos(A00[1:2], A01[1:2]),
        add_combos(A00[2:3], A01[2:3]),
        add_combos(A00[3:4], A01[3:4])
    ]
    R4 = [
        B11[0:1], B11[1:2], B11[2:3], B11[3:4]
    ]
    L_terms.append(L4)
    R_terms.append(R4)
    
    # T5: (A10-A00) * (B00+B01)
    L5 = [
        sub_combos(A10[0:1], A00[0:1]),
        sub_combos(A10[1:2], A00[1:2]),
        sub_combos(A10[2:3], A00[2:3]),
        sub_combos(A10[3:4], A00[3:4])
    ]
    R5 = [
        add_combos(B00[0:1], B01[0:1]),
        add_combos(B00[1:2], B01[1:2]),
        add_combos(B00[2:3], B01[2:3]),
        add_combos(B00[3:4], B01[3:4])
    ]
    L_terms.append(L5)
    R_terms.append(R5)
    
    # T6: (A01-A11) * (B10+B11)
    L6 = [
        sub_combos(A01[0:1], A11[0:1]),
        sub_combos(A01[1:2], A11[1:2]),
        sub_combos(A01[2:3], A11[2:3]),
        sub_combos(A01[3:4], A11[3:4])
    ]
    R6 = [
        add_combos(B10[0:1], B11[0:1]),
        add_combos(B10[1:2], B11[1:2]),
        add_combos(B10[2:3], B11[2:3]),
        add_combos(B10[3:4], B11[3:4])
    ]
    L_terms.append(L6)
    R_terms.append(R6)
    
    # Sub-contributions of Mk to the 2x2 result P (local indices 00, 01, 10, 11)
    sub_contrib = [
        [(0, 1), (3, 1)],      # M1: P00, P11
        [(2, 1), (3, -1)],     # M2: P10, P11
        [(1, 1), (3, 1)],      # M3: P01, P11
        [(0, 1), (2, 1)],      # M4: P00, P10
        [(0, -1), (1, 1)],     # M5: P00, P01
        [(3, 1)],              # M6: P11
        [(0, 1)]               # M7: P00
    ]
    
    def get_global_c_indices(block_bi, block_bj, lr, lc):
        r = 2 * block_bi + lr
        c = 2 * block_bj + lc
        return w_idx(r, c)
        
    block_maps = [
        [(0, 0, 1), (1, 1, 1)],  # t=0
        [(1, 0, 1), (1, 1, -1)], # t=1
        [(0, 1, 1), (1, 1, 1)],  # t=2
        [(0, 0, 1), (1, 0, 1)],  # t=3
        [(0, 0, -1), (0, 1, 1)], # t=4
        [(1, 1, 1)],             # t=5
        [(0, 0, 1)]              # t=6
    ]
    
    for t in range(7):
        L = L_terms[t]
        R = R_terms[t]
        
        l00, l01, l10, l11 = L[0], L[1], L[2], L[3]
        r00, r01, r10, r11 = R[0], R[1], R[2], R[3]
        
        x1 = add_combos(l00, l11)
        y1 = add_combos(r00, r11)
        
        x2 = add_combos(l10, l11)
        y2 = r00
        
        x3 = l00
        y3 = sub_combos(r01, r11)
        
        x4 = l11
        y4 = sub_combos(r10, r00)
        
        x5 = add_combos(l00, l01)
        y5 = r11
        
        x6 = sub_combos(l10, l00)
        y6 = add_combos(r00, r01)
        
        x7 = sub_combos(l01, l11)
        y7 = add_combos(r10, r11)
        
        sub_prods = [
            (x1, y1),
            (x2, y2),
            (x3, y3),
            (x4, y4),
            (x5, y5),
            (x6, y6),
            (x7, y7)
        ]
        
        for k in range(7):
            rank_idx = t * 7 + k
            x_k, y_k = sub_prods[k]
            
            for idx, coeff in x_k:
                U[idx, rank_idx] += coeff
            for idx, coeff in y_k:
                V[idx, rank_idx] += coeff
                
            for (local_idx, sub_coeff) in sub_contrib[k]:
                lr = local_idx // 2
                lc = local_idx % 2
                
                for (bi, bj, block_coeff) in block_maps[t]:
                    w_idx_val = get_global_c_indices(bi, bj, lr, lc)
                    W[w_idx_val, rank_idx] += sub_coeff * block_coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
