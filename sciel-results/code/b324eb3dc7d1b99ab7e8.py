import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    """
    start_time = time.time()
    DEADLINE = 140
    
    U = np.zeros((16, 49), dtype=np.int64)
    V = np.zeros((16, 49), dtype=np.int64)
    W = np.zeros((16, 49), dtype=np.int64)
    
    def u_idx(i, j): return i * 4 + j
    def v_idx(i, j): return i * 4 + j
    def w_idx(i, j): return j * 4 + i
    
    def to_combo(indices):
        return [(i, 1) for i in indices]
        
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
    
    def get_block_indices(bi, bj, is_a):
        if is_a:
            return [
                u_idx(2*bi, 2*bj),
                u_idx(2*bi, 2*bj+1),
                u_idx(2*bi+1, 2*bj),
                u_idx(2*bi+1, 2*bj+1)
            ]
        else:
            return [
                v_idx(2*bi, 2*bj),
                v_idx(2*bi, 2*bj+1),
                v_idx(2*bi+1, 2*bj),
                v_idx(2*bi+1, 2*bj+1)
            ]
    
    A00 = to_combo(get_block_indices(0, 0, True))
    A01 = to_combo(get_block_indices(0, 1, True))
    A10 = to_combo(get_block_indices(1, 0, True))
    A11 = to_combo(get_block_indices(1, 1, True))
    
    B00 = to_combo(get_block_indices(0, 0, False))
    B01 = to_combo(get_block_indices(0, 1, False))
    B10 = to_combo(get_block_indices(1, 0, False))
    B11 = to_combo(get_block_indices(1, 1, False))
    
    terms = [
        ([add_combos(A00[i:i+1], A11[i:i+1]) for i in range(4)],
         [add_combos(B00[i:i+1], B11[i:i+1]) for i in range(4)]),
        ([add_combos(A10[i:i+1], A11[i:i+1]) for i in range(4)],
         [B00[i:i+1] for i in range(4)]),
        ([A00[i:i+1] for i in range(4)],
         [sub_combos(B01[i:i+1], B11[i:i+1]) for i in range(4)]),
        ([A11[i:i+1] for i in range(4)],
         [sub_combos(B10[i:i+1], B00[i:i+1]) for i in range(4)]),
        ([add_combos(A00[i:i+1], A01[i:i+1]) for i in range(4)],
         [B11[i:i+1] for i in range(4)]),
        ([sub_combos(A10[i:i+1], A00[i:i+1]) for i in range(4)],
         [add_combos(B00[i:i+1], B01[i:i+1]) for i in range(4)]),
        ([sub_combos(A01[i:i+1], A11[i:i+1]) for i in range(4)],
         [add_combos(B10[i:i+1], B11[i:i+1]) for i in range(4)])
    ]
    
    sub_contrib = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)]
    ]
    
    def get_w_idx(bi, bj, lr, lc):
        r = 2 * bi + lr
        c = 2 * bj + lc
        return w_idx(r, c)
        
    block_maps = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for t in range(7):
        L, R = terms[t]
        l00, l01, l10, l11 = L[0], L[1], L[2], L[3]
        r00, r01, r10, r11 = R[0], R[1], R[2], R[3]
        
        sub_prods = [
            (add_combos(l00, l11), add_combos(r00, r11)),
            (add_combos(l10, l11), r00),
            (l00, sub_combos(r01, r11)),
            (l11, sub_combos(r10, r00)),
            (add_combos(l00, l01), r11),
            (sub_combos(l10, l00), add_combos(r00, r01)),
            (sub_combos(l01, l11), add_combos(r10, r11))
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
                    w_idx_val = get_w_idx(bi, bj, lr, lc)
                    W[w_idx_val, rank_idx] += sub_coeff * block_coeff
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

