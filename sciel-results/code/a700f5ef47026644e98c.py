import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    start_time = time.time()
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((16, rank), dtype=np.int64)
    W = np.zeros((16, rank), dtype=np.int64)
    
    def u_idx(i, j): return i * 4 + j
    def v_idx(i, j): return i * 4 + j
    def w_idx(i, j): return j * 4 + i  # Trace convention: W[k*n+i, r]
    
    # Strassen for 2x2: M0..M6, Z00=M0+M3-M4+M6, Z01=M2+M4, Z10=M1+M3, Z11=M0-M1+M2+M5
    # Block maps for 7 Strassen terms (P0..P6) to C blocks
    # P0: C00(+1), C11(+1); P1: C10(+1), C11(-1); P2: C01(+1), C11(+1)
    # P3: C00(+1), C10(+1); P4: C00(-1), C01(+1); P5: C11(+1); P6: C00(+1)
    block_maps = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    # Sub-contributions of Mk to local Z (0=Z00, 1=Z01, 2=Z10, 3=Z11)
    sub_contribs = [
        [(0, 1), (3, 1)],   # M0
        [(2, 1), (3, -1)],  # M1
        [(1, 1), (3, 1)],   # M2
        [(0, 1), (2, 1)],   # M3
        [(0, -1), (1, 1)],  # M4
        [(3, 1)],           # M5
        [(0, 1)]            # M6
    ]
    
    # A/B block element indices (global)
    # A00: [0,1,4,5], A01: [2,3,6,7], A10: [8,9,12,13], A11: [10,11,14,15]
    # B same indexing
    
    def ce(idx, coeff=1):
        return [(idx, coeff)]
    
    def add_c(c1, c2):
        d = {}
        for i, v in c1: d[i] = d.get(i, 0) + v
        for i, v in c2: d[i] = d.get(i, 0) + v
        return [(k, v) for k, v in d.items() if v != 0]
    
    def sub_c(c1, c2):
        d = {}
        for i, v in c1: d[i] = d.get(i, 0) + v
        for i, v in c2: d[i] = d.get(i, 0) - v
        return [(k, v) for k, v in d.items() if v != 0]
    
    def block_add(M1, M2):
        return [add_c(M1[i], M2[i]) for i in range(4)]
    
    def block_sub(M1, M2):
        return [sub_c(M1[i], M2[i]) for i in range(4)]
    
    # Define block elements as lists of 4 combos (00, 01, 10, 11)
    A00 = [[(0,1)], [(1,1)], [(4,1)], [(5,1)]]
    A01 = [[(2,1)], [(3,1)], [(6,1)], [(7,1)]]
    A10 = [[(8,1)], [(9,1)], [(12,1)], [(13,1)]]
    A11 = [[(10,1)], [(11,1)], [(14,1)], [(15,1)]]
    B00 = [[(0,1)], [(1,1)], [(4,1)], [(5,1)]]
    B01 = [[(2,1)], [(3,1)], [(6,1)], [(7,1)]]
    B10 = [[(8,1)], [(9,1)], [(12,1)], [(13,1)]]
    B11 = [[(10,1)], [(11,1)], [(14,1)], [(15,1)]]
    
    # 7 block-level Strassen terms: (X, Y) where X,Y are 2x2 matrices of combos
    X_list = [
        block_add(A00, A11),   # T0: A00+A11
        block_add(A10, A11),   # T1: A10+A11
        A00,                   # T2: A00
        A11,                   # T3: A11
        block_add(A00, A01),   # T4: A00+A01
        block_sub(A10, A00),   # T5: A10-A00
        block_sub(A01, A11)    # T6: A01-A11
    ]
    Y_list = [
        block_add(B00, B11),   # T0: B00+B11
        B00,                   # T1: B00
        block_sub(B01, B11),   # T2: B01-B11
        block_sub(B10, B00),   # T3: B10-B00
        B11,                   # T4: B11
        block_add(B00, B01),   # T5: B00+B01
        block_add(B10, B11)    # T6: B10+B11
    ]
    
    for t in range(7):
        X = X_list[t]
        Y = Y_list[t]
        x00, x01, x10, x11 = X[0], X[1], X[2], X[3]
        y00, y01, y10, y11 = Y[0], Y[1], Y[2], Y[3]
        
        ms = [
            (add_c(x00, x11), add_c(y00, y11)),   # M0
            (add_c(x10, x11), y00),                # M1
            (x00, sub_c(y01, y11)),                # M2
            (x11, sub_c(y10, y00)),                # M3
            (add_c(x00, x01), y11),                # M4
            (sub_c(x10, x00), add_c(y00, y01)),    # M5
            (sub_c(x01, x11), add_c(y10, y11))     # M6
        ]
        
        for k in range(7):
            rank_idx = t * 7 + k
            l_combo, r_combo = ms[k]
            
            for idx, coeff in l_combo:
                U[idx, rank_idx] += coeff
            for idx, coeff in r_combo:
                V[idx, rank_idx] += coeff
            
            for z_idx, coeff in sub_contribs[k]:
                lr = z_idx // 2
                lc = z_idx % 2
                for c_br, c_bc, c_coeff in block_maps[t]:
                    gr = 2 * c_br + lr
                    gc = 2 * c_bc + lc
                    W[w_idx(gr, gc), rank_idx] += coeff * c_coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
