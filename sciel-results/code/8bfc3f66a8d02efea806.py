import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the matrices dynamically
    # Start with a list of tuples (u_vec, v_vec, w_vec)
    # u_vec is dict {global_idx: val}
    # v_vec is dict {global_idx: val}
    # w_vec is dict {global_idx: val}
    
    terms = []
    
    def add_rank1(u, v, w):
        # u, v, w are dicts {idx: val}
        terms.append((u, v, w))
        
    def add_strassen_222(i_off, j_off, k_off):
        # Local indices 0,1 for i, j, k
        # A indices: (i_off+i)*m + (j_off+j)
        # B indices: (j_off+j)*p + (k_off+k)
        # C indices: (k_off+k)*n + (i_off+i)
        
        # Strassen 7 terms
        # M1 = (A00+A11)(B00+B11)
        u1 = {
            (i_off+0)*m + (j_off+0): 1,
            (i_off+1)*m + (j_off+1): 1
        }
        v1 = {
            (j_off+0)*p + (k_off+0): 1,
            (j_off+0)*p + (k_off+1): 1
        }
        # C00 = M1 + ...
        # C11 = M1 - ...
        # w1 contributes to C00 and C11
        w1 = {
            (k_off+0)*n + (i_off+0): 1,
            (k_off+1)*n + (i_off+1): 1
        }
        add_rank1(u1, v1, w1)
        
        # M2 = (A10+A11)B00
        u2 = {
            (i_off+1)*m + (j_off+0): 1,
            (i_off+1)*m + (j_off+1): 1
        }
        v2 = {
            (j_off+0)*p + (k_off+0): 1
        }
        # C10 = M2 + ...
        # C11 = M1 - M2 + ...
        w2 = {
            (k_off+0)*n + (i_off+1): 1,
            (k_off+1)*n + (i_off+1): -1
        }
        add_rank1(u2, v2, w2)
        
        # M3 = A00(B01-B11)
        u3 = {
            (i_off+0)*m + (j_off+0): 1
        }
        v3 = {
            (j_off+0)*p + (k_off+1): 1,
            (j_off+1)*p + (k_off+1): -1
        }
        # C01 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        w3 = {
            (k_off+1)*n + (i_off+0): 1,
            (k_off+1)*n + (i_off+1): 1
        }
        add_rank1(u3, v3, w3)
        
        # M4 = A11(B10-B00)
        u4 = {
            (i_off+1)*m + (j_off+1): 1
        }
        v4 = {
            (j_off+1)*p + (k_off+0): 1,
            (j_off+0)*p + (k_off+0): -1
        }
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        w4 = {
            (k_off+0)*n + (i_off+0): 1,
            (k_off+0)*n + (i_off+1): 1
        }
        add_rank1(u4, v4, w4)
        
        # M5 = (A00+A01)B11
        u5 = {
            (i_off+0)*m + (j_off+0): 1,
            (i_off+0)*m + (j_off+1): 1
        }
        v5 = {
            (j_off+1)*p + (k_off+1): 1
        }
        # C01 = M3 + M5
        # C00 = M1 + M4 - M5 + M7
        w5 = {
            (k_off+1)*n + (i_off+0): 1,
            (k_off+0)*n + (i_off+0): -1
        }
        add_rank1(u5, v5, w5)
        
        # M6 = (A10-A00)(B00+B01)
        u6 = {
            (i_off+1)*m + (j_off+0): 1,
            (i_off+0)*m + (j_off+0): -1
        }
        v6 = {
            (j_off+0)*p + (k_off+0): 1,
            (j_off+0)*p + (k_off+1): 1
        }
        # C11 = M1 - M2 + M3 + M6
        w6 = {
            (k_off+1)*n + (i_off+1): 1
        }
        add_rank1(u6, v6, w6)
        
        # M7 = (A01-A11)(B10+B11)
        u7 = {
            (i_off+0)*m + (j_off+1): 1,
            (i_off+1)*m + (j_off+1): -1
        }
        v7 = {
            (j_off+1)*p + (k_off+0): 1,
            (j_off+1)*p + (k_off+1): 1
        }
        # C00 = M1 + M4 - M5 + M7
        w7 = {
            (k_off+0)*n + (i_off+0): 1
        }
        add_rank1(u7, v7, w7)

    def add_schoolbook_221(i_off, j_off, k_off):
        # 2x2 A, 2x1 B, 2x1 C
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        # 4 terms
        add_rank1({(i_off+0)*m + (j_off+0): 1}, {(j_off+0)*p + k_off: 1}, {(k_off)*n + (i_off+0): 1})
        add_rank1({(i_off+0)*m + (j_off+1): 1}, {(j_off+1)*p + k_off: 1}, {(k_off)*n + (i_off+0): 1})
        add_rank1({(i_off+1)*m + (j_off+0): 1}, {(j_off+0)*p + k_off: 1}, {(k_off)*n + (i_off+1): 1})
        add_rank1({(i_off+1)*m + (j_off+1): 1}, {(j_off+1)*p + k_off: 1}, {(k_off)*n + (i_off+1): 1})

    def add_schoolbook_212(i_off, j_off, k_off):
        # 2x1 A, 1x2 B, 2x2 C
        # C00 = A00 B00
        # C01 = A00 B01
        # C10 = A10 B00
        # C11 = A10 B01
        add_rank1({(i_off+0)*m + (j_off): 1}, {(j_off)*p + (k_off+0): 1}, {(k_off+0)*n + (i_off+0): 1})
        add_rank1({(i_off+0)*m + (j_off): 1}, {(j_off)*p + (k_off+1): 1}, {(k_off+1)*n + (i_off+0): 1})
        add_rank1({(i_off+1)*m + (j_off): 1}, {(j_off)*p + (k_off+0): 1}, {(k_off+0)*n + (i_off+1): 1})
        add_rank1({(i_off+1)*m + (j_off): 1}, {(j_off)*p + (k_off+1): 1}, {(k_off+1)*n + (i_off+1): 1})

    def add_schoolbook_211(i_off, j_off, k_off):
        # 2x1 A, 1x1 B, 2x1 C
        # C00 = A00 B00
        # C10 = A10 B00
        add_rank1({(i_off+0)*m + (j_off): 1}, {(j_off)*p + (k_off): 1}, {(k_off)*n + (i_off+0): 1})
        add_rank1({(i_off+1)*m + (j_off): 1}, {(j_off)*p + (k_off): 1}, {(k_off)*n + (i_off+1): 1})

    # Call the blocks
    add_strassen_222(0, 0, 0)
    add_strassen_222(0, 0, 2)
    add_schoolbook_221(0, 0, 4)
    add_strassen_222(0, 2, 0)
    add_strassen_222(0, 2, 2)
    add_schoolbook_221(0, 2, 4)
    add_schoolbook_212(0, 4, 0)
    add_schoolbook_212(0, 4, 2)
    add_schoolbook_211(0, 4, 4)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][c] = val
        for idx, val in v.items():
            V[idx][c] = val
        for idx, val in w.items():
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
