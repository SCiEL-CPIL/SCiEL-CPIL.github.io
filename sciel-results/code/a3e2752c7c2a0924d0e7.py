import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # We will build the decomposition manually.
    # Split indices:
    # i in {0, 1}
    # j in {0, 1, 2, 3, 4}
    # k in {0, 1, 2, 3, 4, 5}
    
    # Block 1: j in {0, 1}, k in {0, 1, 2, 3, 4, 5}
    # A1 is 2x2, B1 is 2x6.
    # Use 3 Strassen blocks for k pairs (0,1), (2,3), (4,5).
    
    # Block 2: j in {2, 3}, k in {0, 1, 2, 3, 4, 5}
    # A2 is 2x2, B2 is 2x6.
    # Use 3 Strassen blocks.
    
    # Block 3: j in {4}, k in {0, 1, 2, 3, 4, 5}
    # A3 is 2x1, B3 is 1x6.
    # Use 6 standard multiplications.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(i_base, j_base, k_base):
        # Adds a 2x2 by 2x2 Strassen block
        # i_base: 0 or 1 (row of A)
        # j_base: 0, 1, 2, 3 (col of A, offset by 0 or 2)
        # k_base: 0, 2, 4 (col of B, offset by 0, 2, 4)
        
        # A indices: (i_base, j_base), (i_base, j_base+1)
        #             (i_base+1, j_base), (i_base+1, j_base+1)
        # B indices: (j_base, k_base), (j_base, k_base+1)
        #             (j_base+1, k_base), (j_base+1, k_base+1)
        
        # U vectors for A
        # V vectors for B
        # W vectors for C
        
        # M1 = (a+d)(e+g)
        # U: a=1, d=1; V: e=1, g=1; W: c00=1, c11=1
        # M2 = (c+d)g
        # U: c=1, d=1; V: g=1; W: c10=1, c11=1
        # M3 = a(f-h)
        # U: a=1; V: f=1, h=-1; W: c01=1
        # M4 = d(h-f)
        # U: d=1; V: h=1, f=-1; W: c01=1, c11=1
        # M5 = (a+b)e
        # U: a=1, b=1; V: e=1; W: c00=1, c10=1
        # M6 = (c+d)(h-f)
        # U: c=1, d=1; V: h=1, f=-1; W: c11=1
        # M7 = (a-b)(e+g)
        # U: a=1, b=-1; V: e=1, g=1; W: c00=1, c10=1
        
        # Note: C indices are (i, k)
        # C00 corresponds to (i_base, k_base)
        # C01 corresponds to (i_base, k_base+1)
        # C10 corresponds to (i_base+1, k_base)
        # C11 corresponds to (i_base+1, k_base+1)
        
        # We need to map these to global indices.
        # U index: i * m + j
        # V index: j * p + k
        # W index: k * n + i
        
        terms = []
        
        # M1
        u = {i_base * m + j_base: 1, (i_base+1) * m + (j_base+1): 1}
        v = {j_base * p + k_base: 1, (j_base+1) * p + (k_base+1): 1}
        w = {k_base * n + i_base: 1, (k_base+1) * n + (i_base+1): 1}
        terms.append((u, v, w))
        
        # M2
        u = {i_base * m + (j_base+1): 1, (i_base+1) * m + (j_base+1): 1} # c, d
        v = {(j_base+1) * p + k_base: 1} # g
        w = {(k_base) * n + (i_base+1): 1, (k_base+1) * n + (i_base+1): 1} # c10, c11
        terms.append((u, v, w))
        
        # M3
        u = {i_base * m + j_base: 1} # a
        v = {j_base * p + (k_base+1): 1, (j_base+1) * p + (k_base+1): -1} # f, -h
        w = {k_base * n + i_base: 1, (k_base+1) * n + i_base: 1} # c00? No.
        # C01 = M3 + M4. So W for M3 is C01.
        # C01 is (i_base, k_base+1).
        w = {(k_base+1) * n + i_base: 1}
        terms.append((u, v, w))
        
        # M4
        u = {(i_base+1) * m + (j_base+1): 1} # d
        v = {(j_base+1) * p + (k_base+1): 1, j_base * p + (k_base+1): -1} # h, -f
        w = {(k_base+1) * n + (i_base+1): 1} # C11 part
        # Wait, C11 = M1 - M3 + M6 - M5.
        # C01 = M3 + M4.
        # So M4 contributes to C01 and C11?
        # Standard Strassen:
        # C00 = M1 - M4 + M7 - M2
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M3 + M6 - M5
        
        # Let's re-verify M4.
        # M4 = d(h-f).
        # Contribution to C01: +1.
        # Contribution to C11: +1.
        w = {(k_base+1) * n + i_base: 1, (k_base+1) * n + (i_base+1): 1}
        terms.append((u, v, w))
        
        # M5
        u = {i_base * m + j_base: 1, i_base * m + (j_base+1): 1} # a, b
        v = {j_base * p + k_base: 1} # e
        w = {k_base * n + i_base: 1, k_base * n + (i_base+1): 1} # C00, C10
        terms.append((u, v, w))
        
        # M6
        u = {i_base * m + (j_base+1): 1, (i_base+1) * m + (j_base+1): 1} # c, d
        v = {(j_base+1) * p + (k_base+1): 1, j_base * p + (k_base+1): -1} # h, -f
        w = {(k_base+1) * n + (i_base+1): 1} # C11 part
        terms.append((u, v, w))
        
        # M7
        u = {i_base * m + j_base: 1, i_base * m + (j_base+1): -1} # a, -b
        v = {j_base * p + k_base: 1, (j_base+1) * p + k_base: 1} # e, g
        w = {k_base * n + i_base: 1, k_base * n + (i_base+1): 1} # C00, C10
        terms.append((u, v, w))
        
        return terms

    def add_standard_block(i_base, j_base, k_base):
        # 2x1 by 1x1
        # A: (i_base, j_base), (i_base+1, j_base)
        # B: (j_base, k_base)
        # C: (i_base, k_base), (i_base+1, k_base)
        terms = []
        # M1 = a * e
        u = {i_base * m + j_base: 1}
        v = {j_base * p + k_base: 1}
        w = {k_base * n + i_base: 1}
        terms.append((u, v, w))
        
        # M2 = c * e
        u = {(i_base+1) * m + j_base: 1}
        v = {j_base * p + k_base: 1}
        w = {k_base * n + (i_base+1): 1}
        terms.append((u, v, w))
        
        return terms

    # Block 1: j_base=0, i_base=0
    for k_off in [0, 2, 4]:
        terms = add_strassen_block(0, 0, k_off)
        for u, v, w in terms:
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)

    # Block 2: j_base=2, i_base=0
    for k_off in [0, 2, 4]:
        terms = add_strassen_block(0, 2, k_off)
        for u, v, w in terms:
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)

    # Block 3: j_base=4, i_base=0
    for k_off in range(6):
        terms = add_standard_block(0, 4, k_off)
        for u, v, w in terms:
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)

    rank = len(U_cols)
    
    # Convert sparse dicts to dense lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx in range(rank):
        u_dict = U_cols[col_idx]
        v_dict = V_cols[col_idx]
        w_dict = W_cols[col_idx]
        
        for idx, val in u_dict.items():
            U[idx][col_idx] = val
        for idx, val in v_dict.items():
            V[idx][col_idx] = val
        for idx, val in w_dict.items():
            W[idx][col_idx] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
