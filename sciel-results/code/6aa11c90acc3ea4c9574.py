import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 0
    U = []
    V = []
    W = []

    def add_standard(i, j, k):
        nonlocal rank
        U.append([0] * rank + [1])
        V.append([0] * rank + [1])
        W.append([0] * rank + [1])
        rank += 1
        # Wait, this appends to lists. 
        # Better to pre-allocate or build lists and convert.
        # Since rank is not known beforehand, we build lists.
        # But U, V, W have fixed row counts.
        # U has n*m rows. V has m*p rows. W has p*n rows.
        # Each row has 'rank' columns.
        # This dynamic appending is inefficient and hard to manage with fixed rows.
        
    # Better approach:
    # Determine total rank first? No, it depends on the decomposition.
    # We can build the columns as lists of tuples (row, value).
    # Then flatten.
    
    u_cols = []
    v_cols = []
    w_cols = []
    
    def add_standard(i, j, k):
        u_cols.append([(i * m + j, 1)])
        v_cols.append([(j * p + k, 1)])
        w_cols.append([(k * n + i, 1)])
        
    def add_strassen(i1, i2, j1, j2, k1, k2):
        # Strassen's 7 products for 2x2 matrices
        # A indices: (i1,j1), (i1,j2), (i2,j1), (i2,j2)
        # B indices: (j1,k1), (j1,k2), (j2,k1), (j2,k2)
        # C indices: (i1,k1), (i1,k2), (i2,k1), (i2,k2)
        
        a00 = (i1, j1)
        a01 = (i1, j2)
        a10 = (i2, j1)
        a11 = (i2, j2)
        
        b00 = (j1, k1)
        b01 = (j1, k2)
        b10 = (j2, k1)
        b11 = (j2, k2)
        
        c00 = (i1, k1)
        c01 = (i1, k2)
        c10 = (i2, k1)
        c11 = (i2, k2)
        
        # M1 = (A00 + A11)(B00 + B11)
        u1 = [(i1 * m + j1, 1), (i2 * m + j2, 1)]
        v1 = [(j1 * p + k1, 1), (j2 * p + k2, 1)]
        w1 = [(k1 * n + i1, 1), (k2 * n + i1, -1), (k1 * n + i2, -1), (k2 * n + i2, 1)]
        u_cols.append(u1)
        v_cols.append(v1)
        w_cols.append(w1)
        
        # M2 = (A10 + A11)B00
        u2 = [(i2 * m + j1, 1), (i2 * m + j2, 1)]
        v2 = [(j1 * p + k1, 1)]
        w2 = [(k1 * n + i2, 1), (k2 * n + i2, 1)] # Wait, C10 and C11?
        # C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
        # So M2 contributes 1 to C10 and -1 to C11.
        w2 = [(k1 * n + i2, 1), (k2 * n + i2, -1)]
        u_cols.append(u2)
        v_cols.append(v2)
        w_cols.append(w2)
        
        # M3 = A00(B01 - B11)
        u3 = [(i1 * m + j1, 1)]
        v3 = [(j1 * p + k2, 1), (j2 * p + k2, -1)]
        w3 = [(k2 * n + i1, 1), (k2 * n + i2, 1)]
        # C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
        # So M3 contributes 1 to C01 and 1 to C11.
        w3 = [(k2 * n + i1, 1), (k2 * n + i2, 1)]
        u_cols.append(u3)
        v_cols.append(v3)
        w_cols.append(w3)
        
        # M4 = A11(B10 - B00)
        u4 = [(i2 * m + j2, 1)]
        v4 = [(j2 * p + k1, 1), (j1 * p + k1, -1)]
        w4 = [(k1 * n + i1, 1), (k1 * n + i2, 1)]
        # C00 = M1 + M4 - M5 + M7. C10 = M2 + M4.
        # So M4 contributes 1 to C00 and 1 to C10.
        w4 = [(k1 * n + i1, 1), (k1 * n + i2, 1)]
        u_cols.append(u4)
        v_cols.append(v4)
        w_cols.append(w4)
        
        # M5 = (A00 + A01)B11
        u5 = [(i1 * m + j1, 1), (i1 * m + j2, 1)]
        v5 = [(j2 * p + k2, 1)]
        w5 = [(k2 * n + i1, 1), (k2 * n + i2, -1)]
        # C00 = M1 + M4 - M5 + M7. C01 = M3 + M5.
        # So M5 contributes -1 to C00 and 1 to C01.
        w5 = [(k1 * n + i1, -1), (k2 * n + i1, 1)]
        u_cols.append(u5)
        v_cols.append(v5)
        w_cols.append(w5)
        
        # M6 = (A10 - A00)(B00 + B01)
        u6 = [(i2 * m + j1, 1), (i1 * m + j1, -1)]
        v6 = [(j1 * p + k1, 1), (j1 * p + k2, 1)]
        w6 = [(k2 * n + i2, 1)]
        # C11 = M1 - M2 + M3 + M6.
        # So M6 contributes 1 to C11.
        w6 = [(k2 * n + i2, 1)]
        u_cols.append(u6)
        v_cols.append(v6)
        w_cols.append(w6)
        
        # M7 = (A01 - A11)(B10 + B11)
        u7 = [(i1 * m + j2, 1), (i2 * m + j2, -1)]
        v7 = [(j2 * p + k1, 1), (j2 * p + k2, 1)]
        w7 = [(k1 * n + i1, 1)]
        # C00 = M1 + M4 - M5 + M7.
        # So M7 contributes 1 to C00.
        w7 = [(k1 * n + i1, 1)]
        u_cols.append(u7)
        v_cols.append(v7)
        w_cols.append(w7)

    # 1. i=0, j in 0..3, k in 0..6
    for j in range(4):
        for k in range(7):
            add_standard(0, j, k)
            
    # 2. i in 1,2, j in 0,3, k in 0..6
    # Split k into pairs (0,1), (2,3), (4,5), (6)
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    for k1, k2 in k_pairs:
        add_strassen(1, 2, 0, 3, k1, k2)
    # Single k=6
    for j in [0, 3]:
        add_standard(1, j, 6)
        add_standard(2, j, 6)
        
    # 3. i in 1,2, j in 1,2, k in 0..6
    for k1, k2 in k_pairs:
        add_strassen(1, 2, 1, 2, k1, k2)
    # Single k=6
    for j in [1, 2]:
        add_standard(1, j, 6)
        add_standard(2, j, 6)

    rank = len(u_cols)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for row, val in u_cols[c]:
            U[row][c] = val
        for row, val in v_cols[c]:
            V[row][c] = val
        for row, val in w_cols[c]:
            W[row][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
