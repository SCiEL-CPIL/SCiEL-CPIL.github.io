"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    p1, p2 = 4, 3
    
    rank = 97
    
    U_cols = [[] for _ in range(n * m)]       # 16 rows
    V_cols = [[] for _ in range(m * p)]       # 28 rows
    W_cols = [[] for _ in range(p * n)]       # 28 rows
    
    strassen_2x2 = [
        {'A': [(0,1),(1,1)], 'B': [(0,1),(1,1)], 'C': [(0,1),(3,-1)]},
        {'A': [(2,1),(3,1)], 'B': [(0,1)], 'C': [(2,1)]},
        {'A': [(0,1)], 'B': [(2,1),(3,-1)], 'C': [(1,1)]},
        {'A': [(3,1)], 'B': [(1,1),(0,-1)], 'C': [(0,1),(2,1)]},
        {'A': [(0,1),(1,1)], 'B': [(3,1)], 'C': [(0,-1),(1,1)]},
        {'A': [(2,1),(0,-1)], 'B': [(0,1),(1,1)], 'C': [(3,1)]},
        {'A': [(1,1),(3,-1)], 'B': [(2,1),(3,1)], 'C': [(0,1),(3,1)]},
    ]
    
    strassen_4x4 = [
        {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'C': [(0,0,1), (1,1,-1)]},
        {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1)]},
        {'A': [(0,0,1)], 'B': [(1,0,1), (1,1,-1)], 'C': [(0,1,1)]},
        {'A': [(1,1,1)], 'B': [(0,1,1), (0,0,-1)], 'C': [(0,0,1), (1,0,1)]},
        {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1), (0,1,1)]},
        {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'C': [(1,1,1)]},
        {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)], 'C': [(0,0,1), (1,1,1)]},
    ]
    
    col_idx = 0
    for st in strassen_4x4:
        for sub_t in strassen_2x2:
            for (br, bc, sign_a) in st['A']:
                for (pos, sign_sub) in sub_t['A']:
                    i, j = pos // 2, pos % 2
                    row = (br*2 + i) * m + (bc*2 + j)
                    U_cols[row].append(sign_a * sign_sub)
            
            for (br, bc, sign_b) in st['B']:
                for (pos, sign_sub) in sub_t['B']:
                    i, j = pos // 2, pos % 2
                    row = (br*2 + i) * p + (bc*2 + j)
                    V_cols[row].append(sign_b * sign_sub)
            
            for (br, bc, sign_c) in st['C']:
                for (pos, sign_sub) in sub_t['C']:
                    i, j = pos // 2, pos % 2
                    row = (bc*2 + j) * n + (br*2 + i)
                    W_cols[row].append(sign_c * sign_sub)
            
            col_idx += 1
    
    assert col_idx == 49, f"Expected 49, got {col_idx}"
    
    for i in range(n):
        for k in range(m):
            for j in range(p2):
                U_cols[i * m + k].append(1)
                V_cols[k * p + (p1 + j)].append(1)
                W_cols[(p1 + j) * n + i].append(1)
                col_idx += 1
    
    assert col_idx == rank, f"Expected {rank}, got {col_idx}"
    
    U = []
    for r in range(n * m):
        row = U_cols[r]
        while len(row) < rank:
            row.append(0)
        U.append(row)
    
    V = []
    for r in range(m * p):
        row = V_cols[r]
        while len(row) < rank:
            row.append(0)
        V.append(row)
    
    W = []
    for r in range(p * n):
        row = W_cols[r]
        while len(row) < rank:
            row.append(0)
        W.append(row)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
