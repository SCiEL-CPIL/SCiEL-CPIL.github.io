import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=4
    # U size: 8 x 28
    # V size: 16 x 28
    # W size: 8 x 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block
    # a_start: index of A block in U (0 for A0, 2 for A1) -> actually U index is i*m+j
    # A0: i in 0..1, j in 0..1 -> indices 0,1,4,5
    # A1: i in 0..1, j in 2..3 -> indices 2,3,6,7
    # b_start: index of B block in V (0 for B00, 2 for B01, 8 for B10, 10 for B11)
    # B00: j in 0..1, k in 0..1 -> indices 0,1,4,5
    # B01: j in 0..1, k in 2..3 -> indices 2,3,6,7
    # B10: j in 2..3, k in 0..1 -> indices 8,9,12,13
    # B11: j in 2..3, k in 2..3 -> indices 10,11,14,15
    # w_start: index of C block in W (0 for C0, 4 for C1)
    # C0: k in 0..1, i in 0..1 -> indices 0,1,2,3
    # C1: k in 2..3, i in 0..1 -> indices 4,5,6,7
    
    def add_strassen_block(u_base, v_base, w_base, col_start):
        # u_base is the offset for the 2x2 A block in U
        # A indices in U: i*m + j
        # If u_base=0, A is A0 (j=0,1). U indices: 0,1,4,5
        # If u_base=2, A is A1 (j=2,3). U indices: 2,3,6,7
        # Mapping:
        # Y00 -> u_base + 0 (if u_base=0, idx 0; if u_base=2, idx 2? No.)
        # Let's define explicit index mappings.
        
        # For A0 (u_base=0):
        # Y00 (i=0, j=0) -> U[0]
        # Y01 (i=0, j=1) -> U[1]
        # Y10 (i=1, j=0) -> U[4]
        # Y11 (i=1, j=1) -> U[5]
        
        # For A1 (u_base=2):
        # Y00 (i=0, j=2) -> U[2]
        # Y01 (i=0, j=3) -> U[3]
        # Y10 (i=1, j=2) -> U[6]
        # Y11 (i=1, j=3) -> U[7]
        
        u_idx = {
            'Y00': u_base,
            'Y01': u_base + 1,
            'Y10': u_base + 4,
            'Y11': u_base + 5
        }
        # Wait, if u_base=2:
        # Y00 -> 2
        # Y01 -> 3
        # Y10 -> 2+4=6
        # Y11 -> 2+5=7
        # This works.
        
        # For B00 (v_base=0):
        # Z00 (j=0, k=0) -> V[0]
        # Z01 (j=0, k=1) -> V[1]
        # Z10 (j=1, k=0) -> V[4]
        # Z11 (j=1, k=1) -> V[5]
        
        # For B01 (v_base=2):
        # Z00 (j=0, k=2) -> V[2]
        # Z01 (j=0, k=3) -> V[3]
        # Z10 (j=1, k=2) -> V[6]
        # Z11 (j=1, k=3) -> V[7]
        
        # For B10 (v_base=8):
        # Z00 (j=2, k=0) -> V[8]
        # Z01 (j=2, k=1) -> V[9]
        # Z10 (j=3, k=0) -> V[12]
        # Z11 (j=3, k=1) -> V[13]
        
        # For B11 (v_base=10):
        # Z00 (j=2, k=2) -> V[10]
        # Z01 (j=2, k=3) -> V[11]
        # Z10 (j=3, k=2) -> V[14]
        # Z11 (j=3, k=3) -> V[15]
        
        v_idx = {
            'Z00': v_base,
            'Z01': v_base + 1,
            'Z10': v_base + 4,
            'Z11': v_base + 5
        }
        
        # For C0 (w_base=0):
        # Z00 -> W[0]
        # Z10 -> W[1]
        # Z01 -> W[2]
        # Z11 -> W[3]
        
        # For C1 (w_base=4):
        # Z00 -> W[4]
        # Z10 -> W[5]
        # Z01 -> W[6]
        # Z11 -> W[7]
        
        w_idx = {
            'Z00': w_base,
            'Z10': w_base + 1,
            'Z01': w_base + 2,
            'Z11': w_base + 3
        }
        
        # Strassen Products
        # M1 = (Y00+Y11)(Z00+Z11)
        U[u_idx['Y00']][col_start + 0] += 1
        U[u_idx['Y11']][col_start + 0] += 1
        V[v_idx['Z00']][col_start + 0] += 1
        V[v_idx['Z11']][col_start + 0] += 1
        W[w_idx['Z00']][col_start + 0] += 1
        W[w_idx['Z11']][col_start + 0] += 1
        
        # M2 = (Y10+Y11)Z00
        U[u_idx['Y10']][col_start + 1] += 1
        U[u_idx['Y11']][col_start + 1] += 1
        V[v_idx['Z00']][col_start + 1] += 1
        W[w_idx['Z10']][col_start + 1] += 1
        W[w_idx['Z11']][col_start + 1] -= 1
        
        # M3 = Y00(Z10-Z11)
        U[u_idx['Y00']][col_start + 2] += 1
        V[v_idx['Z10']][col_start + 2] += 1
        V[v_idx['Z11']][col_start + 2] -= 1
        W[w_idx['Z01']][col_start + 2] += 1
        W[w_idx['Z11']][col_start + 2] += 1
        
        # M4 = Y11(Z01+Z11)
        U[u_idx['Y11']][col_start + 3] += 1
        V[v_idx['Z01']][col_start + 3] += 1
        V[v_idx['Z11']][col_start + 3] += 1
        W[w_idx['Z00']][col_start + 3] += 1
        W[w_idx['Z10']][col_start + 3] += 1
        
        # M5 = (Y00+Y01)Z11
        U[u_idx['Y00']][col_start + 4] += 1
        U[u_idx['Y01']][col_start + 4] += 1
        V[v_idx['Z11']][col_start + 4] += 1
        W[w_idx['Z00']][col_start + 4] -= 1
        W[w_idx['Z01']][col_start + 4] += 1
        
        # M6 = (Y10-Y00)(Z00+Z01)
        U[u_idx['Y10']][col_start + 5] += 1
        U[u_idx['Y00']][col_start + 5] -= 1
        V[v_idx['Z00']][col_start + 5] += 1
        V[v_idx['Z01']][col_start + 5] += 1
        W[w_idx['Z11']][col_start + 5] += 1
        
        # M7 = (Y01-Y11)(Z10+Z11)
        U[u_idx['Y01']][col_start + 6] += 1
        U[u_idx['Y11']][col_start + 6] -= 1
        V[v_idx['Z10']][col_start + 6] += 1
        V[v_idx['Z11']][col_start + 6] += 1
        W[w_idx['Z00']][col_start + 6] += 1

    # Block 0: A0 B00 -> C0
    add_strassen_block(0, 0, 0, 0)
    # Block 1: A1 B10 -> C0
    add_strassen_block(2, 8, 0, 7)
    # Block 2: A0 B01 -> C1
    add_strassen_block(0, 2, 4, 14)
    # Block 3: A1 B11 -> C1
    add_strassen_block(2, 10, 4, 21)

    return {"U": U, "V": V, "W": W}
