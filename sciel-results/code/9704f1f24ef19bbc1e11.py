import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(r, c): return r * m + c
    def v_idx(c, k): return c * p + k
    def w_idx(k, r): return k * n + r
    
    def add_strassen(col_offset, a_rows, a_cols, b_rows, b_cols):
        r1, r2 = a_rows
        c1, c2 = a_cols
        k1, k2 = b_cols
        
        # M1
        U[u_idx(r1, c1)][col_offset] += 1
        U[u_idx(r2, c2)][col_offset] += 1
        V[v_idx(c1, k1)][col_offset] += 1
        V[v_idx(c2, k2)][col_offset] += 1
        W[w_idx(k1, r1)][col_offset] += 1
        W[w_idx(k2, r2)][col_offset] += 1
        
        # M2
        U[u_idx(r2, c1)][col_offset+1] += 1
        U[u_idx(r2, c2)][col_offset+1] += 1
        V[v_idx(c1, k1)][col_offset+1] += 1
        W[w_idx(k1, r2)][col_offset+1] += 1
        
        # M3
        U[u_idx(r1, c1)][col_offset+2] += 1
        V[v_idx(c1, k2)][col_offset+2] -= 1
        V[v_idx(c2, k2)][col_offset+2] += 1
        W[w_idx(k2, r1)][col_offset+2] += 1
        
        # M4
        U[u_idx(r2, c2)][col_offset+3] += 1
        V[v_idx(c2, k1)][col_offset+3] -= 1
        V[v_idx(c1, k1)][col_offset+3] += 1
        W[w_idx(k1, r2)][col_offset+3] += 1
        
        # M5
        U[u_idx(r1, c1)][col_offset+4] += 1
        U[u_idx(r1, c2)][col_offset+4] += 1
        V[v_idx(c2, k2)][col_offset+4] += 1
        W[w_idx(k2, r1)][col_offset+4] += 1
        
        # M6
        U[u_idx(r2, c1)][col_offset+5] += 1
        U[u_idx(r1, c1)][col_offset+5] -= 1
        V[v_idx(c1, k1)][col_offset+5] += 1
        V[v_idx(c1, k2)][col_offset+5] += 1
        W[w_idx(k1, r2)][col_offset+5] += 1
        W[w_idx(k2, r2)][col_offset+5] += 1
        
        # M7
        U[u_idx(r1, c2)][col_offset+6] += 1
        U[u_idx(r2, c2)][col_offset+6] -= 1
        V[v_idx(c2, k1)][col_offset+6] += 1
        V[v_idx(c2, k2)][col_offset+6] += 1
        W[w_idx(k1, r1)][col_offset+6] += 1
        W[w_idx(k2, r1)][col_offset+6] += 1

    col = 0
    
    # Block 1: j in {0, 1}
    # A cols 0, 1. B rows 0, 1.
    # A row pairs: (0,1), (2,3)
    # B col pairs: (0,1), (2,3), (4,5)
    
    add_strassen(col, [0, 1], [0, 1], [0, 1], [0, 1]); col += 7
    add_strassen(col, [0, 1], [0, 1], [0, 1], [2, 3]); col += 7
    add_strassen(col, [0, 1], [0, 1], [0, 1], [4, 5]); col += 7
    add_strassen(col, [2, 3], [0, 1], [0, 1], [0, 1]); col += 7
    add_strassen(col, [2, 3], [0, 1], [0, 1], [2, 3]); col += 7
    add_strassen(col, [2, 3], [0, 1], [0, 1], [4, 5]); col += 7
    
    # Block 2: j in {2, 3}
    # A cols 2, 3. B rows 2, 3.
    
    add_strassen(col, [0, 1], [2, 3], [2, 3], [0, 1]); col += 7
    add_strassen(col, [0, 1], [2, 3], [2, 3], [2, 3]); col += 7
    add_strassen(col, [0, 1], [2, 3], [2, 3], [4, 5]); col += 7
    add_strassen(col, [2, 3], [2, 3], [2, 3], [0, 1]); col += 7
    add_strassen(col, [2, 3], [2, 3], [2, 3], [2, 3]); col += 7
    add_strassen(col, [2, 3], [2, 3], [2, 3], [4, 5]); col += 7
    
    # Block 3: j in {4}
    # A col 4. B row 4.
    # Rank 1 term
    for i in range(n):
        U[u_idx(i, 4)][col] += 1
    for k in range(p):
        V[v_idx(4, k)][col] += 1
    for k in range(p):
        for i in range(n):
            W[w_idx(k, i)][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
