import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_i, a_j, b_j, b_k, c_i, c_k):
        nonlocal col
        # a_i, a_j: offset in A
        # b_j, b_k: offset in B
        # c_i, c_k: offset in C (i is row, k is col)
        
        # Map local (0,0) to global
        # A indices: i in {0,1}, j in {0,1}
        # B indices: j in {0,1}, k in {0,1}
        # C indices: i in {0,1}, k in {0,1}
        
        # U indices: (a_i + di) * m + (a_j + dj)
        # V indices: (b_j + dj) * p + (b_k + dk)
        # W indices: (c_k + dk) * n + (c_i + di)
        
        def u_idx(di, dj): return (a_i + di) * m + (a_j + dj)
        def v_idx(dj, dk): return (b_j + dj) * p + (b_k + dk)
        def w_idx(dk, di): return (c_k + dk) * n + (c_i + di)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx(0,0)][col] += 1
        U[u_idx(1,1)][col] += 1
        V[v_idx(0,0)][col] += 1
        V[v_idx(1,1)][col] += 1
        W[w_idx(0,0)][col] += 1 # C00
        W[w_idx(1,1)][col] += 1 # C11
        col += 1
        
        # M2 = (A10 + A11) B00
        U[u_idx(1,0)][col] += 1
        U[u_idx(1,1)][col] += 1
        V[v_idx(0,0)][col] += 1
        W[w_idx(0,1)][col] += 1 # C10
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[u_idx(0,0)][col] += 1
        V[v_idx(0,1)][col] += 1
        V[v_idx(1,1)][col] -= 1
        W[w_idx(0,0)][col] += 1 # C00
        W[w_idx(0,1)][col] += 1 # C01
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[u_idx(1,1)][col] += 1
        V[v_idx(1,0)][col] += 1
        V[v_idx(0,0)][col] -= 1
        W[w_idx(0,0)][col] += 1 # C00
        W[w_idx(1,0)][col] += 1 # C10
        col += 1
        
        # M5 = (A00 + A01) B11
        U[u_idx(0,0)][col] += 1
        U[u_idx(0,1)][col] += 1
        V[v_idx(1,1)][col] += 1
        W[w_idx(0,0)][col] -= 1 # C00
        W[w_idx(1,1)][col] += 1 # C11
        col += 1
        
        # M6 = (A10 - A00) (B00 + B01)
        U[u_idx(1,0)][col] += 1
        U[u_idx(0,0)][col] -= 1
        V[v_idx(0,0)][col] += 1
        V[v_idx(0,1)][col] += 1
        W[w_idx(1,1)][col] += 1 # C11
        col += 1
        
        # M7 = (A01 - A11) (B10 + B11)
        U[u_idx(0,1)][col] += 1
        U[u_idx(1,1)][col] -= 1
        V[v_idx(1,0)][col] += 1
        V[v_idx(1,1)][col] += 1
        W[w_idx(0,0)][col] += 1 # C00
        W[w_idx(1,1)][col] -= 1 # C11
        col += 1
        
    def add_schoolbook_221(a_i, a_j, b_j, b_k, c_i, c_k):
        # A is 2x2, B is 2x1, C is 2x1
        nonlocal col
        def u_idx(di, dj): return (a_i + di) * m + (a_j + dj)
        def v_idx(dj, dk): return (b_j + dj) * p + (b_k + dk)
        def w_idx(dk, di): return (c_k + dk) * n + (c_i + di)
        
        # C_i0 = A_i0 B_00 + A_i1 B_10
        # i=0: A00 B00 -> C00
        U[u_idx(0,0)][col] += 1
        V[v_idx(0,0)][col] += 1
        W[w_idx(0,0)][col] += 1
        col += 1
        
        # i=0: A01 B10 -> C00
        U[u_idx(0,1)][col] += 1
        V[v_idx(1,0)][col] += 1
        W[w_idx(0,0)][col] += 1
        col += 1
        
        # i=1: A10 B00 -> C10
        U[u_idx(1,0)][col] += 1
        V[v_idx(0,0)][col] += 1
        W[w_idx(0,1)][col] += 1
        col += 1
        
        # i=1: A11 B10 -> C10
        U[u_idx(1,1)][col] += 1
        V[v_idx(1,0)][col] += 1
        W[w_idx(0,1)][col] += 1
        col += 1

    def add_schoolbook_215(a_i, a_j, b_j, b_k, c_i, c_k):
        # A is 2x1, B is 1x5, C is 2x5
        nonlocal col
        def u_idx(di, dj): return (a_i + di) * m + (a_j + dj)
        def v_idx(dj, dk): return (b_j + dj) * p + (b_k + dk)
        def w_idx(dk, di): return (c_k + dk) * n + (c_i + di)
        
        # C_ik = A_i0 B_0k
        for di in range(2):
            for dk in range(5):
                U[u_idx(di, 0)][col] += 1
                V[v_idx(0, dk)][col] += 1
                W[w_idx(dk, di)][col] += 1
                col += 1

    # Block A0 (rows 0,1; cols 0,1) * B0 (rows 0,1; cols 0,1) -> C0 (rows 0,1; cols 0,1)
    add_strassen(0, 0, 0, 0, 0, 0)
    # Block A0 * B0 (rows 0,1; cols 2,3) -> C0 (rows 0,1; cols 2,3)
    add_strassen(0, 0, 0, 2, 0, 2)
    # Block A0 * B0 (rows 0,1; col 4) -> C0 (rows 0,1; col 4)
    add_schoolbook_221(0, 0, 0, 4, 0, 4)
    
    # Block A1 (rows 0,1; cols 2,3) * B1 (rows 2,3; cols 0,1) -> C1 (rows 0,1; cols 0,1)
    add_strassen(0, 2, 2, 0, 0, 0)
    # Block A1 * B1 (rows 2,3; cols 2,3) -> C1 (rows 0,1; cols 2,3)
    add_strassen(0, 2, 2, 2, 0, 2)
    # Block A1 * B1 (rows 2,3; col 4) -> C1 (rows 0,1; col 4)
    add_schoolbook_221(0, 2, 2, 4, 0, 4)
    
    # Block A2 (rows 0,1; col 4) * B2 (row 4; cols 0,1,2,3,4) -> C2 (rows 0,1; cols 0,1,2,3,4)
    add_schoolbook_215(0, 4, 4, 0, 0, 0)
    
    return {"U": U, "V": V, "W": W}
