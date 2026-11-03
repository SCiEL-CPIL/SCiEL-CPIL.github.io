import time
import sys
import random

def construct():
    n, m, p = 4, 5, 5  # A is n x m, B is m x p, C is n x p
    
    # Build the full naive (schoolbook) decomposition as baseline
    # This gives rank = n * m * p = 100
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # W[k*n + i, r] where k is column of C, i is row of C
    
    U_cols = []  # Each element is a dict: {(i, j): val}
    V_cols = []  # Each element is a dict: {(j, k): val}
    W_cols = []  # Each element is a dict: {(i, k): val}
    
    def add_term(u_terms, v_terms, w_terms):
        U_col = {}
        V_col = {}
        W_col = {}
        
        for i, j, val in u_terms:
            key = (i, j)
            U_col[key] = U_col.get(key, 0) + val
            
        for j, k, val in v_terms:
            key = (j, k)
            V_col[key] = V_col.get(key, 0) + val
            
        for i, k, val in w_terms:
            key = (i, k)
            W_col[key] = W_col.get(key, 0) + val
            
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    # Strassen 2x2x2 decomposition
    # For 2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    strassen_terms = [
        # (U terms: (i,j,val), V terms: (j,k,val), W terms: (i,k,val))
        # m1 = (a+d)(e+h): U has (0,0,1),(1,1,1); V has (0,0,1),(1,1,1); W has (0,0,1),(1,1,1)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # m2 = (c+d)e: U has (1,0,1),(1,1,1); V has (0,0,1); W has (1,0,1)
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        # m3 = a(f-h): U has (0,0,1); V has (0,1,1),(1,1,-1); W has (0,1,1)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),
        # m4 = d(g-e): U has (1,1,1); V has (1,0,1),(0,0,-1); W has (0,0,-1),(1,0,1)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,-1),(1,0,1)]),
        # m5 = (a+b)h: U has (0,0,1),(0,1,1); V has (1,1,1); W has (0,0,-1),(0,1,1)
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # m6 = (c-a)(e+f): U has (1,0,1),(0,0,-1); V has (0,0,1),(0,1,1); W has (1,1,1)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # m7 = (b-d)(g+h): U has (0,1,1),(1,1,-1); V has (1,0,1),(1,1,1); W has (0,0,1)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Apply Strassen to the 4x4 top-left block (rows 0-3, cols 0-3 of A; rows 0-3, cols 0-3 of B)
    # This gives C[0:4, 0:4] from A[0:4, 0:4] * B[0:4, 0:4]
    # Use 2x2 Strassen for each of the four 2x2 quadrants
    for qr in range(2):
        for qc in range(2):
            row_off = qr * 2
            col_off = qc * 2
            
            for u_t, v_t, w_t in strassen_terms:
                u_g = [(r + row_off, c + col_off, val) for r, c, val in u_t]
                v_g = [(r + row_off, c + col_off, val) for r, c, val in v_t]
                w_g = [(r + row_off, c + col_off, val) for r, c, val in w_t]
                
                add_term(u_g, v_g, w_g)
    
    # Now handle the cross terms for C[0:4, 0:5] from A[0:4, 4] * B[4, 0:5]
    # A[0:4, 4] is a 4x1 column, B[4, 0:5] is a 1x5 row
    # C[i, k] += A[i, 4] * B[4, k] for i in 0..3, k in 0..4
    for i in range(4):
        for k in range(5):
            add_term([(i, 4, 1)], [(4, k, 1)], [(i, k, 1)])
    
    # Handle C[0:4, 0:5] from A[0:4, 0:4] * B[0:4, 4]
    # But we already handled A[0:4,0:4]*B[0:4,0:4] via Strassen for C[0:4,0:4]
    # We need C[i,k] for k=4: C[i,4] += A[i,j]*B[j,4] for j in 0..3
    for i in range(4):
        for j in range(4):
            add_term([(i, j, 1)], [(j, 4, 1)], [(i, 4, 1)])
    
    # Handle C[0:4, 4] from A[0:4, 4] * B[4, 4]
    for i in range(4):
        add_term([(i, 4, 1)], [(4, 4, 1)], [(i, 4, 1)])
    
    rank = len(U_cols)
    
    # Convert to dense matrices
    # U: shape (n*m, rank), indexed by [i*m + j, r]
    # V: shape (m*p, rank), indexed by [j*p + k, r]
    # W: shape (p*n, rank), indexed by [k*n + i, r]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r_idx, col in enumerate(U_cols):
        for (i, j), val in col.items():
            U[i * m + j][r_idx] = val
    
    for r_idx, col in enumerate(V_cols):
        for (j, k), val in col.items():
            V[j * p + k][r_idx] = val
    
    for r_idx, col in enumerate(W_cols):
        for (i, k), val in col.items():
            W[k * n + i][r_idx] = val
    
    return {"U": U, "V": V, "W": W}
