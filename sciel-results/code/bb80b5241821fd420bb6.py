import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    total_rank = 42
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, value=1):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += value
        for idx in v_indices:
            V[idx][col] += value
        for idx in w_indices:
            W[idx][col] += value
        col += 1

    # Helper for Strassen 2x2x2
    # Local indices: i,j in {0,1}, k in {0,1}
    # Global mapping: i -> gi, j -> gj, k -> gk
    def add_strassen(gi, gj, gk):
        nonlocal col
        # M1
        add_term([gi[0]*m + gj[0], gi[1]*m + gj[1]], 
                 [gj[0]*p + gk[0], gj[1]*p + gk[1]], 
                 [gk[0]*n + gi[0], gk[1]*n + gi[1]])
        # M2
        add_term([gi[1]*m + gj[0], gi[1]*m + gj[1]], 
                 [gj[0]*p + gk[0]], 
                 [gk[0]*n + gi[1]])
        # M3
        add_term([gi[0]*m + gj[0]], 
                 [gj[0]*p + gk[1], gj[1]*p + gk[1]], 
                 [gk[1]*n + gi[0]])
        # M4
        add_term([gi[1]*m + gj[1]], 
                 [gj[1]*p + gk[0], gj[0]*p + gk[0]], 
                 [gk[0]*n + gi[1]])
        # M5
        add_term([gi[0]*m + gj[0], gi[0]*m + gj[1]], 
                 [gj[1]*p + gk[1]], 
                 [gk[1]*n + gi[0]])
        # M6
        add_term([gi[1]*m + gj[0], gi[0]*m + gj[0]], 
                 [gj[0]*p + gk[0], gj[0]*p + gk[1]], 
                 [gk[1]*n + gi[1]])
        # M7
        add_term([gi[0]*m + gj[1], gi[1]*m + gj[1]], 
                 [gj[1]*p + gk[0], gj[1]*p + gk[1]], 
                 [gk[0]*n + gi[0]])

    # Helper for Schoolbook 2x2x1
    def add_schoolbook_221(gi, gj, gk):
        nonlocal col
        for i in range(2):
            for j in range(2):
                add_term([gi[i]*m + gj[j]], 
                         [gj[j]*p + gk], 
                         [gk*n + gi[i]])

    # Helper for Schoolbook 1x2x7
    def add_schoolbook_127(gi, gj, gk_list):
        nonlocal col
        # i is single, j in {0,1}, k in gk_list
        for j in range(2):
            for k in gk_list:
                add_term([gi*m + gj[j]], 
                         [gj[j]*p + k], 
                         [k*n + gi])

    # Helper for Rank 1 (1x1x7)
    def add_rank1_117(gi, gj, gk_list):
        nonlocal col
        # This is outer product, but we need to decompose into rank 1 terms?
        # C[i,k] = A[i,j] * B[j,k]
        # For 1x1x7, it's just A * B_row.
        # We can do this with 7 terms of rank 1?
        # Or 1 term?
        # T = u (1) x v (7) x w (7).
        # u[0] = 1.
        # v[k] = 1.
        # w[k] = 1.
        # But W has shape (p*n, rank).
        # If we use 1 term:
        # U[gi*m+gj] = 1
        # V[gj*p+k] = 1 for all k
        # W[k*n+gi] = 1 for all k
        # This works.
        u_idx = gi*m + gj
        U[u_idx][col] = 1
        for k in gk_list:
            V[gj*p + k][col] = 1
            W[k*n + gi][col] = 1
        col += 1

    # Helper for Rank 2 (2x1x7)
    def add_rank2_217(gi_list, gj, gk_list):
        nonlocal col
        # i in {0,1}, j is single, k in gk_list
        # C[i,k] = A[i,j] * B[j,k]
        # Term 1: i=0
        U[gi_list[0]*m + gj][col] = 1
        for k in gk_list:
            V[gj*p + k][col] = 1
            W[k*n + gi_list[0]][col] = 1
        col += 1
        # Term 2: i=1
        U[gi_list[1]*m + gj][col] = 1
        for k in gk_list:
            V[gj*p + k][col] = 1
            W[k*n + gi_list[1]][col] = 1
        col += 1

    # Part 1: A11 B11 (2x2x7)
    # gi = [0, 1], gj = [0, 1]
    gi = [0, 1]
    gj = [0, 1]
    
    # k=0,1 (Strassen 2x2x2)
    add_strassen(gi, gj, [0, 1])
    
    # k=2 (Schoolbook 2x2x1)
    add_schoolbook_221(gi, gj, 2)
    
    # k=3,4 (Strassen 2x2x2)
    add_strassen(gi, gj, [3, 4])
    
    # k=5,6 (Strassen 2x2x2)
    add_strassen(gi, gj, [5, 6])
    
    # Part 2: A12 B21 (2x1x7)
    # gi = [0, 1], gj = [2]
    add_rank2_217([0, 1], 2, [0, 1, 2, 3, 4, 5, 6])
    
    # Part 3: A21 B11 (1x2x7)
    # gi = [2], gj = [0, 1]
    add_schoolbook_127(2, [0, 1], [0, 1, 2, 3, 4, 5, 6])
    
    # Part 4: A22 B21 (1x1x7)
    # gi = [2], gj = [2]
    add_rank1_117(2, 2, [0, 1, 2, 3, 4, 5, 6])
    
    return {"U": U, "V": V, "W": W}
