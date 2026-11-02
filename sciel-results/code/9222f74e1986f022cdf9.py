import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank calculation:
    # 2 blocks of m
    # Each m-block:
    #   1 block of n=2 (2x2x7) -> 25 rank
    #   1 block of n=1 (1x2x7) -> 14 rank
    # Total per m-block: 39
    # Total rank: 2 * 39 = 78
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(len(u_vec)):
            if u_vec[i] != 0:
                U[i][col] += u_vec[i]
        for i in range(len(v_vec)):
            if v_vec[i] != 0:
                V[i][col] += v_vec[i]
        for i in range(len(w_vec)):
            if w_vec[i] != 0:
                W[i][col] += w_vec[i]
        col += 1

    # Helper to map local indices to global indices
    # A: i (0..2), j (0..3) -> idx = i*4 + j
    # B: j (0..3), k (0..6) -> idx = j*7 + k
    # W: k (0..6), i (0..2) -> idx = k*3 + i
    
    # Define Strassen 2x2x2 terms
    # Local A: 2x2 (4 entries), Local B: 2x2 (4 entries), Local W: 2x2 (4 entries)
    strassen_terms = [
        ([1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Define Schoolbook 2x2x1 terms
    # Local A: 2x2 (4), Local B: 2x1 (2), Local W: 2x1 (2)
    schoolbook_221_terms = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1])
    ]
    
    # Function to add a 2x2x2 block
    # a_offset: (i_start, j_start)
    # b_offset: (j_start, k_start)
    # w_offset: (k_start, i_start)
    def add_222_block(i_start, j_start, k_start):
        for u, v, w in strassen_terms:
            # Map U
            u_global = [0] * (n*m)
            for li in range(2):
                for lj in range(2):
                    u_global[(i_start+li)*m + (j_start+lj)] = u[li*2+lj]
            # Map V
            v_global = [0] * (m*p)
            for lj in range(2):
                for lk in range(2):
                    v_global[(j_start+lj)*p + (k_start+lk)] = v[lj*2+lk]
            # Map W
            w_global = [0] * (p*n)
            for lk in range(2):
                for li in range(2):
                    w_global[(k_start+lk)*n + (i_start+li)] = w[lk*2+li]
            
            add_term(u_global, v_global, w_global)

    # Function to add a 2x2x1 block
    def add_221_block(i_start, j_start, k_start):
        for u, v, w in schoolbook_221_terms:
            u_global = [0] * (n*m)
            for li in range(2):
                for lj in range(2):
                    u_global[(i_start+li)*m + (j_start+lj)] = u[li*2+lj]
            
            v_global = [0] * (m*p)
            for lj in range(2):
                v_global[(j_start+lj)*p + k_start] = v[lj]
            
            w_global = [0] * (p*n)
            for li in range(2):
                w_global[k_start*n + (i_start+li)] = w[li]
            
            add_term(u_global, v_global, w_global)

    # Function to add a 1x2xp block
    def add_12p_block(i_start, j_start, k_start, p_len):
        # i is single row
        for lj in range(2):
            for lk in range(p_len):
                u_global = [0] * (n*m)
                u_global[i_start*m + (j_start+lj)] = 1
                
                v_global = [0] * (m*p)
                v_global[(j_start+lj)*p + (k_start+lk)] = 1
                
                w_global = [0] * (p*n)
                w_global[(k_start+lk)*n + i_start] = 1
                
                add_term(u_global, v_global, w_global)

    # Process m-block 0 (j=0,1)
    # n-block 0 (i=0,1) -> 2x2x7
    #   k=0,1 (2x2x2)
    add_222_block(0, 0, 0)
    #   k=2,3 (2x2x2)
    add_222_block(0, 0, 2)
    #   k=4,5 (2x2x2)
    add_222_block(0, 0, 4)
    #   k=6 (2x2x1)
    add_221_block(0, 0, 6)
    
    # n-block 1 (i=2) -> 1x2x7
    add_12p_block(2, 0, 0, 7)
    
    # Process m-block 1 (j=2,3)
    # n-block 0 (i=0,1) -> 2x2x7
    #   k=0,1 (2x2x2)
    add_222_block(0, 2, 0)
    #   k=2,3 (2x2x2)
    add_222_block(0, 2, 2)
    #   k=4,5 (2x2x2)
    add_222_block(0, 2, 4)
    #   k=6 (2x2x1)
    add_221_block(0, 2, 6)
    
    # n-block 1 (i=2) -> 1x2x7
    add_12p_block(2, 2, 0, 7)

    return {"U": U, "V": V, "W": W}
