"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    Strategy: Partition the middle dimension (size 5) and output dimension (size 5) into
    blocks of sizes (2, 2, 1). This gives four 2x2x2 sub-problems (using Strassen rank-7),
    four 2x2x1 or 2x1x2 sub-problems (schoolbook rank 4 each), and one 2x1x1 (rank 2).
    Total rank: 4*7 + 2*4 + 2*4 + 1*2 = 28 + 8 + 8 + 2 = 46.
    
    Indexing convention:
    - U has shape (n*m, rank) = (10, rank), with U[i*m + j, c]
    - V has shape (m*p, rank) = (25, rank), with V[j*p + k, c]  
    - W has shape (p*n, rank) = (10, rank), with W[k*n + i, c]
    """
    import time
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_entries, v_entries, w_entries):
        nonlocal rank
        col = rank
        for idx, val in u_entries:
            while len(U) <= idx:
                U.append([0] * (rank + 1))
            if len(U[idx]) <= col:
                U[idx].extend([0] * (col + 1 - len(U[idx])))
            U[idx][col] += val
        for idx, val in v_entries:
            while len(V) <= idx:
                V.append([0] * (rank + 1))
            if len(V[idx]) <= col:
                V[idx].extend([0] * (col + 1 - len(V[idx])))
            V[idx][col] += val
        for idx, val in w_entries:
            while len(W) <= idx:
                W.append([0] * (rank + 1))
            if len(W[idx]) <= col:
                W[idx].extend([0] * (col + 1 - len(W[idx])))
            W[idx][col] += val
        rank += 1
    
    mid_blocks = [(0, 2), (2, 4), (4, 5)]
    out_blocks = [(0, 2), (2, 4), (4, 5)]
    
    for b_idx, (j_start, j_end) in enumerate(mid_blocks):
        s_b = j_end - j_start
        for d_idx, (k_start, k_end) in enumerate(out_blocks):
            s_d = k_end - k_start
            
            if time.time() > deadline:
                break
            
            if s_b == 2 and s_d == 2:
                j0, j1 = j_start, j_start + 1
                k0, k1 = k_start, k_start + 1
                
                # M1 = (A00 + A11)(B00 + B11)
                # Contributes: +1 to C00, +1 to C11
                add_term(
                    [(0*m+j0, 1), (1*m+j1, 1)],
                    [(j0*p+k0, 1), (j1*p+k1, 1)],
                    [(k0*n+0, 1), (k1*n+1, 1)],
                )
                # M2 = (A10 + A11)B00
                # Contributes: +1 to C10, -1 to C11
                add_term(
                    [(1*m+j0, 1), (1*m+j1, 1)],
                    [(j0*p+k0, 1)],
                    [(k0*n+1, 1), (k1*n+1, -1)],
                )
                # M3 = A00(B01 - B11)
                # Contributes: +1 to C01, +1 to C11
                add_term(
                    [(0*m+j0, 1)],
                    [(j0*p+k1, 1), (j1*p+k1, -1)],
                    [(k1*n+0, 1), (k1*n+1, 1)],
                )
                # M4 = A11(B10 - B00)
                # Contributes: +1 to C00, +1 to C10
                add_term(
                    [(1*m+j1, 1)],
                    [(j1*p+k0, 1), (j0*p+k0, -1)],
                    [(k0*n+0, 1), (k0*n+1, 1)],
                )
                # M5 = (A00 + A01)B11
                # Contributes: -1 to C00, +1 to C01
                add_term(
                    [(0*m+j0, 1), (0*m+j1, 1)],
                    [(j1*p+k1, 1)],
                    [(k0*n+0, -1), (k1*n+0, 1)],
                )
                # M6 = (A10 - A00)(B00 + B01)
                # Contributes: +1 to C11
                add_term(
                    [(1*m+j0, 1), (0*m+j0, -1)],
                    [(j0*p+k0, 1), (j0*p+k1, 1)],
                    [(k1*n+1, 1)],
                )
                # M7 = (A01 - A11)(B10 + B11)
                # Contributes: +1 to C00
                add_term(
                    [(0*m+j1, 1), (1*m+j1, -1)],
                    [(j1*p+k0, 1), (j1*p+k1, 1)],
                    [(k0*n+0, 1)],
                )
                
            elif s_b == 2 and s_d == 1:
                j0, j1 = j_start, j_start + 1
                k0 = k_start
                for j in [j0, j1]:
                    add_term(
                        [(0*m+j, 1), (1*m+j, 1)],
                        [(j*p+k0, 1)],
                        [(k0*n+0, 1), (k0*n+1, 1)],
                    )
                    
            elif s_b == 1 and s_d == 2:
                j0 = j_start
                for k in [k_start, k_start+1]:
                    add_term(
                        [(0*m+j0, 1), (1*m+j0, 1)],
                        [(j0*p+k, 1)],
                        [(k*n+0, 1), (k*n+1, 1)],
                    )
                    
            elif s_b == 1 and s_d == 1:
                j0 = j_start
                k0 = k_start
                add_term(
                    [(0*m+j0, 1), (1*m+j0, 1)],
                    [(j0*p+k0, 1)],
                    [(k0*n+0, 1), (k0*n+1, 1)],
                )
    
    target_U_rows = n * m
    target_V_rows = m * p
    target_W_rows = p * n
    
    while len(U) < target_U_rows:
        U.append([0] * rank)
    while len(V) < target_V_rows:
        V.append([0] * rank)
    while len(W) < target_W_rows:
        W.append([0] * rank)
    
    for i in range(target_U_rows):
        while len(U[i]) < rank:
            U[i].append(0)
    for i in range(target_V_rows):
        while len(V[i]) < rank:
            V[i].append(0)
    for i in range(target_W_rows):
        while len(W[i]) < rank:
            W[i].append(0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
