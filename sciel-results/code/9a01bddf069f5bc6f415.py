"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen local terms: (u_local, v_local, w_local)
    # Indices are flattened: A[i*2+j], B[j*2+k], C[k*2+i]
    strassen_terms = [
        # Term 1: (A00+A11)(B00+B11) -> C00+1, C11+1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # Term 2: (A10+A11)B00 -> C10+1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
        # Term 3: A00(B01-B11) -> C01+1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]),
        # Term 4: A11(B10-B00) -> C00+1, C10+1
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        # Term 5: (A00+A01)B11 -> C00-1, C01+1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # Term 6: (A10-A00)(B00+B01) -> C11+1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # Term 7: (A01-A11)(B10+B11) -> C00+1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Define the 4 blocks: (a_offset_i, a_offset_j, b_offset_j, b_offset_k, c_offset_k, c_offset_i)
    # A global index: i*m + j. Local i, j in 0..1. Global i = i + ai, j = j + aj.
    # B global index: j*p + k. Local j, k in 0..1. Global j = j + bj, k = k + bk.
    # C global index: k*n + i. Local k, i in 0..1. Global k = k + ck, i = i + ci.
    
    blocks = [
        # Block 1: A_L, B_TL, C_L
        (0, 0, 0, 0, 0, 0),
        # Block 2: A_R, B_BL, C_L
        (0, 2, 2, 0, 0, 0),
        # Block 3: A_L, B_TR, C_R
        (0, 0, 0, 2, 2, 0),
        # Block 4: A_R, B_BR, C_R
        (0, 2, 2, 2, 2, 0),
    ]
    
    col = 0
    for (ai, aj, bj, bk, ck, ci) in blocks:
        for u_loc, v_loc, w_loc in strassen_terms:
            # Map local indices to global
            # u_loc index l corresponds to local A[i, j] where l = i*2 + j
            for l in range(4):
                if u_loc[l] != 0:
                    i_loc = l // 2
                    j_loc = l % 2
                    i_glob = i_loc + ai
                    j_glob = j_loc + aj
                    u_idx = i_glob * m + j_glob
                    U[u_idx][col] += u_loc[l]
                    
            for l in range(4):
                if v_loc[l] != 0:
                    j_loc = l // 2
                    k_loc = l % 2
                    j_glob = j_loc + bj
                    k_glob = k_loc + bk
                    v_idx = j_glob * p + k_glob
                    V[v_idx][col] += v_loc[l]
                    
            for l in range(4):
                if w_loc[l] != 0:
                    k_loc = l // 2
                    i_loc = l % 2
                    k_glob = k_loc + ck
                    i_glob = i_loc + ci
                    w_idx = k_glob * n + i_glob
                    W[w_idx][col] += w_loc[l]
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
