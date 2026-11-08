import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 145.0
    safety_margin = 5.0

    n, m, p = DIMENSIONS
    # Target rank: 60 (from Strassen-based tiling)
    target_rank = 60

    # Initialize factor matrices as numpy arrays for efficiency
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))

    col = 0

    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        if col >= target_rank:
            return
        u_vec = np.asarray(u_vec)
        v_vec = np.asarray(v_vec)
        w_vec = np.asarray(w_vec)
        U[:, col] += u_vec
        V[:, col] += v_vec
        W[:, col] += w_vec
        col += 1

    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    # Output factor follows W[k*n+i, r] where k is column of C, i is row of C
    # C[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # So W index is k*n + i
    def w_idx(i, j): return j * n + i

    def add_strassen_block(a_row_start, b_col_start, c_col_start, k_start):
        # Adds 7 Strassen terms for a 2x2 sub-block
        # a_block is rows a_row_start..a_row_start+1, cols 0..1 of A
        # b_block is rows 0..1 of B, cols k_start..k_start+1
        # c_block is rows 0..1 of C, cols k_start..k_start+1

        a00 = u_idx(a_row_start, 0)
        a01 = u_idx(a_row_start, 1)
        a10 = u_idx(a_row_start + 1, 0)
        a11 = u_idx(a_row_start + 1, 1)

        b00 = v_idx(0, k_start)
        b01 = v_idx(0, k_start + 1)
        b10 = v_idx(1, k_start)
        b11 = v_idx(1, k_start + 1)

        c00 = w_idx(0, k_start)
        c01 = w_idx(0, k_start + 1)
        c10 = w_idx(1, k_start)
        c11 = w_idx(1, k_start + 1)

        # m1: (a00+a11)(b00+b11) -> contributes to c00 and c11
        u1 = np.zeros(n*m); v1 = np.zeros(m*p); w1 = np.zeros(p*n)
        u1[a00] = 1; u1[a11] = 1
        v1[b00] = 1; v1[b11] = 1
        w1[c00] = 1; w1[c11] = 1
        add_term(u1, v1, w1)

        # m2: (a10+a11)b00 -> contributes to c10 and c11
        u2 = np.zeros(n*m); v2 = np.zeros(m*p); w2 = np.zeros(p*n)
        u2[a10] = 1; u2[a11] = 1
        v2[b00] = 1
        w2[c10] = 1; w2[c11] = -1
        add_term(u2, v2, w2)

        # m3: a00(b01-b11) -> contributes to c01 and c11
        u3 = np.zeros(n*m); v3 = np.zeros(m*p); w3 = np.zeros(p*n)
        u3[a00] = 1
        v3[b01] = 1; v3[b11] = -1
        w3[c01] = 1; w3[c11] = 1
        add_term(u3, v3, w3)

        # m4: a11(b10-b00) -> contributes to c00 and c10
        u4 = np.zeros(n*m); v4 = np.zeros(m*p); w4 = np.zeros(p*n)
        u4[a11] = 1
        v4[b10] = 1; v4[b00] = -1
        w4[c00] = 1; w4[c10] = 1
        add_term(u4, v4, w4)

        # m5: (a00+a01)b11 -> contributes to c00 and c01
        u5 = np.zeros(n*m); v5 = np.zeros(m*p); w5 = np.zeros(p*n)
        u5[a00] = 1; u5[a01] = 1
        v5[b11] = 1
        w5[c00] = -1; w5[c01] = 1
        add_term(u5, v5, w5)

        # m6: (a10-a00)(b00+b01) -> contributes to c11
        u6 = np.zeros(n*m); v6 = np.zeros(m*p); w6 = np.zeros(p*n)
        u6[a10] = 1; u6[a00] = -1
        v6[b00] = 1; v6[b01] = 1
        w6[c11] = 1
        add_term(u6, v6, w6)

        # m7: (a01-a11)(b10+b11) -> contributes to c00
        u7 = np.zeros(n*m); v7 = np.zeros(m*p); w7 = np.zeros(p*n)
        u7[a01] = 1; u7[a11] = -1
        v7[b10] = 1; v7[b11] = 1
        w7[c00] = 1
        add_term(u7, v7, w7)

    def add_schoolbook_block(a_row_start, b_col_start, c_col_start, k_start, a_rows=2, b_cols=2):
        # Schoolbook for remaining single column block
        for i in range(a_rows):
            for j in range(b_cols):
                u_t = np.zeros(n*m); v_t = np.zeros(m*p); w_t = np.zeros(p*n)
                u_t[u_idx(a_row_start + i, j)] = 1
                v_t[v_idx(j, k_start)] = 1
                w_t[w_idx(a_row_start + i, k_start)] = 1
                add_term(u_t, v_t, w_t)

    # --- Construction ---
    
    # Part 1: Rows 0,1 of A (2x2 block) * All 3 rows of B, 7 cols of C
    # B has 3 rows. We use rows 0,1 of B with Strassen for pairs of k columns.
    # B row 2 is handled separately in Part 3.
    # So for the 2x2 A block, we multiply by B[0:2, :] (2x7)
    
    # k columns 0,1: Strassen
    add_strassen_block(0, 0, 0, 0)
    # k columns 2,3: Strassen
    add_strassen_block(0, 0, 2, 2)
    # k columns 4,5: Strassen
    add_strassen_block(0, 0, 4, 4)
    # k column 6: Schoolbook (only 1 column left)
    add_schoolbook_block(0, 0, 6, 6, a_rows=2, b_cols=2)
    
    # Rank used so far: 7 * 3 + 4 = 25
    
    # Part 2: Row 2 of A (1x2 block) * B[0:2, :] (2x7)
    # This is a 1x2 * 2x7 multiplication.
    # C row 2, cols 0..6 receive contributions from A[2,0]*B[0,:] + A[2,1]*B[1,:]
    # We can use schoolbook here: 2*7 = 14 terms?
    # Or can we do better?
    # M(1, 2, 7) rank is at least 7 (tensor rank of 1x2*2x7).
    # Actually, since A is 1x2 and B is 2x7, the result is 1x7.
    # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k]
    # This is exactly schoolbook. Rank = 14?
    # Wait, can we share terms?
    # The tensor is sum_{j=0}^1 A[2,j] B[j,k] -> C[2,k]
    # This is M(1,2,7). The rank of M(1,2,7) is 1*2*7 = 14 if no structure.
    # But since A has only 1 row, and B has 2 rows, and C has 1 row.
    # It's a bilinear map F(R^2, R^{14}) -> R^7.
    # Actually, let's just use schoolbook for now.
    # 14 terms.
    # Total rank so far: 25 + 14 = 39.
    
    for j in range(2):
        for k in range(7):
            u_t = np.zeros(n*m); v_t = np.zeros(m*p); w_t = np.zeros(p*n)
            u_t[u_idx(2, j)] = 1
            v_t[v_idx(j, k)] = 1
            w_t[w_idx(2, k)] = 1
            add_term(u_t, v_t, w_t)

    # Part 3: All 3 rows of A * B[2, :] (1x7)
    # A is 3x1 (col 2), B is 1x7 (row 2).
    # C[0:3, 0:7] receives contributions from A[i,2]*B[2,k]
    # This is M(3,1,7).
    # Rank = 3*7 = 21.
    # Total rank so far: 39 + 21 = 60.
    
    for i in range(3):
        for k in range(7):
            u_t = np.zeros(n*m); v_t = np.zeros(m*p); w_t = np.zeros(p*n)
            u_t[u_idx(i, 2)] = 1
            v_t[v_idx(2, k)] = 1
            w_t[w_idx(i, k)] = 1
            add_term(u_t, v_t, w_t)

    # Check if we used exactly 60 columns
    # If col < 60, we have unused columns. We should trim or it's fine if we return 60 columns.
    # The problem asks for rank-minimal. If we used 60, that's our rank.
    # If we used less, we can trim.
    
    if col < target_rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        target_rank = col

    # Convert to lists of lists
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
