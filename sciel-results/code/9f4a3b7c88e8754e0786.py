def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 * 2x2
    # Returns list of (u_vec, v_vec, w_vec) where vectors are sparse dicts or lists
    # We define the 7 terms for 2x2 multiplication.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    # P1 = (a+d)(e+h) -> i+, l+
    # P2 = (c+d)e     -> k+, l-
    # P3 = a(f-h)     -> j+, l+
    # P4 = d(g-e)     -> i+, k+
    # P5 = (a+b)h     -> i-, j+
    # P6 = (c-a)(e+f) -> l+
    # P7 = (b-d)(g+h) -> i+
    
    # Indices for 2x2:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    strassen_2x2 = [
        # (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
        # P1
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # P2
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # P3
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        # P4
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        # P5
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # P6
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # P7
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    # We need to decompose 2x4 * 4x8.
    # A is 2x4. B is 4x8.
    # We split B into 4 blocks of 4x2.
    # For each block, we split A into two 2x2 blocks (cols 0-1 and 2-3).
    # C_block = A_left * B_top + A_right * B_bottom.
    # This requires two 2x2 multiplications.
    # Total 7 + 7 = 14 terms per block.
    # 4 blocks * 14 terms = 56 terms.
    
    col = 0
    
    for k_block in range(4):
        # B block columns: 2*k_block, 2*k_block+1
        # B rows: 0,1 for top, 2,3 for bottom
        
        # Part 1: A_left (cols 0,1) * B_top (rows 0,1)
        # A indices: 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
        # In global U (size 8):
        # A[0,0]->0, A[0,1]->1, A[1,0]->4, A[1,1]->5
        # Wait, U index is i*m + j.
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 4
        # i=1, j=1 -> 5
        
        # B indices: j*p + k.
        # j=0, k=2*kb -> 16*kb
        # j=0, k=2*kb+1 -> 16*kb+1
        # j=1, k=2*kb -> 16*kb+8
        # j=1, k=2*kb+1 -> 16*kb+9
        
        # C indices: k*n + i.
        # k=2*kb, i=0 -> 4*kb
        # k=2*kb, i=1 -> 4*kb+1
        # k=2*kb+1, i=0 -> 4*kb+2
        # k=2*kb+1, i=1 -> 4*kb+3
        
        # Map local 2x2 indices to global
        # Local A: 0->0, 1->1, 2->4, 3->5
        # Local B: 0->16*kb, 1->16*kb+1, 2->16*kb+8, 3->16*kb+9
        # Local C: 0->4*kb, 1->4*kb+1, 2->4*kb+2, 3->4*kb+3
        
        map_U_1 = {0:0, 1:1, 2:4, 3:5}
        map_V_1 = {0:16*kb, 1:16*kb+1, 2:16*kb+8, 3:16*kb+9}
        map_W_1 = {0:4*kb, 1:4*kb+1, 2:4*kb+2, 3:4*kb+3}
        
        for u_idx, u_val, v_idx, v_val, w_idx, w_val in strassen_2x2:
            for idx, val in zip(u_idx, u_val):
                U[map_U_1[idx]][col] += val
            for idx, val in zip(v_idx, v_val):
                V[map_V_1[idx]][col] += val
            for idx, val in zip(w_idx, w_val):
                W[map_W_1[idx]][col] += val
            col += 1
            
        # Part 2: A_right (cols 2,3) * B_bottom (rows 2,3)
        # A indices:
        # A[0,2]->2, A[0,3]->3, A[1,2]->6, A[1,3]->7
        # B indices:
        # j=2, k=2*kb -> 16*kb+16
        # j=2, k=2*kb+1 -> 16*kb+17
        # j=3, k=2*kb -> 16*kb+24
        # j=3, k=2*kb+1 -> 16*kb+25
        
        map_U_2 = {0:2, 1:3, 2:6, 3:7}
        map_V_2 = {0:16*kb+16, 1:16*kb+17, 2:16*kb+24, 3:16*kb+25}
        # W indices are the same as Part 1 because the output C is the same 2x2 block.
        # C = A_left*B_top + A_right*B_bottom.
        # So we add to the same W indices.
        
        for u_idx, u_val, v_idx, v_val, w_idx, w_val in strassen_2x2:
            for idx, val in zip(u_idx, u_val):
                U[map_U_2[idx]][col] += val
            for idx, val in zip(v_idx, v_val):
                V[map_V_2[idx]][col] += val
            for idx, val in zip(w_idx, w_val):
                W[map_W_1[idx]][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
