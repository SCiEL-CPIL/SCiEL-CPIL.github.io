import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal current_rank
        if u_val != 0:
            U[u_idx][current_rank] += u_val
        if v_val != 0:
            V[v_idx][current_rank] += v_val
        if w_val != 0:
            W[w_idx][current_rank] += w_val
        current_rank += 1

    def add_block(j_a_offset, j_b_offset, k_b_offset):
        # j_a_offset: 0 or 2 (column offset in A)
        # j_b_offset: 0 or 2 (row offset in B)
        # k_b_offset: 0 or 2 (col offset in B)
        
        # Strassen part for rows 0, 1 of A
        # A indices: i in {0,1}, j in {j_a_offset, j_a_offset+1}
        # B indices: j in {j_b_offset, j_b_offset+1}, k in {k_b_offset, k_b_offset+1}
        
        # Local indices for Strassen:
        # a00 = A[0, j_a_offset]
        # a01 = A[0, j_a_offset+1]
        # a10 = A[1, j_a_offset]
        # a11 = A[1, j_a_offset+1]
        
        # b00 = B[j_b_offset, k_b_offset]
        # b01 = B[j_b_offset, k_b_offset+1]
        # b10 = B[j_b_offset+1, k_b_offset]
        # b11 = B[j_b_offset+1, k_b_offset+1]
        
        # Output C indices:
        # i in {0,1}, k in {k_b_offset, k_b_offset+1}
        
        # Term 1: (a00 + a11)(b00 + b11)
        # U: A[0, ja], A[1, ja+1]
        # V: B[jb, kb], B[jb+1, kb+1]
        # W: C[0, kb] += 1, C[1, kb+1] += -1
        
        u_indices = [0 * m + j_a_offset, 1 * m + (j_a_offset + 1)]
        v_indices = [j_b_offset * p + k_b_offset, (j_b_offset + 1) * p + (k_b_offset + 1)]
        w_indices = [k_b_offset * n + 0, (k_b_offset + 1) * n + 1]
        w_vals = [1, -1]
        
        # We need to add multiple entries to U, V, W for a single rank term.
        # The add_term function above adds to a single column.
        # But Strassen terms involve sums.
        # So we need to accumulate into the current column.
        
        # Let's refactor: instead of add_term, we just modify U, V, W directly for the current_rank.
        
        # Reset current_rank logic? No, just use current_rank index.
        
        # Term 1
        col = current_rank
        U[0 * m + j_a_offset][col] += 1
        U[1 * m + (j_a_offset + 1)][col] += 1
        V[j_b_offset * p + k_b_offset][col] += 1
        V[(j_b_offset + 1) * p + (k_b_offset + 1)][col] += 1
        W[k_b_offset * n + 0][col] += 1
        W[(k_b_offset + 1) * n + 1][col] += -1
        current_rank += 1
        
        # Term 2: (a10 + a11)b00
        col = current_rank
        U[1 * m + j_a_offset][col] += 1
        U[1 * m + (j_a_offset + 1)][col] += 1
        V[j_b_offset * p + k_b_offset][col] += 1
        W[k_b_offset * n + 1][col] += 1 # C[1, kb]
        W[(k_b_offset + 1) * n + 1][col] += 1 # C[1, kb+1]
        current_rank += 1
        
        # Term 3: a00(b01 - b11)
        col = current_rank
        U[0 * m + j_a_offset][col] += 1
        V[j_b_offset * p + (k_b_offset + 1)][col] += 1
        V[(j_b_offset + 1) * p + (k_b_offset + 1)][col] += -1
        W[(k_b_offset + 1) * n + 0][col] += 1 # C[0, kb+1]
        current_rank += 1
        
        # Term 4: a11(b10 - b00)
        col = current_rank
        U[1 * m + (j_a_offset + 1)][col] += 1
        V[(j_b_offset + 1) * p + k_b_offset][col] += 1
        V[j_b_offset * p + k_b_offset][col] += -1
        W[k_b_offset * n + 0][col] += -1 # C[0, kb]
        W[k_b_offset * n + 1][col] += 1 # C[1, kb]
        current_rank += 1
        
        # Term 5: (a00 + a01)b11
        col = current_rank
        U[0 * m + j_a_offset][col] += 1
        U[0 * m + (j_a_offset + 1)][col] += 1
        V[(j_b_offset + 1) * p + (k_b_offset + 1)][col] += 1
        W[(k_b_offset + 1) * n + 0][col] += 1 # C[0, kb+1]
        W[(k_b_offset + 1) * n + 1][col] += 1 # C[1, kb+1]
        current_rank += 1
        
        # Term 6: (a01 - a10)(b10 + b11)
        col = current_rank
        U[0 * m + (j_a_offset + 1)][col] += 1
        U[1 * m + j_a_offset][col] += -1
        V[(j_b_offset + 1) * p + k_b_offset][col] += 1
        V[(j_b_offset + 1) * p + (k_b_offset + 1)][col] += 1
        W[k_b_offset * n + 0][col] += 1 # C[0, kb]
        current_rank += 1
        
        # Term 7: (a00 - a10)(b00 + b10)
        col = current_rank
        U[0 * m + j_a_offset][col] += 1
        U[1 * m + j_a_offset][col] += -1
        V[j_b_offset * p + k_b_offset][col] += 1
        V[(j_b_offset + 1) * p + k_b_offset][col] += 1
        W[k_b_offset * n + 0][col] += -1 # C[0, kb]
        W[(k_b_offset + 1) * n + 1][col] += 1 # C[1, kb+1]
        current_rank += 1
        
        # Trivial part for row 2 of A
        # a20 = A[2, j_a_offset]
        # a21 = A[2, j_a_offset+1]
        
        # Term 8: a20 b00
        col = current_rank
        U[2 * m + j_a_offset][col] += 1
        V[j_b_offset * p + k_b_offset][col] += 1
        W[k_b_offset * n + 2][col] += 1 # C[2, kb]
        current_rank += 1
        
        # Term 9: a21 b10
        col = current_rank
        U[2 * m + (j_a_offset + 1)][col] += 1
        V[(j_b_offset + 1) * p + k_b_offset][col] += 1
        W[k_b_offset * n + 2][col] += 1 # C[2, kb]
        current_rank += 1
        
        # Term 10: a20 b01
        col = current_rank
        U[2 * m + j_a_offset][col] += 1
        V[j_b_offset * p + (k_b_offset + 1)][col] += 1
        W[(k_b_offset + 1) * n + 2][col] += 1 # C[2, kb+1]
        current_rank += 1
        
        # Term 11: a21 b11
        col = current_rank
        U[2 * m + (j_a_offset + 1)][col] += 1
        V[(j_b_offset + 1) * p + (k_b_offset + 1)][col] += 1
        W[(k_b_offset + 1) * n + 2][col] += 1 # C[2, kb+1]
        current_rank += 1

    # Block 0: A_left (j=0,1), B_00 (j=0,1, k=0,1)
    add_block(0, 0, 0)
    
    # Block 1: A_left (j=0,1), B_01 (j=0,1, k=2,3)
    add_block(0, 0, 2)
    
    # Block 2: A_right (j=2,3), B_10 (j=2,3, k=0,1)
    add_block(2, 2, 0)
    
    # Block 3: A_right (j=2,3), B_11 (j=2,3, k=2,3)
    add_block(2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
