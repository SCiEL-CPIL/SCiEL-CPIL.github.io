"""Matrix multiplication 3x4 by 4x6 - rank 66 decomposition."""

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Rank 66 decomposition
    # Split m=4 into two blocks of size 2.
    # A = [A1, A2] where A1, A2 are 3x2.
    # B = [B1; B2] where B1, B2 are 2x6.
    # C = A1 B1 + A2 B2.
    # Rank(A1 B1) <= 33. Rank(A2 B2) <= 33.
    # Total rank 66.
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)] # 12 x 66
    V = [[0] * rank for _ in range(m * p)] # 24 x 66
    W = [[0] * rank for _ in range(p * n)] # 18 x 66
    
    # Helper to add a rank-1 term
    # u_idx, v_idx, w_idx, col
    # u_idx is index in U (i*m + j)
    # v_idx is index in V (j*p + k)
    # w_idx is index in W (k*n + i)
    
    def add_term(u_idx, v_idx, w_idx, col, val_u=1, val_v=1, val_w=1):
        U[u_idx][col] += val_u
        V[v_idx][col] += val_v
        W[w_idx][col] += val_w

    col = 0
    
    # We need to define the 11-term decomposition for 3x2x2.
    # Inputs: A (3x2), B (2x2)
    # A indices: (0,0), (0,1), (1,0), (1,1), (2,0), (2,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1), (2,0), (2,1)
    
    # Strassen for top 2x2 of A:
    # A_top = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Schoolbook for bottom 1x2 of A:
    # A_bot = [a20, a21]
    # C20 = a20 b00 + a21 b10
    # C21 = a20 b01 + a21 b11
    # Terms:
    # M8 = a20 b00
    # M9 = a21 b10
    # M10 = a20 b01
    # M11 = a21 b11
    
    # Total 11 terms.
    
    # We need to map these to the global indices.
    # Global A is 3x4.
    # Block 1: A1 (cols 0,1). A2 (cols 2,3).
    # Global B is 4x6.
    # Block 1: B1 (rows 0,1). B2 (rows 2,3).
    
    # For each block of A (2 blocks) and each block of B (3 blocks of 2x2 within the 2x6),
    # we add 11 terms.
    # Total 2 * 3 * 11 = 66 terms.
    
    for block_a in range(2):
        a_base = block_a * 2 # 0 or 2
        # A rows are 0, 1, 2.
        # A indices in U: i*m + j.
        # i is 0..2, j is a_base, a_base+1.
        
        for block_b in range(3):
            b_base = block_b * 2 # 0, 2, 4
            # B cols are 0..5.
            # B indices in V: j*p + k.
            # j is 0, 1 (relative to block? No, global rows).
            # Wait, B is 4x6.
            # We split B into B1 (rows 0,1) and B2 (rows 2,3).
            # So for block_a=0, we use B1 (rows 0,1).
            # For block_a=1, we use B2 (rows 2,3).
            
            b_row_base = block_a * 2 # 0 or 2
            
            # Now we process the 2x2 block of B at rows [b_row_base, b_row_base+1] and cols [b_base, b_base+1].
            
            # Term 1: (a00+a11)(b00+b11)
            # u: a00 (i=0, j=a_base), a11 (i=1, j=a_base+1)
            # v: b00 (j=b_row_base, k=b_base), b11 (j=b_row_base+1, k=b_base+1)
            # w: c00 (i=0, k=b_base), c11 (i=1, k=b_base+1)
            
            # We need to add these to the correct columns.
            # Since we are adding rank-1 terms, we just set the values.
            
            # Helper to get U index
            def u_idx(i, j): return i * m + j
            # Helper to get V index
            def v_idx(j, k): return j * p + k
            # Helper to get W index
            def w_idx(k, i): return k * n + i
            
            # Term 1
            U[u_idx(0, a_base)][col] += 1
            U[u_idx(1, a_base+1)][col] += 1
            V[v_idx(b_row_base, b_base)][col] += 1
            V[v_idx(b_row_base+1, b_base+1)][col] += 1
            W[w_idx(b_base, 0)][col] += 1
            W[w_idx(b_base+1, 1)][col] += 1
            col += 1
            
            # Term 2: (a10+a11)b00
            U[u_idx(1, a_base)][col] += 1
            U[u_idx(1, a_base+1)][col] += 1
            V[v_idx(b_row_base, b_base)][col] += 1
            W[w_idx(b_base, 1)][col] += 1
            col += 1
            
            # Term 3: a00(b10-b11)
            U[u_idx(0, a_base)][col] += 1
            V[v_idx(b_row_base+1, b_base)][col] += 1
            V[v_idx(b_row_base+1, b_base+1)][col] -= 1
            W[w_idx(b_base, 0)][col] += 1
            W[w_idx(b_base+1, 0)][col] -= 1
            col += 1
            
            # Term 4: a11(b01-b00)
            U[u_idx(1, a_base+1)][col] += 1
            V[v_idx(b_row_base, b_base+1)][col] += 1
            V[v_idx(b_row_base, b_base)][col] -= 1
            W[w_idx(b_base+1, 1)][col] += 1
            W[w_idx(b_base, 1)][col] -= 1
            col += 1
            
            # Term 5: (a00+a01)b11
            U[u_idx(0, a_base)][col] += 1
            U[u_idx(0, a_base+1)][col] += 1
            V[v_idx(b_row_base+1, b_base+1)][col] += 1
            W[w_idx(b_base, 0)][col] += 1
            W[w_idx(b_base+1, 0)][col] += 1
            col += 1
            
            # Term 6: (a10-a00)(b00+b01)
            U[u_idx(1, a_base)][col] += 1
            U[u_idx(0, a_base)][col] -= 1
            V[v_idx(b_row_base, b_base)][col] += 1
            V[v_idx(b_row_base, b_base+1)][col] += 1
            W[w_idx(b_base, 1)][col] += 1
            W[w_idx(b_base+1, 1)][col] += 1
            col += 1
            
            # Term 7: (a01-a11)(b10+b11)
            U[u_idx(0, a_base+1)][col] += 1
            U[u_idx(1, a_base+1)][col] -= 1
            V[v_idx(b_row_base+1, b_base)][col] += 1
            V[v_idx(b_row_base+1, b_base+1)][col] += 1
            W[w_idx(b_base, 0)][col] += 1
            W[w_idx(b_base+1, 0)][col] += 1
            col += 1
            
            # Term 8: a20 b00
            U[u_idx(2, a_base)][col] += 1
            V[v_idx(b_row_base, b_base)][col] += 1
            W[w_idx(b_base, 2)][col] += 1
            col += 1
            
            # Term 9: a21 b10
            U[u_idx(2, a_base+1)][col] += 1
            V[v_idx(b_row_base+1, b_base)][col] += 1
            W[w_idx(b_base, 2)][col] += 1
            col += 1
            
            # Term 10: a20 b01
            U[u_idx(2, a_base)][col] += 1
            V[v_idx(b_row_base, b_base+1)][col] += 1
            W[w_idx(b_base+1, 2)][col] += 1
            col += 1
            
            # Term 11: a21 b11
            U[u_idx(2, a_base+1)][col] += 1
            V[v_idx(b_row_base+1, b_base+1)][col] += 1
            W[w_idx(b_base+1, 2)][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
