"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition for 2x4x8
    # Based on block structure:
    # A (2x4) = [A1, A2] (2x2 blocks)
    # B (4x8) = [B_top, B_bot] (2x8 blocks)
    # C (2x8) = A1 B_top + A2 B_bot
    # Rank(M_2x2x8) = 28
    # Total Rank = 28 + 28 = 56
    
    rank = 56
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Helper to add a rank-7 Strassen-like block for 2x2x2 multiplication
    # Indices:
    # U: i*m+j
    # V: j*p+k
    # W: k*n+i
    
    # We need to map the logical blocks to the global indices.
    # A1 is rows 0-1, cols 0-1 of A (i in 0..1, j in 0..1)
    # A2 is rows 0-1, cols 2-3 of A (i in 0..1, j in 2..3)
    # B_top is rows 0-1 of B (j in 0..1)
    # B_bot is rows 2-3 of B (j in 2..3)
    
    # M_2x2x8 decomposition:
    # A (2x2), B (2x8), C (2x8)
    # B is split into 4 blocks of 2x2: B1, B2, B3, B4
    # C is split into 4 blocks of 2x2: C1, C2, C3, C4
    # Ci = A Bi
    # Each is M_2x2x2 (rank 7)
    
    def add_strassen_block(u_offset, v_offset, w_offset, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Adds a 2x2x2 Strassen decomposition.
        u_offset: start index in U (n*m)
        v_offset: start index in V (m*p)
        w_offset: start index in W (p*n)
        
        a_row_offset: row offset in A (0 or 1 for 2x2 block)
        a_col_offset: col offset in A (0, 2, etc)
        b_row_offset: row offset in B (0, 2, etc)
        b_col_offset: col offset in B (0, 2, etc)
        c_row_offset: row offset in C
        c_col_offset: col offset in C
        """
        # Strassen formulas for 2x2x2
        # A = [[a, b], [c, d]]
        # B = [[e, f], [g, h]]
        # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        
        # 7 multiplications:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # C11 = m1 - m4 + m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m3 + m6 - m7
        
        # We need to map these to U, V, W entries.
        # U[i*m+j, col]
        # V[j*p+k, col]
        # W[k*n+i, col]
        
        # A indices:
        # a: i=0, j=a_col_offset
        # b: i=0, j=a_col_offset+1
        # c: i=1, j=a_col_offset
        # d: i=1, j=a_col_offset+1
        
        # B indices:
        # e: j=b_row_offset, k=b_col_offset
        # f: j=b_row_offset, k=b_col_offset+1
        # g: j=b_row_offset+1, k=b_col_offset
        # h: j=b_row_offset+1, k=b_col_offset+1
        
        # C indices:
        # C11: k=c_col_offset, i=c_row_offset
        # C12: k=c_col_offset+1, i=c_row_offset
        # C21: k=c_col_offset, i=c_row_offset+1
        # C22: k=c_col_offset+1, i=c_row_offset+1
        
        def u_idx(i, j): return i*m + j
        def v_idx(j, k): return j*p + k
        def w_idx(k, i): return k*n + i
        
        # m1
        col = w_offset
        # (a+d) -> U[0,a]+U[1,a+1]
        U[u_idx(0, a_col_offset)][col] += 1
        U[u_idx(1, a_col_offset+1)][col] += 1
        # (e+h) -> V[0,e]+V[1,h]
        V[v_idx(b_row_offset, b_col_offset)][col] += 1
        V[v_idx(b_row_offset+1, b_col_offset+1)][col] += 1
        # C11 += m1
        W[w_idx(c_col_offset, c_row_offset)][col] += 1
        # C22 += m1
        W[w_idx(c_col_offset+1, c_row_offset+1)][col] += 1
        
        # m2
        col = w_offset + 1
        # (c+d) -> U[1,a]+U[1,a+1]
        U[u_idx(1, a_col_offset)][col] += 1
        U[u_idx(1, a_col_offset+1)][col] += 1
        # e -> V[0,e]
        V[v_idx(b_row_offset, b_col_offset)][col] += 1
        # C21 += m2
        W[w_idx(c_col_offset, c_row_offset+1)][col] += 1
        
        # m3
        col = w_offset + 2
        # a -> U[0,a]
        U[u_idx(0, a_col_offset)][col] += 1
        # (f-h) -> V[0,f]-V[1,h]
        V[v_idx(b_row_offset, b_col_offset+1)][col] += 1
        V[v_idx(b_row_offset+1, b_col_offset+1)][col] -= 1
        # C12 += m3
        W[w_idx(c_col_offset+1, c_row_offset)][col] += 1
        # C22 -= m3
        W[w_idx(c_col_offset+1, c_row_offset+1)][col] -= 1
        
        # m4
        col = w_offset + 3
        # d -> U[1,a+1]
        U[u_idx(1, a_col_offset+1)][col] += 1
        # (g-e) -> V[1,g]-V[0,e]
        V[v_idx(b_row_offset+1, b_col_offset)][col] += 1
        V[v_idx(b_row_offset, b_col_offset)][col] -= 1
        # C11 -= m4
        W[w_idx(c_col_offset, c_row_offset)][col] -= 1
        # C21 += m4
        W[w_idx(c_col_offset, c_row_offset+1)][col] += 1
        
        # m5
        col = w_offset + 4
        # (a+b) -> U[0,a]+U[0,a+1]
        U[u_idx(0, a_col_offset)][col] += 1
        U[u_idx(0, a_col_offset+1)][col] += 1
        # h -> V[1,h]
        V[v_idx(b_row_offset+1, b_col_offset+1)][col] += 1
        # C11 += m5
        W[w_idx(c_col_offset, c_row_offset)][col] += 1
        # C12 += m5
        W[w_idx(c_col_offset+1, c_row_offset)][col] += 1
        
        # m6
        col = w_offset + 5
        # (c-a) -> U[1,a]-U[0,a]
        U[u_idx(1, a_col_offset)][col] += 1
        U[u_idx(0, a_col_offset)][col] -= 1
        # (e+f) -> V[0,e]+V[0,f]
        V[v_idx(b_row_offset, b_col_offset)][col] += 1
        V[v_idx(b_row_offset, b_col_offset+1)][col] += 1
        # C22 += m6
        W[w_idx(c_col_offset+1, c_row_offset+1)][col] += 1
        
        # m7
        col = w_offset + 6
        # (b-d) -> U[0,a+1]-U[1,a+1]
        U[u_idx(0, a_col_offset+1)][col] += 1
        U[u_idx(1, a_col_offset+1)][col] -= 1
        # (g+h) -> V[1,g]+V[1,h]
        V[v_idx(b_row_offset+1, b_col_offset)][col] += 1
        V[v_idx(b_row_offset+1, b_col_offset+1)][col] += 1
        # C11 += m7
        W[w_idx(c_col_offset, c_row_offset)][col] += 1
        # C22 -= m7
        W[w_idx(c_col_offset+1, c_row_offset+1)][col] -= 1

    # Block 1: A1 (cols 0-1) with B_top (rows 0-1)
    # B_top is 2x8, split into 4 blocks of 2x2
    # B_top,0: rows 0-1, cols 0-1
    # B_top,1: rows 0-1, cols 2-3
    # B_top,2: rows 0-1, cols 4-5
    # B_top,3: rows 0-1, cols 6-7
    
    # C is 2x8.
    # C_top,0: rows 0-1, cols 0-1 (from A1 B_top,0)
    # C_top,1: rows 0-1, cols 2-3 (from A1 B_top,1)
    # ...
    
    # A1 is at a_row_offset=0, a_col_offset=0
    # B_top is at b_row_offset=0
    
    # Add 4 Strassen blocks for A1 * B_top
    for k in range(4):
        b_col = k * 2
        c_col = k * 2
        add_strassen_block(0, 0, k*7, 0, 0, 0, b_col, 0, c_col)
        
    # Block 2: A2 (cols 2-3) with B_bot (rows 2-3)
    # A2 is at a_row_offset=0, a_col_offset=2
    # B_bot is at b_row_offset=2
    
    # Add 4 Strassen blocks for A2 * B_bot
    for k in range(4):
        b_col = 2 + k * 2
        c_col = k * 2
        add_strassen_block(0, 0, 28 + k*7, 0, 2, 2, b_col, 0, c_col)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
