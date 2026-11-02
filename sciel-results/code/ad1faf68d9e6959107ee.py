"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = 2, 4, 7
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to add a 2x2 by 2x2 Strassen decomposition
    # a_rows: [r1, r2] indices in A
    # a_cols: [c1, c2] indices in A
    # b_rows: [r1, r2] indices in B
    # b_cols: [c1, c2] indices in B
    # w_k: [k1, k2] indices in W (output columns)
    # w_i: [i1, i2] indices in W (output rows)
    # col_offset: starting column index in U, V, W
    def add_strassen_22(a_rows, a_cols, b_rows, b_cols, w_k, w_i, col_offset):
        # P1 = (X00+X11)(Y00+Y11)
        U[a_rows[0]*m + a_cols[0]][col_offset] += 1
        U[a_rows[1]*m + a_cols[1]][col_offset] += 1
        V[b_rows[0]*p + b_cols[0]][col_offset] += 1
        V[b_rows[1]*p + b_cols[1]][col_offset] += 1
        W[w_k[0]*n + w_i[0]][col_offset] += 1 # Z00 += P1
        W[w_k[1]*n + w_i[0]][col_offset] += 1 # Z01? No, Z01 is P3+P5. 
        # Wait, W index is k*n + i.
        # Z00 is at i=0, k=k0. Z01 is at i=0, k=k1.
        # Z10 is at i=1, k=k0. Z11 is at i=1, k=k1.
        
        # Let's map W indices correctly.
        # w_k[0] corresponds to col k0 of output.
        # w_i[0] corresponds to row i0 of output.
        # So W[w_k[0]*n + w_i[0]] is Z[i0, k0].
        
        # P1 contributes to Z00 and Z11.
        # Z00: i=0, k=k0 -> W[w_k[0]*n + w_i[0]]
        # Z11: i=1, k=k1 -> W[w_k[1]*n + w_i[1]]
        W[w_k[0]*n + w_i[0]][col_offset] += 1
        W[w_k[1]*n + w_i[1]][col_offset] += 1
        
        # P2 = (X10+X11)Y00
        U[a_rows[1]*m + a_cols[0]][col_offset+1] += 1
        U[a_rows[1]*m + a_cols[1]][col_offset+1] += 1
        V[b_rows[0]*p + b_cols[0]][col_offset+1] += 1
        # P2 contributes to Z10.
        # Z10: i=1, k=k0 -> W[w_k[0]*n + w_i[1]]
        W[w_k[0]*n + w_i[1]][col_offset+1] += 1

        # P3 = X00(Y01-Y11)
        U[a_rows[0]*m + a_cols[0]][col_offset+2] += 1
        V[b_rows[0]*p + b_cols[1]][col_offset+2] += 1
        V[b_rows[1]*p + b_cols[1]][col_offset+2] -= 1
        # P3 contributes to Z01 and Z11.
        # Z01: i=0, k=k1 -> W[w_k[1]*n + w_i[0]]
        # Z11: i=1, k=k1 -> W[w_k[1]*n + w_i[1]]
        W[w_k[1]*n + w_i[0]][col_offset+2] += 1
        W[w_k[1]*n + w_i[1]][col_offset+2] += 1

        # P4 = X11(Y10-Y00)
        U[a_rows[1]*m + a_cols[1]][col_offset+3] += 1
        V[b_rows[1]*p + b_cols[0]][col_offset+3] += 1
        V[b_rows[0]*p + b_cols[0]][col_offset+3] -= 1
        # P4 contributes to Z00 and Z10.
        # Z00: i=0, k=k0 -> W[w_k[0]*n + w_i[0]]
        # Z10: i=1, k=k0 -> W[w_k[0]*n + w_i[1]]
        W[w_k[0]*n + w_i[0]][col_offset+3] += 1
        W[w_k[0]*n + w_i[1]][col_offset+3] += 1

        # P5 = (X00+X01)Y11
        U[a_rows[0]*m + a_cols[0]][col_offset+4] += 1
        U[a_rows[0]*m + a_cols[1]][col_offset+4] += 1
        V[b_rows[1]*p + b_cols[1]][col_offset+4] += 1
        # P5 contributes to Z01 and Z00.
        # Z01: i=0, k=k1 -> W[w_k[1]*n + w_i[0]]
        # Z00: i=0, k=k0 -> W[w_k[0]*n + w_i[0]] (with -1)
        W[w_k[1]*n + w_i[0]][col_offset+4] += 1
        W[w_k[0]*n + w_i[0]][col_offset+4] -= 1

        # P6 = (X10-X00)(Y00+Y01)
        U[a_rows[1]*m + a_cols[0]][col_offset+5] += 1
        U[a_rows[0]*m + a_cols[0]][col_offset+5] -= 1
        V[b_rows[0]*p + b_cols[0]][col_offset+5] += 1
        V[b_rows[0]*p + b_cols[1]][col_offset+5] += 1
        # P6 contributes to Z11 (with -1).
        # Z11: i=1, k=k1 -> W[w_k[1]*n + w_i[1]]
        W[w_k[1]*n + w_i[1]][col_offset+5] -= 1

        # P7 = (X01-X11)(Y10+Y11)
        U[a_rows[0]*m + a_cols[1]][col_offset+6] += 1
        U[a_rows[1]*m + a_cols[1]][col_offset+6] -= 1
        V[b_rows[1]*p + b_cols[0]][col_offset+6] += 1
        V[b_rows[1]*p + b_cols[1]][col_offset+6] += 1
        # P7 contributes to Z00 and Z11.
        # Z00: i=0, k=k0 -> W[w_k[0]*n + w_i[0]]
        # Z11: i=1, k=k1 -> W[w_k[1]*n + w_i[1]] (with -1)
        W[w_k[0]*n + w_i[0]][col_offset+6] += 1
        W[w_k[1]*n + w_i[1]][col_offset+6] -= 1

    # Helper to add a 2x2 by 2x1 schoolbook decomposition
    def add_schoolbook_221(a_rows, a_cols, b_rows, b_col, w_k, w_i, col_offset):
        # Z0 = X00 Y0 + X01 Y1
        U[a_rows[0]*m + a_cols[0]][col_offset] += 1
        V[b_rows[0]*p + b_col][col_offset] += 1
        W[w_k*n + w_i[0]][col_offset] += 1

        U[a_rows[0]*m + a_cols[1]][col_offset+1] += 1
        V[b_rows[1]*p + b_col][col_offset+1] += 1
        W[w_k*n + w_i[0]][col_offset+1] += 1

        # Z1 = X10 Y0 + X11 Y1
        U[a_rows[1]*m + a_cols[0]][col_offset+2] += 1
        V[b_rows[0]*p + b_col][col_offset+2] += 1
        W[w_k*n + w_i[1]][col_offset+2] += 1

        U[a_rows[1]*m + a_cols[1]][col_offset+3] += 1
        V[b_rows[1]*p + b_col][col_offset+3] += 1
        W[w_k*n + w_i[1]][col_offset+3] += 1

    # Block 1: A1 (rows 0,1 cols 0,1) * B1 (rows 0,1)
    # B1 cols 0,1
    add_strassen_22([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
    # B1 cols 2,3
    add_strassen_22([0,1], [0,1], [0,1], [2,3], [2,3], [0,1], 7)
    # B1 cols 4,5
    add_strassen_22([0,1], [0,1], [0,1], [4,5], [4,5], [0,1], 14)
    # B1 col 6
    add_schoolbook_221([0,1], [0,1], [0,1], 6, 6, [0,1], 21)

    # Block 2: A2 (rows 0,1 cols 2,3) * B2 (rows 2,3)
    # B2 cols 0,1
    add_strassen_22([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], 25)
    # B2 cols 2,3
    add_strassen_22([0,1], [2,3], [2,3], [2,3], [2,3], [0,1], 32)
    # B2 cols 4,5
    add_strassen_22([0,1], [2,3], [2,3], [4,5], [4,5], [0,1], 39)
    # B2 col 6
    add_schoolbook_221([0,1], [2,3], [2,3], 6, 6, [0,1], 46)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
