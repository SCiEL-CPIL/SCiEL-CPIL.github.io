"""Matrix multiplication 3,3,4 rank-minimal decomposition."""
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    rank = 34
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    def add_block(u_off, v_off, w_off, u_entries, v_entries, w_entries, start):
        for idx, (u_row, u_val, v_row, v_val, w_row, w_val) in enumerate(u_entries):
            col = start + idx
            U[u_off + u_row][col] = u_val
            V[v_off + v_row][col] = v_val
            W[w_off + w_row][col] = w_val

    # Strassen 2x2 multiplication entries
    # For 2x2 * 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # A11 (top-left 2x2) * B_top (2x4 = two 2x2 blocks B1, B2)
    # A11 rows 0,1 (indices 0,1 in A); B_top rows 0,1 (indices 0,1 in B)
    # Output: C top-left 2x2 (W rows for i=0,1 and k=0,1) and C top-right 2x2 (W rows for i=0,1 and k=2,3)
    
    # Block 1: A11 * B_top[:, 0:2] -> C[0:2, 0:2]
    # u_off = 0 (A rows 0,1), v_off = 0 (B rows 0,1), w_off = 0 (C rows: k*n+i)
    # C[0,0]: i=0,k=0 -> W[0]
    # C[0,1]: i=0,k=1 -> W[1]
    # C[1,0]: i=1,k=0 -> W[3]
    # C[1,1]: i=1,k=1 -> W[4]
    # Wait, W index is k*n+i. n=3.
    # C[0,0]: i=0,k=0 -> W[0*3+0]=W[0]
    # C[0,1]: i=0,k=1 -> W[1*3+0]=W[3]
    # C[1,0]: i=1,k=0 -> W[0*3+1]=W[1]
    # C[1,1]: i=1,k=1 -> W[1*3+1]=W[4]

    # B_top[:, 0:2] is B rows 0,1 and cols 0,1.
    # V indices: j*p+k. For B row j, col k.
    # B[0,0]: V[0*4+0]=V[0], B[0,1]: V[0*4+1]=V[1]
    # B[1,0]: V[1*4+0]=V[4], B[1,1]: V[1*4+1]=V[5]

    # U indices: i*m+j. For A row i, col j.
    # A[0,0]: U[0*3+0]=U[0], A[0,1]: U[0*3+1]=U[1]
    # A[1,0]: U[1*3+0]=U[3], A[1,1]: U[1*3+1]=U[4]

    strassen_entries = [
        # M1 = (A00+A11)*(B00+B11)
        (0, 1, 0, 1, 0, 1),   # C00 += 1
        (1, 1, 0, 1, 3, 1),   # C01 += 1 (wait, C[0,1] is i=0,k=1 -> W[3])
        (3, 1, 0, 1, 1, 1),   # C10 += 1 (wait, C[1,0] is i=1,k=0 -> W[1])
        (4, 1, 0, 1, 4, 1),   # C11 += 1
        # Let's re-verify Strassen formulas and W indices carefully.
    ]

    # Let's build the Strassen entries for a 2x2*2x2 -> 2x2.
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # M1 = (a00+a11)(b00+b11) -> c00+=1, c01+=1, c10+=1, c11+=1? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # M1: U: a00(1)+a11(1), V: b00(1)+b11(1)
    # M2: U: a10(1)+a11(1), V: b00(1)
    # M3: U: a00(1), V: b01(1)-b11(1)
    # M4: U: a11(1), V: b10(1)-b00(1)
    # M5: U: a00(1)+a01(1), V: b11(1)
    # M6: U: a10(1)-a00(1), V: b00(1)+b01(1)
    # M7: U: a01(1)-a11(1), V: b10(1)+b11(1)

    # For block A11*B1 (cols 0,1 of B_top):
    # U offsets: 0,1 (A rows), V offsets: 0,1 (B rows), W offsets: 0 (i=0,k=0), 3 (i=0,k=1), 1 (i=1,k=0), 4 (i=1,k=1)
    # Wait, W index = k*n + i.
    # i=0,k=0 -> 0
    # i=0,k=1 -> 3
    # i=1,k=0 -> 1
    # i=1,k=1 -> 4

    def make_strassen(u_off, v_off, c00_w, c01_w, c10_w, c11_w, start):
        entries = []
        # M1
        entries.append((0, 1, 0, 1, c00_w, 1))
        entries.append((1, 1, 0, 1, c01_w, 1))
        entries.append((3, 1, 0, 1, c10_w, 1))
        entries.append((4, 1, 0, 1, c11_w, 1))
        # M2
        entries.append((3, 1, 0, 1, c10_w, 1))
        entries.append((4, 1, 0, 1, c11_w, -1))
        # M3
        entries.append((0, 1, 1, 1, c01_w, 1))
        entries.append((0, 1, 5, -1, c01_w, 1))
        entries.append((0, 1, 4, -1, c11_w, 1))
        # M4
        entries.append((4, 1, 4, 1, c00_w, 1))
        entries.append((4, 1, 0, -1, c00_w, 1))
        entries.append((4, 1, 4, 1, c10_w, 1))
        # M5
        entries.append((0, 1, 5, 1, c00_w, -1))
        entries.append((1, 1, 5, 1, c00_w, -1))
        entries.append((0, 1, 5, 1, c01_w, 1))
        entries.append((1, 1, 5, 1, c01_w, 1))
        # M6
        entries.append((3, 1, 0, 1, c11_w, 1))
        entries.append((0, -1, 0, 1, c11_w, 1))
        entries.append((3, 1, 1, 1, c11_w, 1))
        entries.append((0, -1, 1, 1, c11_w, 1))
        # M7
        entries.append((1, 1, 4, 1, c00_w, 1))
        entries.append((4, -1, 4, 1, c00_w, 1))
        entries.append((1, 1, 5, 1, c00_w, 1))
        entries.append((4, -1, 5, 1, c00_w, 1))
        
        for idx, (u_row, u_val, v_row, v_val, w_row, w_val) in enumerate(entries):
            col = start + idx
            U[u_off + u_row][col] += u_val
            V[v_off + v_row][col] += v_val
            W[w_off + w_row][col] += w_val
            # Wait, this doesn't work because multiple entries can hit the same U/V/W cell.
            # Actually, each multiplication M is a single column in the decomposition.
            # So I need one column per M.
            pass

    # Let's do it properly: each M is one column.
    col = 0

    def add_strassen_block(u_off, v_off, w_off, b_col_off, start):
        # b_col_off is the column offset in B (0 for cols 0-1, 2 for cols 2-3)
        # w_off is the row offset in W
        # M1
        U[u_off + 0][col + 0] += 1; U[u_off + 4][col + 0] += 1
        V[v_off + 0 + b_col_off][col + 0] += 1; V[v_off + 4 + b_col_off][col + 0] += 1
        W[w_off + 0][col + 0] += 1; W[w_off + 3][col + 0] += 1; W[w_off + 1][col + 0] += 1; W[w_off + 4][col + 0] += 1
        
        # M2
        U[u_off + 3][col + 1] += 1; U[u_off + 4][col + 1] += 1
        V[v_off + 0 + b_col_off][col + 1] += 1
        W[w_off + 1][col + 1] += 1; W[w_off + 4][col + 1] -= 1
        
        # M3
        U[u_off + 0][col + 2] += 1
        V[v_off + 1 + b_col_off][col + 2] += 1; V[v_off + 5 + b_col_off][col + 2] -= 1
        W[w_off + 3][col + 2] += 1; W[w_off + 4][col + 2] += 1
        
        # M4
        U[u_off + 4][col + 3] += 1
        V[v_off + 4 + b_col_off][col + 3] += 1; V[v_off + 0 + b_col_off][col + 3] -= 1
        W[w_off + 0][col + 3] += 1; W[w_off + 1][col + 3] += 1
        
        # M5
        U[u_off + 0][col + 4] += 1; U[u_off + 1][col + 4] += 1
        V[v_off + 5 + b_col_off][col + 4] += 1
        W[w_off + 0][col + 4] -= 1; W[w_off + 3][col + 4] += 1
        
        # M6
        U[u_off + 3][col + 5] += 1; U[u_off + 0][col + 5] -= 1
        V[v_off + 0 + b_col_off][col + 5] += 1; V[v_off + 1 + b_col_off][col + 5] += 1
        W[w_off + 4][col + 5] += 1
        
        # M7
        U[u_off + 1][col + 6] += 1; U[u_off + 4][col + 6] -= 1
        V[v_off + 4 + b_col_off][col + 6] += 1; V[v_off + 5 + b_col_off][col + 6] += 1
        W[w_off + 0][col + 6] += 1
        
        return start + 7

    # A11 * B_top[:, 0:2] -> C[0:2, 0:2]
    # u_off=0, v_off=0, w_off=0, b_col_off=0
    col = add_strassen_block(0, 0, 0, 0, col)
    
    # A11 * B_top[:, 2:4] -> C[0:2, 2:4]
    # C[0,2]: i=0,k=2 -> W[6]
    # C[0,3]: i=0,k=3 -> W[9]
    # C[1,2]: i=1,k=2 -> W[7]
    # C[1,3]: i=1,k=3 -> W[10]
    # So w_off for this block: the W rows are 6,9,7,10. Not a simple offset.
    # Let's generalize: pass the W row indices for c00, c01, c10, c11.
    
    # Actually, let's just hardcode the W row indices for each block.
    # For A11*B1: W rows are 0,3,1,4 (c00,c01,c10,c11)
    # For A11*B2: W rows are 6,9,7,10
    # For A12*B1: A12 is A rows 0,1 col 2. So u_off=2. B rows 0,1. v_off=0.
    #   Output C[0:2, 0:2]. W rows 0,3,1,4.
    # For A12*B2: W rows 6,9,7,10.
    # For A21*B1: A21 is A row 2, cols 0,1. u_off=6. B rows 0,1. v_off=0.
    #   Output C[2, 0:2]. W rows: i=2,k=0 -> W[2], i=2,k=1 -> W[5].
    #   This is a 1x2 * 2x2 = 1x2. Rank 4? Or 2?
    #   y0 = a*b00 + a*b10 = a*(b00+b10). y1 = a*b01 + a*b11 = a*(b01+b11).
    #   2 multiplications.
    # For A21*B2: W rows 8, 11.
    # For A22*B1: A22 is A[2,2]. u_off=8. B rows 0,1. v_off=0.
    #   Output C[2, 0:2]. W rows 2, 5.
    #   y0 = a*b00, y1 = a*b01. 2 mults.
    # For A22*B2: W rows 8, 11.
    #   y0 = a*b20, y1 = a*b21. 2 mults.

    # Let's redo with explicit W rows.
    col = 0

    def add_strassen_2x2_2x2(u_off, v_off, w_rows, b_col_off, start):
        # w_rows = [c00_w, c01_w, c10_w, c11_w]
        c00_w, c01_w, c10_w, c11_w = w_rows
        # M1
        U[u_off+0][col] += 1; U[u_off+4][col] += 1
        V[v_off+b_col_off][col] += 1; V[v_off+4+b_col_off][col] += 1
        W[c00_w][col] += 1; W[c01_w][col] += 1; W[c10_w][col] += 1; W[c11_w][col] += 1
        col += 1
        # M2
        U[u_off+3][col] += 1; U[u_off+4][col] += 1
        V[v_off+b_col_off][col] += 1
        W[c10_w][col] += 1; W[c11_w][col] -= 1
        col += 1
        # M3
        U[u_off+0][col] += 1
        V[v_off+1+b_col_off][col] += 1; V[v_off+5+b_col_off][col] -= 1
        W[c01_w][col] += 1; W[c11_w][col] += 1
        col += 1
        # M4
        U[u_off+4][col] += 1
        V[v_off+4+b_col_off][col] += 1; V[v_off+b_col_off][col] -= 1
        W[c00_w][col] += 1; W[c10_w][col] += 1
        col += 1
        # M5
        U[u_off+0][col] += 1; U[u_off+1][col] += 1
        V[v_off+5+b_col_off][col] += 1
        W[c00_w][col] -= 1; W[c01_w][col] += 1
        col += 1
        # M6
        U[u_off+3][col] += 1; U[u_off+0][col] -= 1
        V[v_off+b_col_off][col] += 1; V[v_off+1+b_col_off][col] += 1
        W[c11_w][col] += 1
        col += 1
        # M7
        U[u_off+1][col] += 1; U[u_off+4][col] -= 1
        V[v_off+4+b_col_off][col] += 1; V[v_off+5+b_col_off][col] += 1
        W[c00_w][col] += 1
        col += 1
        return col

    # A11 * B1
    col = add_strassen_2x2_2x2(0, 0, [0,3,1,4], 0, col)
    # A11 * B2
    col = add_strassen_2x2_2x2(0, 0, [6,9,7,10], 2, col)
    # A12 * B1
    col = add_strassen_2x2_2x2(2, 0, [0,3,1,4], 0, col)
    # A12 * B2
    col = add_strassen_2x2_2x2(2, 0, [6,9,7,10], 2, col)

    # A21 * B1: A row 2, cols 0,1 (u_off=6). B rows 0,1 (v_off=0). C row 2, cols 0,1.
    # y0 = a*(b00+b10), y1 = a*(b01+b11)
    # W rows: i=2,k=0 -> 2, i=2,k=1 -> 5
    U[6][col] += 1; V[0][col] += 1; V[4][col] += 1; W[2][col] += 1
    col += 1
    U[6][col] += 1; V[1][col] += 1; V[5][col] += 1; W[5][col] += 1
    col += 1

    # A21 * B2
    U[6][col] += 1; V[2][col] += 1; V[6][col] += 1; W[8][col] += 1
    col += 1
    U[6][col] += 1; V[3][col] += 1; V[7][col] += 1; W[11][col] += 1
    col += 1

    # A22 * B1: A[2,2] (u_off=8). B rows 0,1. C row 2, cols 0,1.
    # y0 = a*b00, y1 = a*b01
    U[8][col] += 1; V[0][col] += 1; W[2][col] += 1
    col += 1
    U[8][col] += 1; V[1][col] += 1; W[5][col] += 1
    col += 1

    # A22 * B2
    U[8][col] += 1; V[2][col] += 1; W[8][col] += 1
    col += 1
    U[8][col] += 1; V[3][col] += 1; W[11][col] += 1
    col += 1

    assert col == 34, f"Expected 34, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
