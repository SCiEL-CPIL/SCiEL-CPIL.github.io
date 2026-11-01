import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 13

    # ---------- Schoolbook fallback (rank 52), guaranteed valid ----------
    def schoolbook():
        rank = n * m * p  # 52
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    U_fb, V_fb, W_fb = schoolbook()

    # ---------- Strassen-style structured decomposition ----------
    # Split p columns into pairs (0,1),(2,3),...,(10,11) plus single column 12.
    # For each 2-column chunk t (columns 2t, 2t+1), use Strassen rank-7.
    # For the single last column, use 4 direct terms.
    # Total rank = 6*7 + 4 = 46.

    def build_strassen():
        rank = 6 * 7 + 4  # 46
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Strassen factors for 2x2 @ 2x2 (A fixed, B varies per chunk).
        # A indices: 00,01,10,11 -> flat 0,1,2,3
        # B chunk indices: b00,b01,b10,b11 (rows 0,1; cols 0,1 within chunk)
        # C chunk indices: c00,c01,c10,c11
        #
        # p1 = (A00+A11)*(B00+B11)
        # p2 = (A10+A11)*B00
        # p3 = A00*(B10-B00)
        # p4 = A11*(B01-B11)
        # p5 = (A00+A01)*B11
        # p6 = (A10-A00)*(B00+B01)
        # p7 = (A01-A11)*(B10+B11)
        #
        # C00 = p1 + p4 - p5 + p7
        # C01 = p3 + p5
        # C10 = p2 + p4
        # C11 = p1 - p2 + p3 + p6

        # Strassen factor definitions: (U_entries, V_entries, W_entries)
        # Each entry is a list of (flat_index, coeff).
        A00, A01, A10, A11 = 0, 1, 2, 3

        strassen_factors = [
            # p1 = (A00+A11)*(B00+B11)
            ([(A00, 1), (A11, 1)], [(0, 1), (1, 1)],
             [(0, 1), (3, 1)]),  # C00 +1, C11 +1
            # p2 = (A10+A11)*B00
            ([(A10, 1), (A11, 1)], [(0, 1)],
             [(2, 1)]),  # C10 +1
            # p3 = A00*(B10-B00)
            ([(A00, 1)], [(1, 1), (0, -1)],
             [(1, 1), (3, 1)]),  # C01 +1, C11 +1
            # p4 = A11*(B01-B11)
            ([(A11, 1)], [(0, 1), (1, -1)],
             [(0, 1), (2, 1)]),  # C00 +1, C10 +1
            # p5 = (A00+A01)*B11
            ([(A00, 1), (A01, 1)], [(1, 1)],
             [(0, -1), (1, 1)]),  # C00 -1, C01 +1
            # p6 = (A10-A00)*(B00+B01)
            ([(A10, 1), (A00, -1)], [(0, 1), (0, 1)],
             [(3, 1)]),  # C11 +1
            # p7 = (A01-A11)*(B10+B11)
            ([(A01, 1), (A11, -1)], [(1, 1), (1, 1)],
             [(0, 1), (2, -1)]),  # C00 +1, C10 -1
        ]

        # 6 Strassen chunks for column pairs (0,1),(2,3),...,(10,11)
        for t in range(6):
            col_base = 2 * t  # first column of this chunk
            for (u_ent, v_ent, w_ent) in strassen_factors:
                # U entries: flat index in A (0..3)
                for (a_idx, coeff) in u_ent:
                    U[a_idx][col] += coeff
                # V entries: (row, col_within_chunk) -> flat j*p + (col_base + c)
                for (b_row, coeff) in v_ent:
                    # b_row is the row of B (0 or 1); the column within chunk
                    # is encoded: for p1 we need B00 and B11, etc.
                    # v_ent entries are (col_within_chunk, coeff) where
                    # col_within_chunk: 0 -> B[0, col_base], 1 -> B[1, col_base+1]
                    # Wait, I need to be more careful.
                    pass
                col += 1
        # The above V encoding is getting tangled. Let me rebuild with explicit
        # (row, col) pairs for V.
        return None

    # Rebuild with explicit (row, col) for V and (row, col) for W.
    def build_strassen_v2():
        rank = 6 * 7 + 4  # 46
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # A flat: 00->0, 01->1, 10->2, 11->3
        # B flat: row*p + col
        # C flat (trace convention): col*n + row

        A00, A01, A10, A11 = 0, 1, 2, 3

        # Each Strassen factor: list of (U_flat, coeff), (B_row, B_col, coeff), (C_row, C_col, coeff)
        # B_col is the absolute column; C_col is the absolute column.
        def sf(col_base):
            cb = col_base
            # p1 = (A00+A11)*(B00+B11)
            f1 = ([(A00, 1), (A11, 1)],
                  [(0, cb, 1), (1, cb + 1, 1)],
                  [(0, cb, 1), (1, cb + 1, 1)])
            # p2 = (A10+A11)*B00
            f2 = ([(A10, 1), (A11, 1)],
                  [(0, cb, 1)],
                  [(1, cb, 1)])
            # p3 = A00*(B10-B00)
            f3 = ([(A00, 1)],
                  [(1, cb, 1), (0, cb, -1)],
                  [(0, cb + 1, 1), (1, cb + 1, 1)])
            # p4 = A11*(B01-B11)
            f4 = ([(A11, 1)],
                  [(0, cb + 1, 1), (1, cb + 1, -1)],
                  [(0, cb, 1), (1, cb, 1)])
            # p5 = (A00+A01)*B11
            f5 = ([(A00, 1), (A01, 1)],
                  [(1, cb + 1, 1)],
                  [(0, cb, -1), (0, cb + 1, 1)])
            # p6 = (A10-A00)*(B00+B01)
            f6 = ([(A10, 1), (A00, -1)],
                  [(0, cb, 1), (0, cb + 1, 1)],
                  [(1, cb + 1, 1)])
            # p7 = (A01-A11)*(B10+B11)
            f7 = ([(A01, 1), (A11, -1)],
                  [(1, cb, 1), (1, cb + 1, 1)],
                  [(0, cb, 1), (1, cb, -1)])
            return [f1, f2, f3, f4, f5, f6, f7]

        def add_factor(U, V, W, col, factor):
            u_ent, v_ent, w_ent = factor
            for (a_idx, coeff) in u_ent:
                U[a_idx][col] += coeff
            for (b_row, b_col, coeff) in v_ent:
                V[b_row * p + b_col][col] += coeff
            for (c_row, c_col, coeff) in w_ent:
                # Trace convention: C flat = c_col * n + c_row
                W[c_col * n + c_row][col] += coeff

        # 6 Strassen chunks
        for t in range(6):
            cb = 2 * t
            for factor in sf(cb):
                add_factor(U, V, W, col, factor)
                col += 1

        # Single column 12: 4 direct terms
        c12 = 12
        for i in range(n):
            for j in range(m):
                # C[i, 12] += A[i,j] * B[j, 12]
                U[i * m + j][col] = 1
                V[j * p + c12][col] = 1
                W[c12 * n + i][col] = 1
                col += 1

        assert col == rank, f"col={col}, rank={rank}"
        return U, V, W

    U, V, W = build_strassen_v2()

    # ---------- Verification: check that the decomposition is exact ----------
    # The tensor M has M[i, j, k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # The decomposition requires: sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] = M[i,j,k]
    # where M[i,j,k] = 1 for all valid (i,j,k) in the multiplication tensor.
    # Actually, the multiplication tensor T[i, j, k] = 1 if (i,j) is a valid A index,
    # (j,k) is a valid B index, and (i,k) is a valid C index. For matrix multiplication
    # 2x2 @ 2x13, T[i,j,k] = 1 for all i in 0..1, j in 0..1, k in 0..12.
    # The decomposition must satisfy:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    # Wait, that's not quite right. The tensor for matrix multiplication has
    # T[i, j, k] = 1 for all (i,j,k) in the domain, and the rank-r decomposition
    # satisfies T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].

    # Let me verify: for each (i, j, k), the sum over r should equal 1.
    def verify(U, V, W):
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        rank = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    total = 0
                    for r in range(rank):
                        total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
                    if total != 1:
                        return False, (i, j, k, total)
        return True, None

    ok, info = verify(U, V, W)
    if not ok:
        # Fall back to schoolbook
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    if time.time() - start_time > DEADLINE:
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
