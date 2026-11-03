import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m   # 8
    dim_v = m * p   # 28
    dim_w = p * n   # 14

    start_time = time.time()
    deadline = 130  # safety margin before 150s wall-clock limit

    # ------------------------------------------------------------------
    # Canonical building blocks (Kronecker-factor view).
    #
    # A 2x2x2 multiplication tensor is decomposed by Strassen (rank 7).
    # A 2x2x1 multiplication tensor is decomposed by schoolbook (rank 4).
    #
    # Local index convention for a 2x2 block:
    #   U local row = i*2 + j   (A[i, j])
    #   V local row = j*2 + k   (B[j, k])
    #   W local row = k*2 + i   (C[k, i])   (trace convention C[k*n+i])
    #
    # We store each canonical block as three column-stacked lists of
    # (local_row, value) per term, then EMBED it into the global matrices
    # by mapping local rows -> global rows via a permutation.
    # ------------------------------------------------------------------

    # Strassen 2x2x2, rank 7.  Each term: (list of (u_local,val)),
    # (list of (v_local,val)), (list of (w_local,val)).
    # Local rows: A: 00->0,01->1,10->2,11->3 ; B: 00->0,01->1,10->2,11->3 ; C: 00->0,01->2,10->1,11->3
    def _strassen_222():
        terms = []
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        terms.append(([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]))
        # M2 = (A10+A11)B00 -> C10 - C11
        terms.append(([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]))
        # M3 = A00(B01-B11) -> C01 + C11
        terms.append(([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]))
        # M4 = A11(B10-B00) -> C00 + C10
        terms.append(([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]))
        # M5 = (A00+A01)B11 -> -C00 + C01
        terms.append(([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]))
        # M6 = (A10-A00)(B00+B01) -> C11
        terms.append(([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]))
        # M7 = (A01-A11)(B10+B11) -> C00
        terms.append(([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]))
        return terms

    # Schoolbook 2x2x1, rank 4.  B has 2 rows, 1 col; C has 2 rows, 1 col.
    # Local B rows: 0->0, 1->1 ; C local rows: 0->0, 1->1 (single col).
    def _schoolbook_221():
        terms = []
        # N1 = A00 B0 -> C0
        terms.append(([(0,1)], [(0,1)], [(0,1)]))
        # N2 = A01 B1 -> C0
        terms.append(([(1,1)], [(1,1)], [(0,1)]))
        # N3 = A10 B0 -> C1
        terms.append(([(2,1)], [(0,1)], [(1,1)]))
        # N4 = A11 B1 -> C1
        terms.append(([(3,1)], [(1,1)], [(1,1)]))
        return terms

    STRASSEN = _strassen_222()
    SCHOOL_221 = _schoolbook_221()

    def embed(terms, u_map, v_map, w_map, U, V, W, col):
        """Embed a list of canonical terms into global matrices.
        u_map/v_map/w_map: dict local_row -> global_row."""
        for (u_list, v_list, w_list) in terms:
            for (lr, val) in u_list:
                U[u_map[lr], col] += val
            for (lr, val) in v_list:
                V[v_map[lr], col] += val
            for (lr, val) in w_list:
                W[w_map[lr], col] += val
            col += 1
        return col

    # ------------------------------------------------------------------
    # Global rank budget:
    #   Split m=4 into two A-blocks (cols 0-1, cols 2-3).
    #   Split p=7 into three 2-col B-blocks + one 1-col B-block.
    #   Per A-block: 3 Strassen(7) + 1 schoolbook(4) = 25.
    #   Two A-blocks -> total rank 50.
    # ------------------------------------------------------------------
    rank = 50
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))

    col = 0

    # A-block local rows (i*4 + j) for j in {0,1} or {2,3}:
    #   local A row 0 -> global (0, j0) ; 1 -> (0, j0+1) ; 2 -> (1, j0) ; 3 -> (1, j0+1)
    def a_map(j0):
        return {0: 0*m + j0, 1: 0*m + (j0+1), 2: 1*m + j0, 3: 1*m + (j0+1)}

    # B-block local rows (j_local*2 + k_local) for B rows {r0, r0+1}, cols {c0, c0+1}:
    #   local B row 0 -> (r0, c0) ; 1 -> (r0, c0+1) ; 2 -> (r0+1, c0) ; 3 -> (r0+1, c0+1)
    def b_map_2(r0, c0):
        return {0: r0*p + c0, 1: r0*p + (c0+1), 2: (r0+1)*p + c0, 3: (r0+1)*p + (c0+1)}

    # B single-col local rows (r0, c0),(r0+1,c0):
    def b_map_1(r0, c0):
        return {0: r0*p + c0, 1: (r0+1)*p + c0}

    # C-block local rows (k_local*2 + i_local) for C cols {c0,c0+1}, rows {0,1}:
    #   local C row 0 -> (c0, 0) ; 1 -> (c0, 1) ; 2 -> (c0+1, 0) ; 3 -> (c0+1, 1)
    #   global W row = k*n + i
    def c_map_2(c0):
        return {0: c0*n + 0, 1: c0*n + 1, 2: (c0+1)*n + 0, 3: (c0+1)*n + 1}

    # C single-col local rows (c0,0),(c0,1):
    def c_map_1(c0):
        return {0: c0*n + 0, 1: c0*n + 1}

    # B row-blocks: rows {0,1} and {2,3}.
    # C column-blocks: cols {0,1},{2,3},{4,5} (2-wide) and {6} (1-wide).
    b_row_blocks = [0, 2]
    c_col_blocks_2 = [0, 2, 4]
    c_col_block_1 = 6

    for a_j0 in (0, 2):
        for b_r0 in b_row_blocks:
            am = a_map(a_j0)
            for c_c0 in c_col_blocks_2:
                col = embed(STRASSEN, am, b_map_2(b_r0, c_c0), c_map_2(c_c0), U, V, W, col)
            col = embed(SCHOOL_221, am, b_map_1(b_r0, c_col_block_1), c_map_1(c_col_block_1), U, V, W, col)

    if col != rank:
        # Should not happen; keep a safe fallback below.
        pass

    # ------------------------------------------------------------------
    # Build the multiplication tensor T as a flat vector for verification.
    # ------------------------------------------------------------------
    T_flat = np.zeros(dim_u * dim_v * dim_w)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_flat[idx_u * dim_v * dim_w + idx_v * dim_w + idx_w] = 1.0

    def verify(U_, V_, W_, T_flat, atol=1e-10):
        T_recon = np.einsum('it,jt,kt->ijk', U_, V_, W_).flatten()
        return bool(np.allclose(T_recon, T_flat, atol=atol))

    if col == rank and verify(U, V, W, T_flat):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ------------------------------------------------------------------
    # Fallback: schoolbook rank 56 (guaranteed valid).
    # ------------------------------------------------------------------
    if time.time() - start_time > deadline:
        # Extremely unlikely; return a minimal valid-ish structure.
        pass

    rank_school = n * m * p
    U_s = np.zeros((dim_u, rank_school))
    V_s = np.zeros((dim_v, rank_school))
    W_s = np.zeros((dim_w, rank_school))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m + j, c] = 1
                V_s[j*p + k, c] = 1
                W_s[k*n + i, c] = 1
                c += 1
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
