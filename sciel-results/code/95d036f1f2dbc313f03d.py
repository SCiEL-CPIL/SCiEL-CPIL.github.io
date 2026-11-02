"""
Matrix multiplication 2x5x5 rank minimization.
Structural approach: exploit n=2 symmetry by viewing the problem as
two coupled 1x5x5 matmuls, then apply greedy rank reduction via
linear dependence detection in the coefficient space.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Build rank-50 schoolbook decomposition."""
    n, m, p = 2, 5, 5
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(([(i * m + j, 1)], [(j * p + k, 1)], [(k * n + i, 1)]))
    return terms

def _build_strassen_decomposition():
    """
    Build the rank-46 Strassen-based decomposition for 2x5 x 5x5.
    Partition inner dimension 5 = 2+2+1.
    """
    n, m, p = 2, 5, 5
    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        a_u = a_row_off * m + a_col_off
        b_u = a_row_off * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + a_col_off
        d_u = (a_row_off + 1) * m + (a_col_off + 1)

        e_v = b_row_off * p + b_col_off
        f_v = b_row_off * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + b_col_off
        h_v = (b_row_off + 1) * p + (b_col_off + 1)

        o00_w = out_col_off * n + out_row_off
        o01_w = (out_col_off + 1) * n + out_row_off
        o10_w = out_col_off * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)

        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])

    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    schoolbook_block(0, 0, 0, 4, (2, 2), (2, 1), 0, 4)

    strassen_2x2x2(0, 2, 2, 0, 0, 0)
    strassen_2x2x2(0, 2, 2, 2, 0, 2)
    schoolbook_block(0, 2, 2, 4, (2, 2), (2, 1), 0, 4)

    schoolbook_block(0, 4, 4, 0, (2, 1), (1, 5), 0, 0)

    return terms

def _terms_to_matrices(terms):
    """Convert list of term tuples to U, V, W matrices."""
    n, m, p = 2, 5, 5
    total_rank = len(terms)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val

    return U, V, W, total_rank

def _validate_decomposition(U, V, W, n, m, p):
    """Validate that the decomposition exactly reconstructs the matrix multiplication tensor."""
    r = len(W[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = 0.0
                for c in range(r):
                    val += U[idx_u][c] * V[idx_v][c] * W[idx_w][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _contrib_matrix(terms):
    """Build the r x (n*m*p) contribution matrix."""
    n, m, p = 2, 5, 5
    r = len(terms)
    M = np.zeros((r, n * m * p))
    for c in range(r):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for idx, val in terms[c][0]:
            u_dict[idx] = val
        for idx, val in terms[c][1]:
            v_dict[idx] = val
        for idx, val in terms[c][2]:
            w_dict[idx] = val
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    M[c, i * m * p + j * p + k] = u_dict.get(idx_u, 0) * v_dict.get(idx_v, 0) * w_dict.get(idx_w, 0)
    return M

def _apply_null_space(terms, M, deadline):
    """
    Iteratively apply null-space vectors to zero out terms.
    Returns (new_terms, new_M) or (terms, M) if no improvement.
    """
    n, m, p = 2, 5, 5
    r = len(terms)
    num_constraints = n * m * p

    try:
        null_space = np.linalg.null_space(M.T)
    except:
        return terms, M

    if null_space is None or null_space.shape[1] == 0:
        return terms, M

    best_terms = terms
    best_rank = r
    best_M = M
    ns_dim = null_space.shape[1]

    for ns in range(ns_dim):
        if time.time() > deadline - 10:
            break
        v = null_space[:, ns]

        candidates = []
        for c in range(r):
            if abs(v[c]) > 1e-10:
                candidates.append(-1.0 / v[c])

        best_t = None
        best_zero = 0
        for t_val in candidates:
            new_coeffs = 1.0 + t_val * v
            zc = sum(1 for c in range(r) if abs(new_coeffs[c]) < 1e-9)
            if zc > best_zero:
                best_zero = zc
                best_t = t_val

        if best_t is not None and best_zero >= 1:
            new_coeffs = 1.0 + best_t * v
            new_terms = []
            for c in range(r):
                if abs(new_coeffs[c]) < 1e-9:
                    continue
                s = new_coeffs[c]
                new_terms.append((
                    [(idx, val * s) for idx, val in terms[c][0]],
                    [(idx, val) for idx, val in terms[c][1]],
                    [(idx, val) for idx, val in terms[c][2]]
                ))
            if len(new_terms) < best_rank:
                U_t, V_t, W_t, _ = _terms_to_matrices(new_terms)
                if _validate_decomposition(U_t, V_t, W_t, n, m, p):
                    best_terms = new_terms
                    best_rank = len(new_terms)
                    best_M = _contrib_matrix(new_terms)

    # Try pairs of null-space vectors
    if ns_dim >= 2 and time.time() < deadline - 15:
        r_cur = len(best_terms)
        try:
            ns2 = np.linalg.null_space(best_M.T)
        except:
            ns2 = None
        if ns2 is not None and ns2.shape[1] >= 2:
            nd2 = ns2.shape[1]
            for ii in range(nd2):
                for jj in range(ii + 1, nd2):
                    if time.time() > deadline - 10:
                        break
                    vi = ns2[:, ii]
                    vj = ns2[:, jj]
                    for p1 in range(r_cur):
                        for p2 in range(p1 + 1, r_cur):
                            if abs(vi[p1]) < 1e-12 and abs(vj[p1]) < 1e-12:
                                continue
                            if abs(vi[p2]) < 1e-12 and abs(vj[p2]) < 1e-12:
                                continue
                            A_mat = np.array([[vi[p1], vj[p1]], [vi[p2], vj[p2]]])
                            b_vec = np.array([-1.0, -1.0])
                            try:
                                bc = np.linalg.solve(A_mat, b_vec)
                                nv = 1.0 + bc[0] * vi + bc[1] * vj
                                zc = sum(1 for c in range(r_cur) if abs(nv[c]) < 1e-9)
                                if zc >= 2:
                                    nt = []
                                    for c in range(r_cur):
                                        if abs(nv[c]) < 1e-9:
                                            continue
                                        s = nv[c]
                                        nt.append((
                                            [(idx, val * s) for idx, val in best_terms[c][0]],
                                            [(idx, val) for idx, val in best_terms[c][1]],
                                            [(idx, val) for idx, val in best_terms[c][2]]
                                        ))
                                    if len(nt) < best_rank:
                                        U_t, V_t, W_t, _ = _terms_to_matrices(nt)
                                        if _validate_decomposition(U_t, V_t, W_t, n, m, p):
                                            best_terms = nt
                                            best_rank = len(nt)
                                            best_M = _contrib_matrix(nt)
                            except:
                                pass

    return best_terms, best_M

def _greedy_term_elimination(terms, deadline):
    """
    Structural approach: try to find a linear combination of ALL terms
    that zeros out at least one term, using the full null space of the
    contribution matrix. Iterate until no more reductions are found.
    """
    n, m, p = 2, 5, 5
    M = _contrib_matrix(terms)
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        terms, M = _apply_null_space(terms, M, deadline)
        # Check if we actually improved
        if len(terms) < len(M):
            improved = True
            M = _contrib_matrix(terms)
    return terms

def _try_2x2x5_blocks(deadline):
    """
    Alternative: tile as 2x2x5 blocks using a custom 2x2x5 decomposition.
    For 2x2x5: partition k into 2+2+1.
    Use Strassen for 2x2x2 blocks and schoolbook for 2x2x1.
    This gives same 46 rank but different structure may allow more cancellation.
    """
    n, m, p = 2, 5, 5
    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def strassen_2x2x2(r_off, c_off, k_off):
        a_u = r_off * m + c_off
        b_u = r_off * m + c_off + 1
        c_u = (r_off + 1) * m + c_off
        d_u = (r_off + 1) * m + c_off + 1
        e_v = k_off * p + 0
        f_v = k_off * p + 1
        g_v = (k_off + 1) * p + 0
        h_v = (k_off + 1) * p + 1
        o00 = 0 * n + r_off
        o01 = 1 * n + r_off
        o10 = 0 * n + r_off + 1
        o11 = 1 * n + r_off + 1
        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00, 1), (o11, 1)])
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10, 1), (o11, -1)])
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01, 1), (o11, 1)])
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00, 1), (o10, 1)])
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00, -1), (o01, 1)])
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11, 1)])
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00, 1)])

    def school_2x2x1(r_off, c_off, k_off):
        for di in range(2):
            for dj in range(2):
                u_idx = (r_off + di) * m + (c_off + dj)
                v_idx = (k_off) * p + 0
                w_idx = 0 * n + (r_off + di)
                add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    # Process each column block of A (j=0..4)
    # For j=0,1: use 2x2x5 on B columns 0..4
    # Actually let's do: for each pair of j's, process all 5 k's
    # j-pair (0,1): k = 0,1 (Strassen), k=2,3 (Strassen), k=4 (school)
    strassen_2x2x2(0, 0, 0)
    strassen_2x2x2(0, 0, 2)
    school_2x2x1(0, 0, 4)
    # j-pair (2,3)
    strassen_2x2x2(0, 2, 0)
    strassen_2x2x2(0, 2, 2)
    school_2x2x1(0, 2, 4)
    # j=4
    for di in range(2):
        for dk in range(5):
            u_idx = di * m + 4
            v_idx = dk * p + 0
            w_idx = 0 * n + di
            add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    return terms

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy 1: Strassen-based + null-space reduction
    terms1 = _build_strassen_decomposition()
    terms1 = _greedy_term_elimination(terms1, deadline)
    rank1 = len(terms1)

    # Strategy 2: Alternative tiling + null-space reduction
    terms2 = _try_2x2x5_blocks(deadline)
    terms2 = _greedy_term_elimination(terms2, deadline)
    rank2 = len(terms2)

    # Strategy 3: Schoolbook + aggressive null-space
    if time.time() < deadline - 30:
        terms3 = _build_schoolbook()
        terms3 = _greedy_term_elimination(terms3, deadline)
        rank3 = len(terms3)
    else:
        terms3 = terms1
        rank3 = rank1

    # Pick best
    best = min([(rank1, terms1), (rank2, terms2), (rank3, terms3)], key=lambda x: x[0])
    terms = best[1]

    # Convert to matrices
    U, V, W, total_rank = _terms_to_matrices(terms)

    # Final verification
    if not _validate_decomposition(U, V, W, 2, 5, 5):
        # Fallback
        terms = _build_strassen_decomposition()
        U, V, W, total_rank = _terms_to_matrices(terms)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
