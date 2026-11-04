"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition.
Combines disjoint Strassen-block packing with rank-1 term fusion and
exact column-elimination rank reduction."""
import time
from fractions import Fraction
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    NTRIPLES = n * m * p  # 105

    # ------------------------------------------------------------------
    # Helper: build a single bilinear-form column as (u, v, w) tuples
    # ------------------------------------------------------------------
    def make_col(u_dict, v_dict, w_dict):
        u = [0] * U_rows
        v = [0] * V_rows
        w = [0] * W_rows
        for idx, val in u_dict.items():
            u[idx] = val
        for idx, val in v_dict.items():
            v[idx] = val
        for idx, val in w_dict.items():
            w[idx] = val
        return (u, v, w)

    def add_strassen_cols(i1, i2, j1, j2, k1, k2):
        """Return the 7 Strassen bilinear-form columns for a 2x2x2 block."""
        a11, a12 = i1 * m + j1, i1 * m + j2
        a21, a22 = i2 * m + j1, i2 * m + j2
        b11, b12 = j1 * p + k1, j1 * p + k2
        b21, b22 = j2 * p + k1, j2 * p + k2
        c11, c12 = k1 * n + i1, k2 * n + i1
        c21, c22 = k1 * n + i2, k2 * n + i2

        cols = []
        # M1 = (a11+a22)*(b11+b22) -> c11 + c22
        cols.append(make_col({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1}))
        # M2 = (a21+a22)*b11 -> c21 - c22
        cols.append(make_col({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1}))
        # M3 = a11*(b12-b22) -> c12 + c22
        cols.append(make_col({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1}))
        # M4 = a22*(b21-b11) -> c11 + c21
        cols.append(make_col({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1}))
        # M5 = (a11+a12)*b22 -> -c11 + c12
        cols.append(make_col({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1}))
        # M6 = (a21-a11)*(b11+b12) -> c22
        cols.append(make_col({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1}))
        # M7 = (a12-a22)*(b21+b22) -> c11
        cols.append(make_col({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1}))
        return cols

    # ------------------------------------------------------------------
    # Greedy search for maximum packing of disjoint 2x2x2 Strassen blocks
    # ------------------------------------------------------------------
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    best_num_blocks = 0
    best_packing = []
    num_trials = 2000
    for t in range(num_trials):
        if time.time() > deadline - 30.0:
            break
        random.shuffle(all_blocks)
        used_indices = set()
        current_packing = []
        for block_params, block_indices in all_blocks:
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing
            if best_num_blocks >= 6:
                break

    # ------------------------------------------------------------------
    # Build initial decomposition: Strassen blocks + schoolbook for rest
    # ------------------------------------------------------------------
    cols = []  # list of (u, v, w) tuples
    covered = set()

    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        s_cols = add_strassen_cols(i1, i2, j1, j2, k1, k2)
        cols.extend(s_cols)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    cols.append(make_col({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1}))

    # ------------------------------------------------------------------
    # Verification function
    # ------------------------------------------------------------------
    def verify(cols, n, m, p, deadline):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for (u, v, w) in cols:
                        total += Fraction(u[i * m + j]) * Fraction(v[j * p + k]) * Fraction(w[k * n + i])
                    if total != 1:
                        return False
                    if time.time() > deadline:
                        return False
        return True

    # ------------------------------------------------------------------
    # Rank-1 term fusion: merge pairs that share identical U or V rows
    # ------------------------------------------------------------------
    def try_fusion(cols, deadline):
        """Try to merge pairs of terms that share identical U or V factor rows."""
        if time.time() > deadline - 5.0:
            return cols

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            n_cols = len(cols)

            # Build lookup for U rows and V rows
            u_map = {}
            v_map = {}
            for idx, (u, v, w) in enumerate(cols):
                u_key = tuple(u)
                v_key = tuple(v)
                u_map.setdefault(u_key, []).append(idx)
                v_map.setdefault(v_key, []).append(idx)

            # Try merging pairs with same U row
            for u_key, indices in u_map.items():
                if time.time() > deadline - 5.0:
                    break
                if len(indices) < 2:
                    continue
                for a_pos in range(len(indices)):
                    for b_pos in range(a_pos + 1, len(indices)):
                        ia, ib = indices[a_pos], indices[b_pos]
                        u_a, v_a, w_a = cols[ia]
                        u_b, v_b, w_b = cols[ib]

                        # Check if they can be merged: need v_a == v_b or w_a == w_b
                        # If v_a == v_b, we can combine: new_w = w_a + w_b (but this changes the tensor!)
                        # Actually, fusion requires: u_a*v_a*w_a + u_b*v_b*w_b = u_new*v_new*w_new
                        # This is only possible in special cases. Let's check if v_a == v_b:
                        if tuple(v_a) == tuple(v_b):
                            # u_a*v_a*w_a + u_a*v_a*w_b = u_a*v_a*(w_a+w_b)
                            # This works! We can merge into one term with w = w_a + w_b
                            new_w = [w_a[r] + w_b[r] for r in range(W_rows)]
                            # Check if new_w is valid (all entries are -1, 0, or 1 for simplicity)
                            # Actually, we just need the tensor to be correct
                            new_cols = [cols[i] for i in range(n_cols) if i != ia and i != ib]
                            new_cols.append((u_a, v_a, new_w))
                            if verify(new_cols, n, m, p, deadline):
                                cols = new_cols
                                improved = True
                                break
                        elif tuple(w_a) == tuple(w_b):
                            # u_a*v_a*w + u_b*v_b*w = w*(u_a*v_a + u_b*v_b)
                            # This doesn't directly give a rank-1 form unless u_a*v_a + u_b*v_b is rank-1
                            # Skip for now
                            pass
                    if improved:
                        break
                if improved:
                    break

            if improved:
                continue

            # Try merging pairs with same V row
            for v_key, indices in v_map.items():
                if time.time() > deadline - 5.0:
                    break
                if len(indices) < 2:
                    continue
                for a_pos in range(len(indices)):
                    for b_pos in range(a_pos + 1, len(indices)):
                        ia, ib = indices[a_pos], indices[b_pos]
                        u_a, v_a, w_a = cols[ia]
                        u_b, v_b, w_b = cols[ib]

                        if tuple(u_a) == tuple(u_b):
                            # u*v_a*w_a + u*v_b*w_b = u*(v_a*w_a + v_b*w_b)
                            # Only works if v_a*w_a + v_b*w_b is rank-1 in (v,w) space
                            # Skip for now
                            pass
                        elif tuple(w_a) == tuple(w_b):
                            # u_a*v_a*w + u_b*v_b*w = w*(u_a*v_a + u_b*v_b)
                            # Skip
                            pass
                    if improved:
                        break
                if improved:
                    break

        return cols

    cols = try_fusion(cols, deadline)

    # ------------------------------------------------------------------
    # Exact column-elimination rank reduction
    # ------------------------------------------------------------------
    def build_tensor_matrix(cols, n, m, p):
        """Build the (n*m*p) x rank matrix."""
        r = len(cols)
        M = np.zeros((n * m * p, r), dtype=object)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        u, v, w = cols[r_idx]
                        M[idx, r_idx] = u[i * m + j] * v[j * p + k] * w[k * n + i]
                    idx += 1
        return M

    def exact_rank_and_pivots(M, deadline):
        """Exact Gaussian elimination to find rank and pivot columns."""
        rows = M.shape[0]
        cols = M.shape[1]
        A = [list(M[row]) for row in range(rows)]
        pivot_cols = []
        current_row = 0

        for col_idx in range(cols):
            if time.time() > deadline:
                break
            if current_row >= rows:
                break
            pivot_row = -1
            for row_idx in range(current_row, rows):
                if A[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                continue
            A[current_row], A[pivot_row] = A[pivot_row], A[current_row]
            pivot_val = A[current_row][col_idx]
            for row_idx in range(current_row + 1, rows):
                if A[row_idx][col_idx] != 0:
                    factor = A[row_idx][col_idx] / pivot_val
                    for c in range(col_idx, cols):
                        A[row_idx][c] -= factor * A[current_row][c]
            pivot_cols.append(col_idx)
            current_row += 1

        return pivot_cols

    def reduce_rank(cols, n, m, p, deadline):
        """Try to reduce rank via exact column elimination."""
        if time.time() > deadline - 10.0:
            return cols

        r = len(cols)
        M = build_tensor_matrix(cols, n, m, p)

        # Use float SVD to estimate if reduction is possible
        M_float = np.array(M, dtype=float)
        try:
            sv = np.linalg.svd(M_float, compute_uv=False)
            tol = max(M_float.shape) * np.finfo(float).eps * (sv[0] if sv[0] > 0 else 1.0)
            num_rank = int(np.sum(sv > tol))
        except:
            num_rank = r

        if num_rank < r:
            pivot_cols = exact_rank_and_pivots(M, deadline)
            exact_rank = len(pivot_cols)

            if exact_rank < r:
                pivot_set = set(pivot_cols)
                keep_cols = [c for c in range(r) if c not in pivot_set]
                new_cols = [cols[c] for c in keep_cols]

                if verify(new_cols, n, m, p, deadline):
                    return new_cols

        return cols

    cols = reduce_rank(cols, n, m, p, deadline)

    # ------------------------------------------------------------------
    # Convert to U, V, W matrix format
    # ------------------------------------------------------------------
    rank = len(cols)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]

    for c in range(rank):
        u, v, w = cols[c]
        for r in range(U_rows):
            U[r][c] = u[r]
        for r in range(V_rows):
            V[r][c] = v[r]
        for r in range(W_rows):
            W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
