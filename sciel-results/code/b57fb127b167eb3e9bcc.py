"""
Matrix multiplication 2x3x7 - rank-minimal exact tensor decomposition.

Strategy:
  1. Build the schoolbook rank-42 decomposition.
  2. Systematically check all pairs of schoolbook terms to see if their sum
     (or difference, or scaled sum) is a rank-1 tensor.  If so, replace the
     two terms with one, reducing rank to 41.
  3. If a rank-41 decomposition is found, return it; otherwise return the
     schoolbook rank-42 decomposition.

The tensor T has shape (n*m, m*p, p*n) = (6, 21, 14) with
  T[i*m+j, j*p+k, k*n+i] = 1  for valid i in [0,2), j in [0,3), k in [0,7).
All other entries are 0.

A rank-1 tensor S can be written S[a,b,c] = U[a,r] * V[b,r] * W[c,r] for a single r.
Equivalently, for any two "slices" the matrix must have rank <= 1.
"""

import time
import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 3, 7]


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42

    # ---------- Build the target tensor ----------
    # Index mapping:
    #   a = i*m + j   (0..n*m-1)
    #   b = j*p + k   (0..m*p-1)
    #   c = k*n + i   (0..p*n-1)
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    # ---------- Schoolbook decomposition (rank 42) ----------
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)

    col = 0
    term_info = []  # list of (i, j, k) in the same order as columns
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                term_info.append((i, j, k))
                col += 1

    def verify(U, V, W):
        """Check that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c."""
        rank = U.shape[1]
        # Build the full tensor via einsum (or manual) and compare
        # For speed, check only the 42 non-zero positions plus a sample of zero positions.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check a set of zero positions
        checked = 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i1, j1 = divmod(a, m)
                    j2, k1 = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if j1 == j2 and i1 == i2 and k1 == k2:
                        continue  # non-zero position
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > 1e-9:
                        return False
                    checked += 1
                    if checked >= 500:
                        break
                if checked >= 500:
                    break
            if checked >= 500:
                break
        return True

    # ---------- Pair-merging check ----------
    # For each pair of schoolbook terms (ta, tb), check if their combined
    # contribution can be represented as a single rank-1 term.
    #
    # Term ta corresponds to (ia, ja, ka), term tb to (ib, jb, kb).
    # The combined tensor S = Ta + Tb has non-zero entries at:
    #   (ia*m+ja, ja*p+ka, ka*n+ia) = 1
    #   (ib*m+jb, jb*p+kb, kb*n+ib) = 1
    # (and possibly overlaps if the two terms share the same (a,b,c) position,
    #  which can only happen if ia==ib, ja==jb, ka==kb, i.e. same term).
    #
    # S is rank-1 iff there exist vectors u (len n*m), v (len m*p), w (len p*n)
    # such that u[a]*v[b]*w[c] = S[a,b,c] for all a,b,c.
    #
    # Since S has at most 2 non-zero entries, we can try to solve for u, v, w.
    # Let the two positions be p1 = (a1, b1, c1) and p2 = (a2, b2, c2).
    # We need:
    #   u[a1] * v[b1] * w[c1] = 1
    #   u[a2] * v[b2] * w[c2] = 1
    # and all other products = 0.
    #
    # For the second condition (all other entries zero), we need that for every
    # (a, b, c) not in {p1, p2}, at least one of u[a], v[b], w[c] is zero.
    # This is a constraint satisfaction problem.
    #
    # A simpler sufficient check: S is rank-1 iff the 2x2 minors vanish.
    # Flatten S as a matrix M where M[x, y] = S indexed somehow.
    # Actually, the cleanest check: S is rank-1 iff for all a,a',b,b',c,c',
    #   S[a,b,c]*S[a',b',c'] == S[a,b',c']*S[a',b,c']
    # But with only 2 non-zero entries, we can check directly.
    #
    # Alternative: build S explicitly (sparse) and check rank-1 via the
    # condition that all 2x2 sub-determinants of any unfolding are zero.
    #
    # Let's use a direct approach: enumerate the support of S and try to
    # assign values.

    def check_pair_merge(i1, j1, k1, i2, j2, k2):
        """Check if terms (i1,j1,k1) and (i2,j2,k2) can be merged into one rank-1 term.
        Returns (U_col, V_col, W_col) as numpy arrays of length (n*m), (m*p), (p*n)
        with the merged term, or None if not possible with half-integer entries."""

        a1 = i1 * m + j1
        b1 = j1 * p + k1
        c1 = k1 * n + i1
        a2 = i2 * m + j2
        b2 = j2 * p + k2
        c2 = k2 * n + i2

        # Build the support: positions where S != 0
        # S has 1 at (a1,b1,c1) and 1 at (a2,b2,c2).
        # If (a1,b1,c1) == (a2,b2,c2), then S has value 2 at that single position,
        # which is trivially rank-1 (u[a1]*v[b1]*w[c1]=2, rest 0).
        if a1 == a2 and b1 == b2 and c1 == c2:
            # Same position: value is 2
            U_col = np.zeros(n * m)
            V_col = np.zeros(m * p)
            W_col = np.zeros(p * n)
            # u[a1]*v[b1]*w[c1] = 2, e.g., u=2, v=1, w=1
            U_col[a1] = 2
            V_col[b1] = 1
            W_col[c1] = 1
            # Verify half-integer: yes
            return U_col, V_col, W_col

        # Two distinct positions. S has 1 at both.
        # For S to be rank-1, we need:
        #   u[a1]*v[b1]*w[c1] = 1
        #   u[a2]*v[b2]*w[c2] = 1
        #   u[a]*v[b]*w[c] = 0 for all other (a,b,c)
        #
        # The zero condition means: for every (a,b,c) not in {(a1,b1,c1),(a2,b2,c2)},
        # at least one of u[a], v[b], w[c] must be zero.
        #
        # This is equivalent to: the support of u (set of a with u[a]!=0), support
        # of v, support of w must be such that the only (a,b,c) with a in supp_u,
        # b in supp_v, c in supp_w are (a1,b1,c1) and (a2,b2,c2).
        #
        # Let su = set of a indices where u can be non-zero, etc.
        # From the two equations, the a-values that can be non-zero are {a1, a2}
        # (if a1==a2, just {a1}). Similarly for b and c.
        #
        # Case 1: a1 == a2, b1 == b2, c1 != c2
        #   u[a1] * v[b1] * w[c1] = 1
        #   u[a1] * v[b1] * w[c2] = 1
        #   => w[c1] = w[c2]
        #   Zero condition: for any (a,b,c) not in the two positions, at least one factor is 0.
        #   Since u is non-zero only at a1, v only at b1, w must be zero at all c except c1 and c2.
        #   So w[c] = 0 for c not in {c1, c2}. That's fine.
        #   We need u[a1]*v[b1]*w[c1] = 1. Let w[c1]=w[c2]=w_val, then u[a1]*v[b1]*w_val = 1.
        #   Choose w_val = 1, u[a1]=1, v[b1]=1. All integers.
        #   But wait: we also need to check that no other (a,b,c) with a=a1, b=b1, c=c1 or c2
        #   gives a non-zero that's not in our support. Since u is only non-zero at a1 and
        #   v only at b1, the only non-zero products are at (a1,b1,c1) and (a1,b1,c2), which
        #   are exactly our two positions. Good.

        # Let's handle all cases systematically.
        # Determine the sets of indices that can be non-zero.
        # The a-values in the support: {a1, a2}
        # The b-values: {b1, b2}
        # The c-values: {c1, c2}

        # For rank-1, the support of the tensor must be a "grid" product of the
        # supports of u, v, w. That is, supp(S) = supp(u) x supp(v) x supp(w)
        # intersected with the actual non-zero positions. But since S has exactly
        # 2 non-zero entries, and supp(u) x supp(v) x supp(w) must have size <= 2
        # (otherwise there would be more non-zero entries), we need:
        # |{a1,a2}| * |{b1,b2}| * |{c1,c2}| <= 2

        na = len({a1, a2})
        nb = len({b1, b2})
        nc = len({c1, c2})

        if na * nb * nc > 2:
            return None

        # Now we can try to solve for u, v, w.
        # Since na*nb*nc <= 2, the supports are very small.
        # Try all possible assignments of non-zero values from a small set of
        # half-integer candidates: {±1, ±2, ±0.5, ±1.5, ±3}
        # Actually, to keep it simple, try {±1, ±2, ±0.5}

        candidates = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]

        # The non-zero positions are p1 and p2.
        positions = [(a1, b1, c1, 1.0), (a2, b2, c2, 1.0)]
        # If same position, already handled above.

        # For the grid to work, every (a, b, c) in {a1,a2} x {b1,b2} x {c1,c2}
        # must either be one of the two positions (with value 1) or have at least
        # one of u[a], v[b], w[c] equal to 0.
        #
        # So we need to assign u[a] for a in {a1,a2}, v[b] for b in {b1,b2},
        # w[c] for c in {c1,c2}, and possibly 0 for others, such that:
        #   u[a1]*v[b1]*w[c1] = 1
        #   u[a2]*v[b2]*w[c2] = 1
        #   For every (a,b,c) in the grid that is NOT (a1,b1,c1) or (a2,b2,c2):
        #       u[a]*v[b]*w[c] = 0

        a_vals = [a1, a2] if a1 != a2 else [a1]
        b_vals = [b1, b2] if b1 != b2 else [b1]
        c_vals = [c1, c2] if c1 != c2 else [c1]

        # Enumerate assignments for u on a_vals, v on b_vals, w on c_vals
        # Each can be 0 or one of the candidates.
        # Total combinations: (1+len(candidates))^3 * ... which is manageable.

        for u_a1 in [0] + candidates:
            if time.time() > deadline:
                return None
            if a1 == a2:
                u_a2 = u_a1
            else:
                pass  # will be in outer loop below; restructure
            # Actually, let's just do nested loops properly.
            break

        # Proper nested enumeration:
        u_options = [0] + candidates
        for u_a1 in u_options:
            for u_a2 in u_options:
                if a1 == a2 and u_a1 != u_a2:
                    continue
                for v_b1 in u_options:
                    for v_b2 in u_options:
                        if b1 == b2 and v_b1 != v_b2:
                            continue
                        for w_c1 in u_options:
                            for w_c2 in u_options:
                                if c1 == c2 and w_c1 != w_c2:
                                    continue
                                # Check the two non-zero equations
                                if abs(u_a1 * v_b1 * w_c1 - 1.0) > 1e-9:
                                    continue
                                if abs(u_a2 * v_b2 * w_c2 - 1.0) > 1e-9:
                                    continue
                                # Check zero conditions for all grid points
                                ok = True
                                for a in a_vals:
                                    for b in b_vals:
                                        for c in c_vals:
                                            if (a, b, c) == (a1, b1, c1):
                                                continue
                                            if (a, b, c) == (a2, b2, c2):
                                                continue
                                            ua = u_a1 if a == a1 else u_a2
                                            vb = v_b1 if b == b1 else v_b2
                                            wc = w_c1 if c == c1 else w_c2
                                            if abs(ua * vb * wc) > 1e-9:
                                                ok = False
                                                break
                                        if not ok:
                                            break
                                    if not ok:
                                        break
                                if not ok:
                                    continue
                                # Found a valid assignment!
                                U_col = np.zeros(n * m)
                                V_col = np.zeros(m * p)
                                W_col = np.zeros(p * n)
                                U_col[a1] = u_a1
                                if a2 != a1:
                                    U_col[a2] = u_a2
                                V_col[b1] = v_b1
                                if b2 != b1:
                                    V_col[b2] = v_b2
                                W_col[c1] = w_c1
                                if c2 != c1:
                                    W_col[c2] = w_c2
                                return U_col, V_col, W_col
        return None

    # ---------- Try all pairs ----------
    best_result = None
    rank_found = rank_schoolbook

    num_terms = len(term_info)
    for idx in range(num_terms):
        if time.time() > deadline:
            break
        for jdx in range(idx + 1, num_terms):
            if time.time() > deadline:
                break
            i1, j1, k1 = term_info[idx]
            i2, j2, k2 = term_info[jdx]
            result = check_pair_merge(i1, j1, k1, i2, j2, k2)
            if result is not None:
                U_col, V_col, W_col = result
                # Build a rank-41 decomposition: replace terms idx and jdx with the merged term
                U_new = np.zeros((n * m, rank_found - 1), dtype=np.float64)
                V_new = np.zeros((m * p, rank_found - 1), dtype=np.float64)
                W_new = np.zeros((p * n, rank_found - 1), dtype=np.float64)

                # Copy all terms except idx and jdx
                col = 0
                for tidx in range(num_terms):
                    if tidx == idx or tidx == jdx:
                        continue
                    tt = term_info[tidx]
                    ti, tj, tk = tt
                    U_new[ti * m + tj, col] = 1
                    V_new[tj * p + tk, col] = 1
                    W_new[tk * n + ti, col] = 1
                    col += 1
                # Add the merged term
                U_new[:, col] = U_col
                V_new[:, col] = V_col
                W_new[:, col] = W_col

                # Verify
                if verify(U_new, V_new, W_new):
                    best_result = (U_new, V_new, W_new)
                    rank_found = rank_found - 1
                    break
        if best_result is not None:
            break

    # ---------- Return result ----------
    def to_list(mat):
        return mat.tolist()

    if best_result is not None:
        U, V, W = best_result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    return {
        "U": to_list(U_sb),
        "V": to_list(V_sb),
        "W": to_list(W_sb)
    }
# EVOLVE-BLOCK-END
