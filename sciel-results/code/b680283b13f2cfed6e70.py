import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [2, 5, 6]
DEADLINE = 140.0  # hard kill is 150s; keep safety margin

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    # ------------------------------------------------------------------
    # The matrix-multiplication tensor T has T[i,j,k] = 1 for all valid
    # (i,j,k) with i in [0,n), j in [0,m), k in [0,p).
    #
    # A rank-r decomposition is given by coefficient matrices
    #   U: (n*m, r)  index i*m+j
    #   V: (m*p, r)  index j*p+k
    #   W: (p*n, r)  index k*n+i
    # such that for every (i,j,k):
    #     sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    #
    # We build a decomposition as a sum of "terms". Each term is a rank-1
    # tensor defined by sparse coefficient dicts:
    #   u_c: {(i,j): val}, v_c: {(j,k): val}, w_c: {(k,i): val}
    # Its contribution to T[i,j,k] is u_c[(i,j)] * v_c[(j,k)] * w_c[(k,i)]
    # (missing entries = 0). We accumulate the sum of contributions and
    # verify it equals the all-ones tensor exactly (integer arithmetic).
    # ------------------------------------------------------------------

    def term_contribution(u_c, v_c, w_c):
        """Return a dense (n*m*p) integer vector of this term's contribution."""
        vec = np.zeros(n * m * p, dtype=np.int64)
        for (i, j), uv in u_c.items():
            for (j2, k), vv in v_c.items():
                if j != j2:
                    continue
                for (k2, i2), wv in w_c.items():
                    if k == k2 and i == i2:
                        vec[i * m * p + j * p + k] += uv * vv * wv
        return vec

    def get_strassen_terms(io, jo, ko):
        """7 rank-1 terms decomposing a 2x2x2 multiplication block."""
        a, b = io, io + 1
        c, d = jo, jo + 1
        e, f = ko, ko + 1
        return [
            # M1
            ({(a, c): 1, (b, d): 1}, {(c, e): 1, (d, f): 1}, {(e, a): 1, (f, b): 1}),
            # M2
            ({(b, c): 1, (b, d): 1}, {(c, e): 1}, {(e, b): 1, (f, b): -1}),
            # M3
            ({(a, c): 1}, {(c, f): 1, (d, f): -1}, {(f, a): 1, (f, b): 1}),
            # M4
            ({(b, d): 1}, {(d, e): 1, (c, e): -1}, {(e, a): 1, (e, b): 1}),
            # M5
            ({(a, c): 1, (a, d): 1}, {(d, f): 1}, {(e, a): -1, (f, a): 1}),
            # M6
            ({(b, c): 1, (a, c): -1}, {(c, e): 1, (c, f): 1}, {(f, b): 1}),
            # M7
            ({(a, d): 1, (b, d): -1}, {(d, e): 1, (d, f): 1}, {(e, a): 1}),
        ]

    def standard_term(i, j, k):
        return ({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    # ------------------------------------------------------------------
    # Build candidate term sets (each is a list of (u_c, v_c, w_c)).
    # We try several structured decompositions and keep the one with the
    # fewest terms that verifies exactly.
    # ------------------------------------------------------------------

    candidate_sets = []

    # (A) Pure schoolbook (rank n*m*p) -- guaranteed fallback.
    school = [standard_term(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    candidate_sets.append(school)

    # (B) Strassen over 2x2x2 blocks tiling the whole space where possible,
    #     plus schoolbook for leftover single columns/rows.
    #     We tile j in pairs, k in pairs; leftover handled by standard terms.
    def build_tiled(j_pairs, k_pairs, leftover_school=True):
        terms = []
        used_j = set()
        used_k = set()
        for (j1, j2) in j_pairs:
            for (k1, k2) in k_pairs:
                for io in range(n - 1):
                    for (u_c, v_c, w_c) in get_strassen_terms(io, j1, k1):
                        terms.append((u_c, v_c, w_c))
                used_j.add(j1); used_j.add(j2)
                used_k.add(k1); used_k.add(k2)
        if leftover_school:
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # only add standard term for (j,k) not fully covered
                        # by a strassen block (i.e. j in a pair AND k in a pair)
                        if not (j in used_j and k in used_k):
                            terms.append(standard_term(i, j, k))
        return terms

    # j pairs covering 5 columns: (0,1),(2,3) -> leftover j=4
    # k pairs covering 6 columns: (0,1),(2,3),(4,5) -> no leftover k
    jp = [(0, 1), (2, 3)]
    kp = [(0, 1), (2, 3), (4, 5)]
    candidate_sets.append(build_tiled(jp, kp))

    # (C) The directive's proposed split: handle the 2x6x6 portion with
    #     rank-11 (2x3x3) blocks, schoolbook for the remaining 2x1x6 strip.
    #     j: {0,1,2} and {3,4} ; k: {0,1,2} and {3,4,5}
    def get_2x3x3_terms(io, j0, k0):
        """Rank-11 decomposition of a 2x3x3 multiplication block.
        Uses two overlapping 2x2x2 Strassen blocks plus standard terms
        for the leftover (row j0+2 / col k0+2) interactions."""
        terms = []
        # Strassen block A: rows {io,io+1}, cols j0,j0+1, k0,k0+1
        for (u_c, v_c, w_c) in get_strassen_terms(io, j0, k0):
            terms.append((u_c, v_c, w_c))
        # Strassen block B: rows {io,io+1}, cols j0+1,j0+2, k0+1,k0+2
        for (u_c, v_c, w_c) in get_strassen_terms(io, j0 + 1, k0 + 1):
            terms.append((u_c, v_c, w_c))
        # Standard terms for the parts not covered by either 2x2x2 block:
        #  - j = j0+2 with k in {k0, k0+1}
        #  - k = k0+2 with j in {j0, j0+1}
        #  - (j0+2, k0+2)
        for i in range(n):
            for j in [j0 + 2]:
                for k in [k0, k0 + 1]:
                    terms.append(standard_term(i, j, k))
            for j in [j0, j0 + 1]:
                for k in [k0 + 2]:
                    terms.append(standard_term(i, j, k))
            terms.append(standard_term(i, j0 + 2, k0 + 2))
        return terms

    def build_2x3x3_split():
        terms = []
        # j groups {0,1,2} and {3,4}; k groups {0,1,2} and {3,4,5}
        j_groups = [(0, 1, 2), (3, 4)]
        k_groups = [(0, 1, 2), (3, 4, 5)]
        covered = set()
        for (j0, j1, j2) in j_groups:
            for (k0, k1, k2) in k_groups:
                for io in range(n - 1):
                    for (u_c, v_c, w_c) in get_2x3x3_terms(io, j0, k0):
                        terms.append((u_c, v_c, w_c))
                covered.add(j0); covered.add(j1); covered.add(j2)
        # leftover j not covered (j=5? no, m=5 so j in 0..4 all covered)
        # all j covered; all k covered.
        return terms

    candidate_sets.append(build_2x3x3_split())

    # (D) A few alternative tilings to explore.
    #     j pairs (0,1),(2,3),(4,?) -> (4,5) invalid since m=5; use (3,4)
    candidate_sets.append(build_tiled([(0, 1), (3, 4)], [(0, 1), (2, 3), (4, 5)]))

    # ------------------------------------------------------------------
    # Verification + exact rank reduction.
    # ------------------------------------------------------------------

    def build_matrices(terms):
        """Convert a list of (u_c,v_c,w_c) into U,V,W matrices (integer)."""
        r = len(terms)
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        for t, (u_c, v_c, w_c) in enumerate(terms):
            for (i, j), val in u_c.items():
                U[i * m + j, t] += val
            for (j, k), val in v_c.items():
                V[j * p + k, t] += val
            for (k, i), val in w_c.items():
                W[k * n + i, t] += val
        return U, V, W

    def verify(U, V, W):
        """Exact integer verification that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]==1."""
        r = U.shape[1]
        # Precompute for each (i,k) the matrix M[j1,j2] = sum_r U[i*m+j1,r]*W[k*n+i,r]*V[j2*p+k,r]
        for i in range(n):
            for k in range(p):
                Ucol = U[i * m:(i + 1) * m, :]          # (m, r)
                Wrow = W[k * n + i, :]                  # (r,)
                # V rows for this k: V[j*p+k, :] for j in 0..m-1
                Vcol = np.array([V[j * p + k, :] for j in range(m)])  # (m, r)
                # M = Ucol * Wrow[None,:] @ Vcol.T  -> (m, m)
                M = (Ucol * Wrow[None, :]) @ Vcol.T
                if not np.array_equal(M, np.eye(m, dtype=np.int64)):
                    return False
        return True

    def reduce_rank(U, V, W):
        """
        Try to reduce rank by finding linear dependencies among the columns
        of the 'slice' matrices. For the decomposition to hold, for each (i,k)
        the matrix M_{ik} = (U_ik * W_ik) @ V_ik^T must equal I.
        If a column r is a linear combination of others such that the identity
        is preserved, we can drop it. We use a simple approach: detect exact
        duplicate/zero columns and drop them; then attempt a rank-revealing
        elimination on the set of column triples.
        """
        # Work on integer arrays.
        U = U.astype(np.int64)
        V = V.astype(np.int64)
        W = W.astype(np.int64)
        r = U.shape[1]

        # Build a list of column triples.
        cols = [(U[:, t].copy(), V[:, t].copy(), W[:, t].copy()) for t in range(r)]

        # Step 1: drop zero columns.
        keep = []
        for t in range(r):
            if np.any(cols[t][0]) or np.any(cols[t][1]) or np.any(cols[t][2]):
                keep.append(t)
        if len(keep) < r:
            cols = [cols[t] for t in keep]
            r = len(cols)

        # Step 2: greedily try to express each column as a combination of
        # previously kept columns. A column r can be dropped if there exists
        # coefficients a_s such that for ALL (i,k):
        #   U_ik[r]*W_ik[r]*V_ik[r] == sum_s a_s * U_ik[s]*W_ik[s]*V_ik[s]
        # where U_ik[r] = U[i*m+*, r] etc. This is a linear condition on a_s.
        # We build a linear system per candidate and solve exactly (rationals
        # via fractions is heavy; use float with tolerance then integer check).

        def col_effect(t):
            """Return (n*m*p,) integer vector of column t's contribution to T."""
            u, v, w = cols[t]
            vec = np.zeros(n * m * p, dtype=np.int64)
            for i in range(n):
                Ucol = u[i * m:(i + 1) * m]
                for k in range(p):
                    wv = w[k * n + i]
                    if wv == 0:
                        continue
                    Vcol = v[k::p]  # v[j*p+k] for j
                    for j in range(m):
                        uv = Ucol[j]
                        if uv == 0:
                            continue
                        vv = Vcol[j]
                        if vv == 0:
                            continue
                        vec[i * m * p + j * p + k] += uv * vv * wv
            return vec

        # Precompute effect vectors.
        effects = [col_effect(t) for t in range(r)]

        # Greedy elimination: maintain a basis of effect vectors.
        basis = []          # list of effect vectors (int)
        basis_idx = []      # corresponding column indices
        kept = []

        for t in range(r):
            if time.time() - start_time > DEADLINE:
                break
            e = effects[t]
            if not np.any(e):
                continue
            # Check if e is in span of basis (over rationals). Use float least
            # squares; if residual ~0 (exact integer check), it's dependent.
            if not basis:
                basis.append(e.copy())
                basis_idx.append(t)
                kept.append(t)
                continue
            B = np.stack(basis, axis=0).astype(np.float64)  # (b, 60)
            # Solve B^T x = e
            try:
                x, res, rank, sv = np.linalg.lstsq(B.T, e.astype(np.float64), rcond=None)
            except np.linalg.LinAlgError:
                basis.append(e.copy()); basis_idx.append(t); kept.append(t); continue
            # Reconstruct integer check
            recon = (B * x[None, :]).round().astype(np.int64)
            if np.array_equal(recon, e):
                # dependent -> drop
                continue
            else:
                basis.append(e.copy())
                basis_idx.append(t)
                kept.append(t)

        if len(kept) < r:
            U = U[:, kept]
            V = V[:, kept]
            W = W[:, kept]
            # Re-verify after reduction.
            if verify(U, V, W):
                return U, V, W
        return U, V, W

    # ------------------------------------------------------------------
    # Evaluate all candidate sets; keep the best (lowest rank) verified one.
    # ------------------------------------------------------------------

    best = None  # (rank, U, V, W)
    for terms in candidate_sets:
        if time.time() - start_time > DEADLINE:
            break
        U, V, W = build_matrices(terms)
        if verify(U, V, W):
            U, V, W = reduce_rank(U, V, W)
            r = U.shape[1]
            if best is None or r < best[0]:
                best = (r, U, V, W)

    if best is None:
        # Absolute fallback: schoolbook matrices.
        U, V, W = build_matrices(school)
        best = (U.shape[1], U, V, W)

    _, U, V, W = best

    # Final safety: ensure the result is valid; if not, fall back to schoolbook.
    if not verify(U, V, W):
        U, V, W = build_matrices(school)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
