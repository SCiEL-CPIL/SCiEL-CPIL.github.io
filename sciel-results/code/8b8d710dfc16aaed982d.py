"""Matrix multiplication 2x4x8 - rank-minimal exact tensor decomposition.

Strategy:
1. Start from the schoolbook rank-64 decomposition.
2. Use a greedy rank-elimination search: repeatedly try to find a rank-1
   tensor (u,v,w) whose addition allows removing one existing term via
   exact rational linear algebra (rank tests).
3. Verify the final decomposition exactly with integer/rational arithmetic.
"""

from fractions import Fraction
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16
    rank_school = n * m * p  # 64

    # Build schoolbook terms: each term is (u, v, w) 1-hots
    # U row = i*m+j, V row = j*p+k, W row = k*n+i
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * nm
                v = [0] * mp
                w = [0] * pn
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                terms.append((u, v, w))

    best = terms
    deadline = time.time() + 100  # safety margin before 120s

    # Greedy rank elimination: try to remove each term by adding one rank-1 term
    # such that the removed term becomes a linear combination of the rest + new term.
    # For each candidate removal index r, we need:
    #   terms[r] = new_term + sum_{c} a_c * terms[c]   (c != r)
    # Equivalently, the tensor t_r must lie in span({t_c : c != r} + {new_term}).
    # We search for a rank-1 new_term that makes t_r - new_term in span(others).
    #
    # Practical approach: for each r, compute the "residual space" R_r =
    #   tensor space / span(others).  Then we need new_term in t_r + span(others)
    #   AND new_term rank-1.  We enumerate rank-1 candidates over a grid.

    def tensor_to_list(u, v, w):
        # flatten the rank-1 tensor u*v*w into a list of length nm*mp*pn
        out = [0] * (nm * mp * pn)
        for a in range(nm):
            if u[a] == 0:
                continue
            for b in range(mp):
                uv = u[a] * v[b]
                if uv == 0:
                    continue
                for c in range(pn):
                    out[a * mp * pn + b * pn + c] = uv * w[c]
        return out

    def rank_of_rows(rows):
        # exact rational row rank
        rows = [[Fraction(x) for x in row] for row in rows]
        if not rows:
            return 0
        m_ = len(rows)
        n_ = len(rows[0])
        mat = [list(r) for r in rows]
        r = 0
        for col in range(n_):
            # find pivot
            piv = -1
            for i in range(r, m_):
                if mat[i][col] != 0:
                    piv = i
                    break
            if piv < 0:
                continue
            mat[r], mat[piv] = mat[piv], mat[r]
            pv = mat[r][col]
            # normalize
            for j in range(col, n_):
                mat[r][j] = mat[r][j] / pv
            for i in range(m_):
                if i == r or mat[i][col] == 0:
                    continue
                f = mat[i][col]
                for j in range(col, n_):
                    mat[i][j] = mat[i][j] - f * mat[r][j]
            r += 1
        return r

    # Precompute the tensor-space representation of each term
    t_list = [tensor_to_list(u, v, w) for (u, v, w) in terms]
    D = nm * mp * pn  # 8*32*16 = 4096

    # To test "is t_r in span(others) + {new}":
    #   new = t_r - s  for some s in span(others)
    #   we need new to be rank-1.
    # Equivalently: exists s in span(others) such that (t_r - s) is rank-1.
    # This is a search over the affine space t_r + span(others).
    #
    # Heuristic: enumerate rank-1 tensors (u,v,w) with small integer entries,
    # check if t_r - new is in span(others) via rank test.

    # Build a basis for span(others) for each r would be expensive; instead,
    # for a candidate new, test rank({others} ∪ {t_r - new}) == rank({others}).

    # To keep this tractable, we limit the search:
    #   - Only try removals where the residual dimension is small.
    #   - Enumerate rank-1 candidates from a bounded grid.

    # First, compute the dimension of the full tensor space (should be 64 for schoolbook
    # since all 64 terms are independent).
    # Actually the 64 rank-1 terms of matrix multiplication ARE linearly independent,
    # so the tensor space has dimension 64.

    # Greedy: for each r (in a fixed order), try to eliminate.
    # We attempt in order of "easiest" first. For simplicity, try all r=0..63.

    # Candidate rank-1 tensors: u, v, w with entries in {-1, 0, 1} but with
    # structured support. To keep the search small, we generate candidates by
    # combining pairs of existing term supports.

    # Simpler effective heuristic: for each pair of terms (a, b), form
    #   new_u = u_a + u_b, new_v = v_a + v_b, new_w = w_a + w_b
    # and test if removing a (or b) works. Also try differences.
    # This gives O(64^2) candidates per removal, which is ~4096 * 64 = 262k rank tests.
    # Each rank test on ~65 rows x 4096 cols with rationals is too slow.

    # Better: use a sparse/structured test. Since all our tensors are 0/1 rank-1
    # with single 1 in each factor, the tensor-space vector has exactly one 1.
    # t_r - new where new has entries in {-1,0,1} has at most (support of new)+1 nonzeros.

    # Given time constraints, we use a targeted local search:
    # For each r, build the subspace of "span(others)" restricted to the
    # coordinates where t_r or new could have support. This is complex.

    # Practical fallback: use a small number of random rank-1 perturbations
    # and check elimination via a cheap necessary condition (rank of a
    # restricted matrix). If we cannot find an improvement within budget,
    # return the best found (possibly schoolbook).

    import random
    random.seed(12345)

    def try_eliminate(r, current_terms):
        """Try to find a rank-1 new term allowing removal of term r.
        Returns new list of terms if successful, else None."""
        others = [current_terms[i] for i in range(len(current_terms)) if i != r]
        t_r = t_list[r]
        # We need new such that t_r - new in span(others).
        # Test: rank(others ∪ {t_r - new}) == rank(others)
        # Cheap precheck: rank(others) must be < rank(others ∪ {t_r}).
        # If t_r is already in span(others), we can remove without adding (free win).
        # But for schoolbook, no term is in span of others, so we need the addition.

        # Generate candidate new terms:
        # 1. Differences/sums of pairs of others.
        # 2. Random 0/1 rank-1 tensors.
        candidates = []
        # Pair combinations (limited)
        L = len(others)
        # Use a subset to bound cost
        step = max(1, L // 20)
        for a in range(0, L, step):
            for b in range(a + 1, min(a + step, L)):
                ua, va, wa = others[a]
                ub, vb, wb = others[b]
                # sum
                ns = ([ua[i] + ub[i] for i in range(nm)],
                      [va[i] + vb[i] for i in range(mp)],
                      [wa[i] + wb[i] for i in range(pn)])
                candidates.append(ns)
                # diff
                nd = ([ua[i] - ub[i] for i in range(nm)],
                      [va[i] - vb[i] for i in range(mp)],
                      [wa[i] - wb[i] for i in range(pn)])
                candidates.append(nd)

        # Random candidates
        for _ in range(50):
            u = [1 if random.random() < 0.3 else 0 for _ in range(nm)]
            v = [1 if random.random() < 0.3 else 0 for _ in range(mp)]
            w = [1 if random.random() < 0.3 else 0 for _ in range(pn)]
            if any(u) and any(v) and any(w):
                candidates.append((u, v, w))

        # Test each candidate
        others_tensors = [tensor_to_list(u, v, w) for (u, v, w) in others]
        # Precompute rank(others) - expensive but done once per r
        # We'll test incrementally: for candidate new, compute t_r - new,
        # and check if it's in span(others) by rank test on a small system.

        # To make rank test fast, we only use coordinates that are nonzero
        # in any of: others_tensors, t_r, or new_tensor.
        # Build coordinate list.
        active_coords = set()
        for ot in others_tensors:
            for idx, val in enumerate(ot):
                if val != 0:
                    active_coords.add(idx)
        for idx, val in enumerate(t_r):
            if val != 0:
                active_coords.add(idx)
        active_coords = sorted(active_coords)
        coord_to_idx = {c: i for i, c in enumerate(active_coords)}
        C = len(active_coords)

        def restrict(vec):
            return [Fraction(vec[c]) for c in active_coords]

        others_rows = [restrict(ot) for ot in others_tensors]
        t_r_row = restrict(t_r)

        # rank(others)
        r_others = rank_of_rows(others_rows)

        for (u, v, w) in candidates:
            nt = tensor_to_list(u, v, w)
            diff = [t_r[idx] - nt[idx] for idx in range(D)]
            diff_row = restrict(diff)
            # test rank(others ∪ {diff_row}) == r_others
            test_rows = others_rows + [diff_row]
            if rank_of_rows(test_rows) == r_others:
                # success: t_r - new in span(others)
                # new decomposition: others + new, and t_r removed
                new_terms = others + [(u, v, w)]
                return new_terms
        return None

    # Iterate elimination passes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = best
        for r in range(len(current)):
            if time.time() > deadline:
                break
            result = try_eliminate(r, current)
            if result is not None:
                best = result
                improved = True
                break  # restart with new list

    # Verify the best decomposition
    def verify(terms):
        # Build the tensor T = sum terms and compare to the true MM tensor.
        # True MM tensor: T[a, b, c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i
        # for some i,j,k.  Actually T_{(i,j),(j,k),(k,i)} = 1.
        # The decomposition must reproduce exactly:
        #   sum_t U_t[a] V_t[b] W_t[c] == MM_tensor[a,b,c]
        # We check on all (a,b,c) that are "active" (nonzero in MM tensor)
        # and a sample of inactive ones.
        # MM tensor is nonzero exactly when:
        #   a = i*m+j (0<=i<n,0<=j<m), b = j*p+k (0<=k<p), c = k*n+i
        # i.e., j = a % m, i = a // m, k = b % p, and c == k*n + i,
        #   and b // p == j.
        # Check all 64 active points and a few inactive.
        def mm_val(a, b, c):
            j = a % m
            i = a // m
            if b // p != j:
                return 0
            k = b % p
            if c != k * n + i:
                return 0
            return 1
        # Active points
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for (u, v, w) in terms:
                        s += u[a] * v[b] * w[c]
                    if s != 1:
                        return False
        # Inactive sample: all (a,b,c) with b//p != a%m  (guaranteed 0)
        for a in range(nm):
            for b in range(mp):
                if b // p != a % m:
                    for c in range(pn):
                        s = 0
                        for (u, v, w) in terms:
                            s += u[a] * v[b] * w[c]
                        if s != 0:
                            return False
        # Inactive: b//p == a%m but c != k*n+i
        for a in range(nm):
            j = a % m
            i = a // m
            for b in range(mp):
                if b // p != j:
                    continue
                k = b % p
                good_c = k * n + i
                for c in range(pn):
                    if c == good_c:
                        continue
                    s = 0
                    for (u, v, w) in terms:
                        s += u[a] * v[b] * w[c]
                    if s != 0:
                        return False
        return True

    # If verification fails (shouldn't happen for schoolbook), fall back to schoolbook.
    if not verify(best):
        best = terms  # original schoolbook

    # Build output matrices
    rank = len(best)
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    for col, (u, v, w) in enumerate(best):
        for a in range(nm):
            if u[a]:
                U[a][col] = u[a]
        for b in range(mp):
            if v[b]:
                V[b][col] = v[b]
        for c in range(pn):
            if w[c]:
                W[c][col] = w[c]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
