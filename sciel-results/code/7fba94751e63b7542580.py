import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, R):
    """Exact verification that the factorization reproduces the mult tensor.

    Contract each rank-1 term (a sparse vector a over A-rows, b over B-rows,
    and the transpose-convention output w over C-rows) to build T, then check
    T[i,j,k] == (A[i,j]*B[j,k] -> C[i,k]) for every (i,j,k).
    """
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    T = np.zeros((n, m, p), dtype=object)
    for r in range(R):
        a = U[:, r]
        b = V[:, r]
        w = W[:, r]
        az = np.nonzero(a)[0]
        bz = np.nonzero(b)[0]
        wz = np.nonzero(w)[0]
        for ai in az:
            av = a[ai]
            for bi in bz:
                bv = b[bi]
                val = av * bv
                if val == 0:
                    continue
                i, j = divmod(ai, m)
                j2, k = divmod(bi, p)
                if j2 != j:
                    continue
                for wi in wz:
                    coeff = w[wi]
                    k2, ii = divmod(wi, n)
                    T[ii, j, k2] += val * coeff
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1:
                    return False
    return True


def _cleanup_residue(U, V, W, n, m, p, R):
    """Append rank-1 terms to fix any residual so the decomposition is exact.

    This guarantees a valid (possibly larger-rank) exact factorization.
    """
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    T = np.zeros((n, m, p), dtype=object)
    for r in range(R):
        a = np.asarray(U[:, r], dtype=object)
        b = np.asarray(V[:, r], dtype=object)
        w = np.asarray(W[:, r], dtype=object)
        az = np.nonzero(a)[0]
        bz = np.nonzero(b)[0]
        wz = np.nonzero(w)[0]
        for ai in az:
            av = a[ai]
            for bi in bz:
                bv = b[bi]
                val = av * bv
                if val == 0:
                    continue
                i, j = divmod(ai, m)
                j2, k = divmod(bi, p)
                if j2 != j:
                    continue
                for wi in wz:
                    coeff = w[wi]
                    k2, ii = divmod(wi, n)
                    T[ii, j, k2] += val * coeff
    for i in range(n):
        for j in range(m):
            for k in range(p):
                resid = 1 - T[i, j, k]
                if resid != 0:
                    U.append([0.0] * R + [float(resid)])
                    V.append([0.0] * R + [1.0])
                    W.append([0.0] * R + [1.0])
                    T[i, j, k] = 1
                    R += 1
    return U, V, W, R


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 120.0
    safety = 10.0

    nA = n * m      # 15
    nB = m * p      # 35
    nC = p * n      # 21

    def time_ok():
        return (deadline - time.time()) > safety

    def idx_A(i, j):
        return i * m + j

    def idx_B(j, k):
        return j * p + k

    def idx_C(k, i):
        return k * n + i

    # ---- Candidate 1: schoolbook (always exact, rank 105) ----
    def schoolbook():
        R = n * m * p
        U = [[0.0] * R for _ in range(nA)]
        V = [[0.0] * R for _ in range(nB)]
        W = [[0.0] * R for _ in range(nC)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_A(i, j)][col] = 1.0
                    V[idx_B(j, k)][col] = 1.0
                    W[idx_C(k, i)][col] = 1.0
                    col += 1
        return U, V, W, R

    U_best, V_best, W_best, R_best = schoolbook()

    # ---- Candidate 2: sparse greedy rank-1 update search ----
    # Start from the schoolbook tensor (all ones) and remove terms by
    # replacing groups with cheaper rank-1 terms.  We search for a set of
    # rank-1 terms (each sparse) whose contraction equals the tensor, then
    # greedily keep a minimal verified subset.

    # Build the mult tensor as a set of (i,j,k) -> 1.
    ones = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    # A rank-1 term is represented by (a, b, w) where
    #   a is a set of (i,j) entries, b a set of (j,k) entries, w a set of (k,i).
    # Its contribution to T[i,j,k] is sum over shared j of a[i,j]*b[j,k] * w[k,i].
    # For a *clean* rank-1 term we want a[i,j]*b[j,k]*w[k,i] = 1 for exactly the
    # (i,j,k) in its support and 0 elsewhere.  The simplest clean term covering a
    # single (i0,j0,k0): a[(i0,j0)]=1, b[(j0,k0)]=1, w[(k0,i0)]=1.

    # Greedy: try to cover the tensor with as few rank-1 terms as possible.
    # A rank-1 term can cover a set S of (i,j,k) if there exist scalar factors
    # u[i],v[j],s[k] (with per-term structure) such that product = 1 on S, 0 else.
    # The most general exact rank-1 term has form:
    #   a[i,j] = alpha_i * beta_j,  b[j,k] = gamma_j * delta_k,  w[k,i] = eps_k * zeta_i
    #   => T[i,j,k] = alpha_i*beta_j*gamma_j*delta_k*eps_k*zeta_i
    #              = (alpha_i*zeta_i) * (beta_j*gamma_j) * (delta_k*eps_k)
    # This is a rank-1 outer product over (i,j,k)!  So the tensor (all ones) is
    # itself rank 1 in the (i,j,k) outer-product sense... but that's the
    # multiplicative decomposition, NOT the bilinear mult decomposition.
    #
    # The bilinear constraint couples U(i,j), V(j,k), W(k,i) through the SAME
    # index j for U,V and the SAME i,k structure for W.  A valid rank-1 term must
    # satisfy: the product a[i,j]*b[j,k]*w[k,i] is nonzero only when the indices
    # are consistent.  The all-ones tensor over (i,j,k) is NOT a single valid
    # rank-1 bilinear term (that would require a[i,j]*b[j,k]*w[k,i]=1 for ALL
    # i,j,k which is impossible with the coupling).  Hence schoolbook rank 105 is
    # a natural upper bound and lower-rank exact decompositions are nontrivial.

    # Given the difficulty and strict time budget, we attempt a limited
    # combinatorial improvement: try replacing pairs of schoolbook terms that
    # share structure with a single term when valid, then verify.

    # Represent current factorization as list of terms, each = dict of
    # (i,j)->val, (j,k)->val, (k,i)->val.
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append({
                    'a': {idx_A(i, j): 1.0},
                    'b': {idx_B(j, k): 1.0},
                    'w': {idx_C(k, i): 1.0},
                })

    def contract_terms(terms):
        T = np.zeros((n, m, p), dtype=object)
        for t in terms:
            for ai, av in t['a'].items():
                for bi, bv in t['b'].items():
                    val = av * bv
                    if val == 0:
                        continue
                    i, j = divmod(ai, m)
                    j2, k = divmod(bi, p)
                    if j2 != j:
                        continue
                    for wi, wv in t['w'].items():
                        k2, ii = divmod(wi, n)
                        T[ii, j, k2] += val * wv
        return T

    def terms_to_matrices(terms):
        R = len(terms)
        U = [[0.0] * R for _ in range(nA)]
        V = [[0.0] * R for _ in range(nB)]
        W = [[0.0] * R for _ in range(nC)]
        for r, t in enumerate(terms):
            for idx, val in t['a'].items():
                U[idx][r] = float(val)
            for idx, val in t['b'].items():
                V[idx][r] = float(val)
            for idx, val in t['w'].items():
                W[idx][r] = float(val)
        return U, V, W, R

    # Greedy merge: try to combine two terms into one when their product
    # structure allows.  Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can
    # merge into one term only if there is a single (a,b,w) whose support covers
    # both with value 1.  This requires a[i,j]*b[j,k]*w[k,i] = 1 for both
    # (i1,j1,k1) and (i2,j2,k2) and 0 elsewhere.  With the coupling, the only
    # way to hit two distinct (i,j,k) with one rank-1 term is if they share the
    # necessary indices.  We try a small set of merges and verify exactly.
    def try_merge(terms):
        new_terms = list(terms)
        changed = True
        while changed and time_ok():
            changed = False
            # Try merging term r with term s
            for r in range(len(new_terms)):
                if not time_ok():
                    return None
                for s in range(r + 1, len(new_terms)):
                    tr = new_terms[r]
                    ts = new_terms[s]
                    # Build a merged candidate term: union supports.
                    # The merged term is valid only if a[i,j]*b[j,k]*w[k,i] equals
                    # the sum of the two originals everywhere.  We attempt the
                    # naive union and then verify; if invalid, skip.
                    merged_a = dict(tr['a'])
                    merged_b = dict(tr['b'])
                    merged_w = dict(tr['w'])
                    # Add term s entries (this is a heuristic union; exactness is
                    # checked by full verification below).
                    for idx, val in ts['a'].items():
                        merged_a[idx] = merged_a.get(idx, 0.0) + val
                    for idx, val in ts['b'].items():
                        merged_b[idx] = merged_b.get(idx, 0.0) + val
                    for idx, val in ts['w'].items():
                        merged_w[idx] = merged_w.get(idx, 0.0) + val
                    candidate = [{'a': tr['a'], 'b': tr['b'], 'w': tr['w']}
                                 for x in new_terms[:r]]
                    candidate.append({'a': merged_a, 'b': merged_b, 'w': merged_w})
                    candidate += [{'a': x['a'], 'b': x['b'], 'w': x['w']}
                                  for x in new_terms[r + 1:s]]
                    candidate += [{'a': x['a'], 'b': x['b'], 'w': x['w']}
                                  for x in new_terms[s + 1:]]
                    Tc = contract_terms(candidate)
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if Tc[i, j, k] != 1:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        new_terms = candidate
                        changed = True
                        break
                if changed:
                    break
        return new_terms

    merged = try_merge(terms)
    if merged is not None and len(merged) < R_best:
        Um, Vm, Wm, Rm = terms_to_matrices(merged)
        if _verify(Um, Vm, Wm, n, m, p, Rm):
            U_best, V_best, W_best, R_best = Um, Vm, Wm, Rm

    # Final safety: ensure exactness of whatever we keep.
    if not _verify(U_best, V_best, W_best, n, m, p, R_best):
        U_best, V_best, W_best, R_best = _cleanup_residue(
            U_best, V_best, W_best, n, m, p, R_best)
        if not _verify(U_best, V_best, W_best, n, m, p, R_best):
            # Last resort: pure schoolbook.
            U_best, V_best, W_best, R_best = schoolbook()

    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
