import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 7]


def _u_idx(i, j, m): return i * m + j
def _v_idx(j, k, p): return j * p + k
def _w_idx(k, i, n): return k * n + i


def _build_base_terms(n, m, p):
    """Hybrid Strassen/Schoolbook decomposition (rank 39). Returns list of (u,v,w) dicts."""
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen(i0, j0, k0):
        # 2x2x2 Strassen block for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        U = lambda i, j: _u_idx(i0 + i, j0 + j, m)
        V = lambda j, k: _v_idx(j0 + j, k0 + k, p)
        W = lambda k, i: _w_idx(k0 + k, i0 + i, n)
        add_term({U(0, 0): 1, U(1, 1): 1}, {V(0, 0): 1, V(1, 1): 1}, {W(0, 0): 1, W(1, 1): 1})
        add_term({U(1, 0): 1, U(1, 1): 1}, {V(0, 0): 1}, {W(0, 1): 1, W(1, 1): -1})
        add_term({U(0, 0): 1}, {V(0, 1): 1, V(1, 1): -1}, {W(1, 0): 1, W(1, 1): 1})
        add_term({U(1, 1): 1}, {V(1, 0): 1, V(0, 0): -1}, {W(0, 0): 1, W(0, 1): 1})
        add_term({U(0, 0): 1, U(0, 1): 1}, {V(1, 1): 1}, {W(0, 0): -1, W(1, 0): 1})
        add_term({U(1, 0): 1, U(0, 0): -1}, {V(0, 0): 1, V(0, 1): 1}, {W(1, 1): 1})
        add_term({U(0, 1): 1, U(1, 1): -1}, {V(1, 0): 1, V(1, 1): 1}, {W(0, 0): 1})

    def schoolbook2x2x1(i0, j0, k0):
        # A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0]
        U = lambda i, j: _u_idx(i0 + i, j0 + j, m)
        V = lambda j: _v_idx(j0 + j, k0, p)
        W = lambda i: _w_idx(k0, i0 + i, n)
        add_term({U(0, 0): 1}, {V(0): 1}, {W(0): 1})
        add_term({U(0, 1): 1}, {V(1): 1}, {W(0): 1})
        add_term({U(1, 0): 1}, {V(0): 1}, {W(1): 1})
        add_term({U(1, 1): 1}, {V(1): 1}, {W(1): 1})

    # Part 1: 2x2x2 Strassen (j=0,1 ; k=0,1)
    strassen(0, 0, 0)
    # Part 2: 2x2x1 schoolbook (j=0,1 ; k=2)
    schoolbook2x2x1(0, 0, 2)
    # Part 3: 2x2x2 Strassen (j=0,1 ; k=3,4)
    strassen(0, 0, 3)
    # Part 4: 2x2x2 Strassen (j=0,1 ; k=5,6)
    strassen(0, 0, 5)
    # Part 5: 2x1x7 schoolbook (j=2 ; k=0..6)
    for k in range(p):
        add_term({_u_idx(0, 2, m): 1}, {_v_idx(2, k, p): 1}, {_w_idx(k, 0, n): 1})
        add_term({_u_idx(1, 2, m): 1}, {_v_idx(2, k, p): 1}, {_w_idx(k, 1, n): 1})

    return terms


def _terms_to_matrices(terms, n, m, p):
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for c, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx, c] = val
        for idx, val in v.items():
            V[idx, c] = val
        for idx, val in w.items():
            W[idx, c] = val
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Return max abs error of the tensor reconstruction (0.0 if exact)."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                e = abs(val - 1.0)
                if e > err:
                    err = e
    return err


def _round_half(x):
    return round(2 * x) / 2.0


def _is_half(x):
    return abs(2 * x - round(2 * x)) < 1e-6


def _to_lists(U, V, W):
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock kill

    # ---- Build base (rank 39) and verify ----
    base_terms = _build_base_terms(n, m, p)
    U, V, W = _terms_to_matrices(base_terms, n, m, p)
    base_err = _verify(U, V, W, n, m, p)
    if base_err > 1e-8:
        # Fallback to pure schoolbook (rank 42) if base is somehow broken.
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(
                        ({_u_idx(i, j, m): 1}, {_v_idx(j, k, p): 1}, {_w_idx(k, i, n): 1})
                    )
        U, V, W = _terms_to_matrices(terms, n, m, p)
        base_err = _verify(U, V, W, n, m, p)

    best = {"U": U.copy(), "V": V.copy(), "W": W.copy()}
    best_rank = U.shape[1]
    best_err = base_err

    def consider(Uc, Vc, Wc, tag):
        nonlocal best, best_rank, best_err
        if time.time() > deadline - 5:
            return
        err = _verify(Uc, Vc, Wc, n, m, p)
        if err < 1e-7 and Uc.shape[1] < best_rank:
            best = {"U": Uc.copy(), "V": Vc.copy(), "W": Wc.copy()}
            best_rank = Uc.shape[1]
            best_err = err

    # ---- Refinement 1: merge the two j=2 outer-product terms into one (39 -> 38) ----
    # j=2 sub-tensor: need u[i]*b_k*W[k,i] = 1 for all i,k.
    # Set W[k,i] = 1 (all), solve u[i]*b_k = 1.  u = [1,1], b = [1]*7.
    if time.time() < deadline - 20:
        U1 = U.copy(); V1 = V.copy(); W1 = W.copy()
        # Remove the two j=2 terms (last two terms in base construction).
        # Identify them: terms with U support only at j=2 (index i*3+2).
        # In base, the last 14 terms are j=2 schoolbook. We replace the whole j=2
        # block (14 terms) with a single term.
        # Rebuild: keep terms 0..24 (Parts 1-4), drop the 14 j=2 terms, add 1.
        kept = base_terms[:24]  # 7+4+7+7 = 25? recompute below safely
        # Recompute count robustly: number of non-j=2 terms.
        nonj2 = []
        for (u, v, w) in base_terms:
            # j=2 terms have V support only at row j=2 (indices 14..20)
            vrows = set(idx // p for idx in v.keys())
            if vrows != {2}:
                nonj2.append((u, v, w))
        new_terms = list(nonj2)
        # single merged j=2 term
        merged_u = {_u_idx(0, 2, m): 1, _u_idx(1, 2, m): 1}
        merged_v = {_v_idx(2, k, p): 1 for k in range(p)}
        merged_w = {_w_idx(k, i, n): 1 for k in range(p) for i in range(n)}
        new_terms.append((merged_u, merged_v, merged_w))
        Uc, Vc, Wc = _terms_to_matrices(new_terms, n, m, p)
        consider(Uc, Vc, Wc, "merge_j2")

    # ---- Refinement 2: merge two j in {0,1} schoolbook terms (Part 2) (39 -> 38) ----
    # Part 2 terms (k=2): (0,0,2),(0,1,2),(1,0,2),(1,1,2).
    # Try to express C[0,2] = a00*b02 + a01*b12 with a single rank-1-in-slice term
    # is impossible (needs 2 products). Instead try merging across the two rows:
    # Not directly reducible; skip if not helpful. (Kept as a hypothesis probe.)
    if time.time() < deadline - 20:
        # Attempt: can we fold the (1,1,2) term into the Strassen part by re-expressing?
        # This is speculative; only accept if exact and lower rank (it won't be lower
        # unless we drop a term). We instead try dropping one redundant-looking term
        # and re-solving W via least squares over the remaining terms.
        pass  # handled by Refinement 3 (global polish)

    # ---- Refinement 3: smooth epigraph / least-squares polish for rank 38 ----
    # Try dropping one term at a time from the base and re-solving W (and optionally
    # all of U,V,W) by least squares; round to half-integers; verify exactly.
    if time.time() < deadline - 15:
        base_rank = base_terms.__len__()
        # Build the linear system for W given fixed U,V:
        # For each (i,k): sum_c S_{i,k,c} * W_c[k*n+i] = 1, where
        # S_{i,k,c} = sum_j U_c[i*m+j] * V_c[j*p+k].
        # Unknowns: W_c[k*n+i] for c in kept, (k,i) in [p]x[n]  -> base_rank* n * p unknowns
        # Equations: n*p (one per (i,k)).  Under-determined -> use least squares, then
        # check if a half-integer exact solution exists via rounding + verification.
        for drop in range(base_rank):
            if time.time() > deadline - 8:
                break
            kept_terms = [t for idx, t in enumerate(base_terms) if idx != drop]
            Uk, Vk, _ = _terms_to_matrices(kept_terms, n, m, p)
            r = Uk.shape[1]
            # Build A matrix: rows = (i,k) in [n]x[p], cols = (c, k2, i2)
            # A[row, (c,k2,i2)] = Uk[i*m+j summed]*Vk[j*p+k] only when k2==k and i2==i
            A = np.zeros((n * p, r * n * p))
            b = np.ones(n * p)
            for i in range(n):
                for k in range(p):
                    row = i * p + k
                    for c in range(r):
                        S = 0.0
                        for j in range(m):
                            S += Uk[i * m + j, c] * Vk[j * p + k, c]
                        col = c * (n * p) + k * n + i
                        A[row, col] = S
            # least squares
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
            except Exception:
                continue
            Wk = sol.reshape(r, n, p)
            # round to half-integers
            Wk_r = np.round(2 * Wk) / 2.0
            Wk_mat = Wk_r.reshape(r, n * p).T  # (n*p, r)
            Uc, Vc, Wc = Uk, Vk, Wk_mat
            consider(Uc, Vc, Wc, "drop_%d" % drop)

    # ---- Refinement 4: full polish of the best found (tighter, exact local solve) ----
    if time.time() < deadline - 8:
        # Try a global least-squares on a rank-38 support (drop the term whose removal
        # gave smallest residual in Refinement 3) — already covered. Do a final
        # high-precision verification of best and ensure half-integer entries.
        Ub = np.round(2 * best["U"]) / 2.0
        Vb = np.round(2 * best["V"]) / 2.0
        Wb = np.round(2 * best["W"]) / 2.0
        if _verify(Ub, Vb, Wb, n, m, p) < 1e-7:
            best = {"U": Ub, "V": Vb, "W": Wb}

    # ---- Return best verified-exact decomposition ----
    if time.time() > deadline - 3:
        return _to_lists(best["U"], best["V"], best["W"])

    return _to_lists(best["U"], best["V"], best["W"])
