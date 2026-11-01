"""Matrix multiplication 2,3,7 rank-minimal exact tensor decomposition.

Structural approach (different from gradient descent):
  * Enumerate sparse integer/half-integer factor candidates.
  * Precompute, for every required output index (k*n+i), the candidate
    term-sets that cover it.
  * Solve a multi-cover (set cover) problem with greedy + local repair.
"""
import time
import random
import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 3, 7]


def _halfs(x):
    y = np.round(x * 2.0) / 2.0
    return np.where(np.abs(y - np.round(y)) < 1e-9, np.round(y), y)


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0

    def verify(U, V, W, tol=1e-7):
        try:
            U_3d = U.reshape(n, m, -1)
            V_3d = V.reshape(m, p, -1)
            W_3d = W.reshape(p, n, -1)
            T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            return bool(np.all(np.abs(T - 1.0) < tol))
        except Exception:
            return False

    # ---------- build candidate pool of sparse integer factors ----------
    def make_pool(n, m, p, halfs=False):
        vals = [1, 2] if not halfs else [1, 2, 3]
        u_list, v_list, w_list = [], [], []
        for i in range(n):
            for j in range(m):
                r = i * m + j
                for s in range(1, 4):
                    for a in (1, -1):
                        for v in vals:
                            u = np.zeros(n * m); u[r] = a * v
                            u_list.append(u)
        for j in range(m):
            for k in range(p):
                c = j * p + k
                for s in range(1, 4):
                    for a in (1, -1):
                        for v in vals:
                            vmat = np.zeros(m * p); vmat[c] = a * v
                            v_list.append(vmat)
        for k in range(p):
            for i in range(n):
                o = k * n + i
                for s in range(1, 4):
                    for a in (1, -1):
                        for v in vals:
                            w = np.zeros(p * n); w[o] = a * v
                            w_list.append(w)
        return u_list, v_list, w_list

    def precompute(u_list, v_list, w_list):
        # For each (i,j): list of (u_idx, v_idx) with U[i,j]*V[j,k]*W[k,i] pattern
        # We'll build, for each required out (k,i), the set of candidate terms
        # that cover it. A candidate term is (u, v, w).
        # Instead of all triple combinations (huge), we exploit structure:
        # U has single nonzero at (i,j), V single at (j,k), W single at (o=(k,i)).
        # For term to contribute to T[i,j,k]: need U row (i,j), V row (j,k), W col (k,i).
        # So candidate term = (u at (i,j), v at (j,k), w at (k,i)).
        # Build list of candidate terms keyed by the out (k,i) they cover.
        # Map: for each (i,j) -> u indices; (j,k) -> v indices; (k,i) -> w indices.
        u_by = {}
        for idx, u in enumerate(u_list):
            nz = np.nonzero(u)[0]
            if len(nz) == 1:
                r = int(nz[0]); u_by.setdefault(r, []).append(idx)
        v_by = {}
        for idx, v in enumerate(v_list):
            nz = np.nonzero(v)[0]
            if len(nz) == 1:
                c = int(nz[0]); v_by.setdefault(c, []).append(idx)
        w_by = {}
        for idx, w in enumerate(w_list):
            nz = np.nonzero(w)[0]
            if len(nz) == 1:
                o = int(nz[0]); w_by.setdefault(o, []).append(idx)

        # Each (i,j,k) requires: pick u from u_by[i*m+j], v from v_by[j*p+k],
        # w from w_by[k*n+i], such that product = 1 (since T=1).
        # We'll represent a "term" as (u_idx, v_idx, w_idx).
        # For solving cover: the out (k,i) is covered by a term iff
        #   w_idx in w_by[k*n+i] and (u,v,w) consistent.
        # Build required outputs: for each (k,i) the product must equal 1.
        # Required: for each (i,j,k): T=1.
        # Represent as: needed out (k,i) with required value = sum over j of
        #   U[i,j,c]*V[j,k,c]*W[k,i,c] = 1.
        # This is a sum, not a single product. So the "cover" framing is
        # approximate. We'll instead directly search for full candidate terms
        # (u,v,w) and check they assemble T=1.

        return u_by, v_by, w_by, u_list, v_list, w_list

    # ---------- fallback: schoolbook ----------
    def build_schoolbook():
        terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        rank = len(terms)
        U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
        for c, (i, j, k) in enumerate(terms):
            U[i * m + j, c] = 1.0; V[j * p + k, c] = 1.0; W[k * n + i, c] = 1.0
        return U, V, W

    U, V, W = build_schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    if not verify(best_U, best_V, best_W):
        raise ValueError("schoolbook failed")

    # ---------- candidate-term enumeration with multi-cover greedy ----------
    # Enumerate all single-(i,j) U vectors, single-(j,k) V, single-(k,i) W.
    # A candidate term is (u, v, w). Its contribution to T[i,j,k]:
    #   nonzero only when u row == i*m+j, v row == j*p+k, w col == k*n+i,
    #   and equals u_val * v_val * w_val.
    # We want to select a multiset of terms so that for each (i,j,k) the sum
    # of contributions over selected terms == 1.
    #
    # Group candidate terms by (i,j,k) they touch. For a given (i,j,k), the
    # terms that touch it are all (u in U[i,j], v in V[j,k], w in W[k,i]).
    # Each such term contributes one product to T[i,j,k]. So T[i,j,k] equals
    # the count-sum of selected terms in that group with their product signs.
    #
    # This is a system: for each (i,j,k), sum over selected terms in group G_ijk
    # of (u_val*v_val*w_val) == 1. But a term belongs to exactly one group
    # (since u fixes (i,j), v fixes (j,k), w fixes (k,i)). So groups are
    # disjoint! Thus each (i,j,k) is solved independently:
    #   choose a multiset of (u,v,w) in G_ijk whose product-sum == 1.
    # That means the rank is the SUM over (i,j,k) of min terms per group.
    # For a single group, min terms to make product-sum = 1:
    #   - if any term has product 1: use 1 term.  (e.g. u=1,v=1,w=1)
    #   - so each group needs >=1 term, and 1 term suffices (schoolbook).
    # Therefore the absolute minimum rank with this single-nonzero-per-factor
    # structure is n*m*p = 42. To go below 42, factors must have MULTIPLE
    # nonzeros (shared structure), which is what real low-rank decompositions do.

    # So single-nonzero candidates cannot beat schoolbook. We need multi-nonzero
    # candidates. We'll enumerate candidates where U has nonzeros across a set of
    # (i,j), V across a set of (j,k), W across a set of (k,i), and the outer
    # product sums to the correct T.

    # ---------- multi-nonzero candidate search via structured random init ----------
    # Try to find a rank-r decomposition with factors that have a few nonzeros.
    def rand_decompose(rank, seed, max_nnz=4):
        random.seed(seed)
        np.random.seed(seed)
        U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
        vals = [1, -1, 2, -2, 0.5, -0.5]
        for c in range(rank):
            # pick a random "core" (i,j,k) and add a few related indices
            i = random.randrange(n); j = random.randrange(m); k = random.randrange(p)
            U[idx] = 1.0 if False else 0.0  # placeholder
        # Simpler: random sparse integer factors
        def sparse_mat(rows):
            mat = np.zeros(rows)
            for _ in range(random.randint(1, max_nnz)):
                mat[random.randrange(rows)] = random.choice(vals)
            return mat
        for c in range(rank):
            U[:, c] = sparse_mat(n * m)
            V[:, c] = sparse_mat(m * p)
            W[:, c] = sparse_mat(p * n)
        return U, V, W

    def sse(U, V, W):
        T = np.einsum('ijc,jkc,kic->ijk', U.reshape(n, m, -1),
                      V.reshape(m, p, -1), W.reshape(p, n, -1))
        return float(np.sum((T - 1.0) ** 2))

    # coordinate-descent refinement on sparse structure
    def refine(U, V, W, iters, lr):
        for it in range(iters):
            if time.time() > deadline - 15:
                break
            T = np.einsum('ijc,jkc,kic->ijk', U.reshape(n, m, -1),
                          V.reshape(m, p, -1), W.reshape(p, n, -1))
            E = T - 1.0
            GU = np.einsum('jkc,kic,ijk->ijc', V.reshape(m, p, -1),
                           W.reshape(p, n, -1), E)
            GV = np.einsum('ijc,kic,ijk->jkc', U.reshape(n, m, -1),
                           W.reshape(p, n, -1), E)
            GW = np.einsum('ijc,jkc,ijk->kic', U.reshape(n, m, -1),
                           V.reshape(m, p, -1), E)
            U -= lr * GU.reshape(n * m, -1)
            V -= lr * GV.reshape(m * p, -1)
            W -= lr * GW.reshape(p * n, -1)
        return U, V, W

    # search for rank < 42 with sparse init + refine + snap
    rng = random.Random(123)
    for target_rank in range(41, 36, -1):
        if time.time() > deadline - 15:
            break
        found = False
        for attempt in range(40):
            if time.time() > deadline - 15:
                break
            U_c, V_c, W_c = rand_decompose(target_rank, 1000 + attempt, max_nnz=3)
            # anneal learning rate
            lr = 0.05
            for ph in range(30):
                if time.time() > deadline - 15:
                    break
                U_c, V_c, W_c = refine(U_c, V_c, W_c, iters=20, lr=lr)
                cur = sse(U_c, V_c, W_c)
                if cur < 1e-8:
                    break
                lr *= 0.93
            U_c, V_c, W_c = _halfs(U_c), _halfs(V_c), _halfs(W_c)
            if verify(U_c, V_c, W_c):
                best_rank = target_rank
                best_U, best_V, best_W = U_c, V_c, W_c
                found = True
                break
        if found:
            continue
        # if we can't do this rank, continue trying next (lower) rank
    # final verification / fallback
    if not verify(best_U, best_V, best_W):
        U, V, W = build_schoolbook()
        best_rank = n * m * p
        best_U, best_V, best_W = U, V, W
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist(),
    }
