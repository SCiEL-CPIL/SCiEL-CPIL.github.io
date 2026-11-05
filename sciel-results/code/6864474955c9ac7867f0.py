"""Search for a rank-minimal exact tensor decomposition of 3x5 * 5x7 matrix mult."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _build_target(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _reconstruct(U, V, W):
    # U (n*m, r), V (m*p, r), W (p*n, r)
    return np.einsum('ac,bc,dc->abd', U, V, W)

def _residual(U, V, W, T):
    return np.max(np.abs(_reconstruct(U, V, W) - T))

def _round_half(M):
    return np.round(2.0 * M) / 2.0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _tensor_contract0(T, u):
    return np.einsum('abd,a->bd', T, u)
def _tensor_contract1(T, v):
    return np.einsum('abd,b->ad', T, v)
def _tensor_contract2(T, w):
    return np.einsum('abd,d->ab', T, w)

def _best_rank1(T, iters=20, seed=0):
    rng = np.random.RandomState(seed)
    n0, n1, n2 = T.shape
    u = rng.randn(n0); v = rng.randn(n1); w = rng.randn(n2)
    for _ in range(iters):
        u = _tensor_contract1(_tensor_contract2(T, w), v)
        v = _tensor_contract0(_tensor_contract2(T, w), u)
        w = _tensor_contract0(_tensor_contract1(T, u), v)
        if np.linalg.norm(u) < 1e-12 or np.linalg.norm(v) < 1e-12 or np.linalg.norm(w) < 1e-12:
            break
        u /= np.linalg.norm(u); v /= np.linalg.norm(v); w /= np.linalg.norm(w)
    # scale so that the term is u⊗v⊗w with a chosen normalization
    return u, v, w

def _greedy_exact(T, max_terms=200, seed=0):
    rng = np.random.RandomState(seed)
    R = T.copy()
    terms_u, terms_v, terms_w = [], [], []
    for c in range(max_terms):
        if np.max(np.abs(R)) < 1e-9:
            break
        u, v, w = _best_rank1(R, iters=30, seed=seed + c)
        # choose scale: project to best scalar alpha minimizing ||R - alpha u⊗v⊗w||
        num = np.einsum('abd,abd', R, np.einsum('a,b,d->abd', u, v, w))
        den = np.einsum('abd,abd', u * v[:, None, None] * w[None, :, :],
                        u * v[:, None, None] * w[None, :, :])
        if den < 1e-12:
            continue
        alpha = num / den
        R = R - alpha * np.einsum('a,b,d->abd', u, v, w)
        terms_u.append(alpha * u); terms_v.append(v); terms_w.append(w)
    if np.max(np.abs(R)) > 1e-7:
        return None
    r = len(terms_u)
    U = np.column_stack(terms_u) if r else np.zeros((T.shape[0], 0))
    V = np.column_stack(terms_v) if r else np.zeros((T.shape[1], 0))
    W = np.column_stack(terms_w) if r else np.zeros((T.shape[2], 0))
    return U, V, W

def _local_reduce(U, V, W, T, deadline, max_steps=4000):
    # coordinate descent on entries to keep residual small, then try dropping columns
    best = (U.copy(), V.copy(), W.copy(), _residual(U, V, W, T))
    U, V, W = best[0].copy(), best[1].copy(), best[2].copy()
    rng = np.random.RandomState(123)
    r = U.shape[1]
    for step in range(max_steps):
        if time.time() > deadline:
            break
        # try dropping a random column if residual stays low after re-optimizing
        if r > 1 and rng.rand() < 0.3:
            c = rng.randint(r)
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            # quick re-opt: a few gradient steps on W only
            for _ in range(3):
                grad = np.einsum('abd,ac,bc->dc', _reconstruct(U2, V2, W2) - T, U2, V2)
                W2 -= 0.01 * grad
            res2 = _residual(U2, V2, W2, T)
            if res2 < 1e-6:
                U, V, W = U2, V2, W2
                r = U.shape[1]
                best = (U.copy(), V.copy(), W.copy(), res2)
                continue
        # small perturbation / coordinate update on a random entry of W
        d = rng.randint(W.shape[0]); c = rng.randint(W.shape[1])
        old = W[d, c]
        W[d, c] = old + rng.randn() * 0.05
        res = _residual(U, V, W, T)
        if res > 1e-5:
            W[d, c] = old
    return best

def construct():
    n, m, p = DIMENSIONS
    T = _build_target(n, m, p)
    start = time.time()
    deadline = start + 105.0  # safety margin before 120s / 150s kill

    # Guaranteed fallback
    U0, V0, W0 = _schoolbook(n, m, p)
    assert _residual(U0, V0, W0, T) < 1e-9
    best_rank = U0.shape[1]
    best = (U0, V0, W0)

    def try_record(U, V, W):
        nonlocal best_rank, best
        # round to half-integers and verify exactness
        Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
        if _residual(Ur, Vr, Wr, T) < 1e-9:
            r = Ur.shape[1]
            if r < best_rank:
                best_rank = r
                best = (Ur, Vr, Wr)

    # 1) Greedy exact decomposition attempts
    for seed in range(0, 6):
        if time.time() > deadline:
            break
        out = _greedy_exact(T, max_terms=120, seed=seed)
        if out is not None:
            U, V, W = out
            try_record(U, V, W)
            # local reduce on the (real) solution then record
            Ub, Vb, Wb, resb = _local_reduce(U, V, W, T, deadline, max_steps=800)
            try_record(Ub, Vb, Wb)

    # 2) Random restart: random U,V (integer-ish), solve W by least squares, round
    rng = np.random.RandomState(7)
    for r in range(35, min(105, best_rank)):
        if time.time() > deadline:
            break
        for restart in range(4):
            if time.time() > deadline:
                break
            U = rng.choice([-1, 0, 1], size=(n * m, r)).astype(float)
            V = rng.choice([-1, 0, 1], size=(m * p, r)).astype(float)
            # build linear system for W: unknowns W flattened (p*n)*r
            # eq for each (a,b,d): sum_c U[a,c] V[b,c] W[d,c] = T[a,b,d]
            Arows = []
            bvec = []
            # vectorize: for each c, coefficient on W[:,c] is U[:,c]*V[:,c] (length n*m*m*p)
            # We'll build a big sparse-ish system via dense for small sizes.
            # Unknown count = (p*n)*r. Equation count = (n*m)*(m*p)*(p*n) = 15*35*21=11025.
            # Build M (11025 x 21*r)
            M = np.zeros((n * m * m * p * p * n, p * n * r))
            bv = np.zeros(n * m * m * p * p * n)
            idx = 0
            for a in range(n * m):
                for b in range(m * p):
                    for d in range(p * n):
                        for c in range(r):
                            M[idx, d * r + c] = U[a, c] * V[b, c]
                        bv[idx] = T[a, b, d]
                        idx += 1
            sol, *_ = np.linalg.lstsq(M, bv, rcond=None)
            W = sol.reshape(p * n, r)
            res = _residual(U, V, W, T)
            if res < 1e-6:
                try_record(U, V, W)

    # 3) Local reduce on schoolbook (try to shave rank while staying exact)
    if time.time() < deadline:
        Ub, Vb, Wb, resb = _local_reduce(U0, V0, W0, T, deadline, max_steps=2000)
        try_record(Ub, Vb, Wb)

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
