import time
import numpy as np

DIMENSIONS = [3, 4, 6]
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s

def _target_triples():
    n, m, p = DIMENSIONS
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    return triples

def _verify(U, V, W):
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    n, m, p = DIMENSIONS
    R = U.shape[1]
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    # compute all (a,b,c) = sum_r U[a,r] V[b,r] W[c,r]
    # Use: for each r, outer product U[:,r] (n*m) x V[:,r] (m*p) gives (a,b) matrix, then contract with W[:,r]
    # We need value[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Build via: M_r = U[:,r][:,None] * V[:,r][None,:]  shape (nm, mp); then value[a,b,c] += M_r[a,b]*W[c,r]
    # Efficient: value = sum_r (U[:,r] * V[:,r])[:, :, None] * W[:, r][None, None, :]
    # That's (nm, mp, pn) array = 12*24*18 = 5184, times R. For R up to 72, 72*5184 ~ 373k, fine.
    val = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        val += (U[:, r][:, None] * V[:, r][None, :])[:, :, None] * W[:, r][None, None, :]
    # check active triples == 1, others == 0
    for (a, b, c) in _target_triples():
        if abs(val[a, b, c] - 1.0) > 1e-9:
            return False
    # check inactive == 0
    mask = np.ones((n*m, m*p, p*n), dtype=bool)
    for (a, b, c) in _target_triples():
        mask[a, b, c] = False
    if np.any(np.abs(val[mask]) > 1e-9):
        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _round_half(M):
    return np.round(2.0 * M) / 2.0

def _search():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    triples = _target_triples()
    # target tensor
    T = np.zeros((nm, mp, pn))
    for (a,b,c) in triples:
        T[a,b,c] = 1.0
    best = None
    best_rank = None
    # Try ranks below schoolbook
    for R in range(20, 72):
        if time.time() > _DEADLINE - 5:
            break
        for trial in range(3):
            if time.time() > _DEADLINE - 5:
                break
            rng = np.random.default_rng(12345 + R*100 + trial)
            U = rng.standard_normal((nm, R))
            V = rng.standard_normal((mp, R))
            W = rng.standard_normal((pn, R))
            # alternating least squares: fix two, solve third
            for it in range(40):
                if time.time() > _DEADLINE - 5:
                    break
                # solve U: for each a, U[a,:] satisfies sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all (b,c)
                # This is: U[a,:] * (V[:,r] W[c,r]) ... reshape: for fixed a, define M[a, (b,c), r] = V[b,r]*W[c,r]; then U[a,:] @ M[a] = T[a,:,:].ravel()
                # Solve least squares per a.
                for a in range(nm):
                    M = (V[:, None, :] * W[None, :, :]).reshape(mp*pn, R)  # (mp*pn, R)
                    rhs = T[a].ravel()
                    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
                    U[a] = sol
                # solve V
                for b in range(mp):
                    M = (U[:, None, :] * W[None, :, :]).reshape(nm*pn, R)
                    rhs = T[:, b, :].reshape(nm*pn)
                    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
                    V[b] = sol
                # solve W
                for c in range(pn):
                    M = (U[:, None, :] * V[None, :, :]).reshape(nm*mp, R)
                    rhs = T[:, :, c].reshape(nm*mp)
                    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
                    W[c] = sol
            # round to half-integers and verify
            Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
            if _verify(Ur, Vr, Wr):
                if best_rank is None or R < best_rank:
                    best_rank = R
                    best = (Ur.tolist(), Vr.tolist(), Wr.tolist())
                    break
            # also try unrounded verify (in case integers already)
            if _verify(U, V, W):
                if best_rank is None or R < best_rank:
                    best_rank = R
                    best = (U.tolist(), V.tolist(), W.tolist())
                    break
    return best, best_rank

def construct():
    t0 = time.time()
    # attempt search
    try:
        best, best_rank = _search()
        if best is not None:
            U, V, W = best
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    # fallback schoolbook
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}
