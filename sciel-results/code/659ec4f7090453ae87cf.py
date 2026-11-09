import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 4]
T0 = time.time()
DEADLINE = T0 + 105  # safety margin under the 120s/150s evaluator limit

def _build_T():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a*m + b
                j = b*p + c
                k = c*n + a
                T[i, j, k] = 1.0
    return T

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _als(T, R, iters=100, seed=0, init=None):
    """ALS tensor decomposition. init: optional (U,V,W) numpy arrays to warm-start."""
    rng = np.random.default_rng(seed)
    n, m, p = T.shape
    if init is not None:
        U, V, W = [a.copy() for a in init]
    else:
        U = rng.normal(size=(n, R))
        V = rng.normal(size=(m, R))
        W = rng.normal(size=(p, R))

    for it in range(iters):
        for r in range(R):
            v_r = V[:, r]; w_r = W[:, r]
            prod = np.outer(v_r, w_r)
            numer = np.tensordot(T, prod, axes=([1, 2], [0, 1]))
            denom = np.sum(prod**2)
            U[:, r] = numer / denom if denom > 1e-14 else 0.0
        for r in range(R):
            u_r = U[:, r]; w_r = W[:, r]
            prod = np.outer(u_r, w_r)
            numer = np.tensordot(T, prod, axes=([0, 2], [0, 1]))
            denom = np.sum(prod**2)
            V[:, r] = numer / denom if denom > 1e-14 else 0.0
        for r in range(R):
            u_r = U[:, r]; v_r = V[:, r]
            prod = np.outer(u_r, v_r)
            numer = np.tensordot(T, prod, axes=([0, 1], [0, 1]))
            denom = np.sum(prod**2)
            W[:, r] = numer / denom if denom > 1e-14 else 0.0

    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    res = np.linalg.norm(T - recon)
    return U, V, W, res

def _als_polish(T, U, V, W, iters=200):
    """High-precision polish: more ALS iterations with tighter zero threshold."""
    n, m, p = T.shape
    R = U.shape[1]
    U, V, W = U.copy(), V.copy(), W.copy()
    for it in range(iters):
        for r in range(R):
            prod = np.outer(V[:, r], W[:, r])
            numer = np.tensordot(T, prod, axes=([1, 2], [0, 1]))
            denom = np.sum(prod**2)
            U[:, r] = numer / denom if denom > 1e-16 else 0.0
        for r in range(R):
            prod = np.outer(U[:, r], W[:, r])
            numer = np.tensordot(T, prod, axes=([0, 2], [0, 1]))
            denom = np.sum(prod**2)
            V[:, r] = numer / denom if denom > 1e-16 else 0.0
        for r in range(R):
            prod = np.outer(U[:, r], V[:, r])
            numer = np.tensordot(T, prod, axes=([0, 1], [0, 1]))
            denom = np.sum(prod**2)
            W[:, r] = numer / denom if denom > 1e-16 else 0.0
        # Early exit if converged
        if it % 50 == 49:
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            res = np.linalg.norm(T - recon)
            if res < 1e-10:
                break
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    res = np.linalg.norm(T - recon)
    return U, V, W, res

def _exact_verify(U, V, W, T_int):
    n, m, p = T_int.shape
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(R):
                    s += Fraction(U[i][r]) * Fraction(V[j][r]) * Fraction(W[k][r])
                if s != T_int[i, j, k]:
                    return False
    return True

def _to_frac_list(arr, denom):
    out = []
    for row in arr:
        r = []
        for x in row:
            r.append(round(x * denom) / denom)
        out.append(r)
    return out

def _try_exactify(U, V, W, T_int):
    """Try rational rounding with several denominators; return (Uf,Vf,Wf) or None."""
    for denom in [1, 2, 4, 8, 16, 32, 64, 128]:
        Uf = _to_frac_list(U, denom)
        Vf = _to_frac_list(V, denom)
        Wf = _to_frac_list(W, denom)
        if _exact_verify(Uf, Vf, Wf, T_int):
            return (Uf, Vf, Wf)
    return None

def _fmm3_init(T, R, seed=0):
    """Structured FMM-3-inspired warm start: seed U,V,W with a near-sparse
    pattern derived from the 3x3 Winograd structure, then zero-pad to R."""
    rng = np.random.default_rng(seed)
    n, m, p = T.shape
    U = rng.normal(size=(n, R)) * 0.1
    V = rng.normal(size=(m, R)) * 0.1
    W = rng.normal(size=(p, R)) * 0.1
    # Inject a schoolbook-like sparse core to help convergence
    core = min(R, n*m*p)
    for c in range(core):
        i = (c // (m*p)) % n
        j = (c // p) % m
        k = c % p
        U[i, c] += 1.0
        V[j, c] += 1.0
        W[k, c] += 1.0
    return U, V, W

def _try_direct_rank23():
    """Attempt a direct rank-23 construction for 3x3x4 using structured patterns."""
    n, m, p = 3, 3, 4
    R = 23
    # Try several structured initializations
    for seed in range(50):
        if time.time() > DEADLINE:
            return None
        rng = np.random.default_rng(seed)
        U = rng.normal(size=(n, R))
        V = rng.normal(size=(m, R))
        W = rng.normal(size=(p, R))
        # Scale to help convergence
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        U *= 2.0
        V *= 2.0
        W *= 2.0
        return U, V, W, seed
    return None

def construct():
    T = _build_T()
    T_int = T.astype(int)
    n, m, p = DIMENSIONS
    fallback = _schoolbook()

    # Search from a low bound up to (but not including) schoolbook rank.
    # Known: 3x3x3 rank <= 23 (FMM-3). For 3x3x4, schoolbook is 36.
    # We attempt ranks 19..35 with multiple seeds, FMM warm-starts, and polish.
    best = None
    
    # First try rank 23 with more aggressive search (it's the most promising)
    for R in [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 19, 20, 21, 22]:
        if time.time() > DEADLINE:
            break
        found = None
        # More seeds for lower ranks, fewer for higher ranks
        max_seeds = 60 if R <= 25 else 30
        for seed in range(0, max_seeds):
            if time.time() > DEADLINE:
                break
            # Use FMM init for even seeds, random for odd
            if seed % 2 == 0:
                init = _fmm3_init(T, R, seed=seed)
            else:
                init = None
            U, V, W, res = _als(T, R, iters=100, seed=seed, init=init)
            if res < 1e-5:
                # Polish at high precision
                U, V, W, res = _als_polish(T, U, V, W, iters=400)
                if res < 1e-7:
                    found = _try_exactify(U, V, W, T_int)
                    if found:
                        break
        if found:
            best = found
            break

    if best is not None:
        Uf, Vf, Wf = best
        return {"U": Uf, "V": Vf, "W": Wf}

    return fallback
