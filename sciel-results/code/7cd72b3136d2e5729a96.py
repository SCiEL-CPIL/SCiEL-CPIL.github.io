"""Rank-minimal tensor decomposition for 2x3 @ 3x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == k:
                    T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Check that U @ V @ W reconstructs T exactly (within tol)."""
    # T[i,k,r] = sum_c U[i,c] V[k,c] W[r,c]
    # Reshape: U (nm, r), V (mp, r), W (pn, r)
    # For each c: outer of U[:,c], V[:,c], W[:,c]
    nm, r = U.shape
    mp = V.shape[0]
    pn = W.shape[0]
    rec = np.zeros((nm, mp, pn), dtype=np.float64)
    for c in range(r):
        rec += np.einsum('i,k,r->ikr', U[:, c], V[:, c], W[:, c])
    return np.max(np.abs(rec - T)) < tol

def _refine(U, V, W, T, iters=300, tol=1e-12):
    """Alternating least squares to refine U,V,W to fit T exactly."""
    nm, r = U.shape
    mp = V.shape[0]
    pn = W.shape[0]
    Tf = T.astype(np.float64)
    prev_err = np.inf
    for it in range(iters):
        # Update U: for each i, U[i,:] solves sum_c U[i,c] * (V[:,c] ⊗ W[:,c]) = T[i,:,:]
        # Flatten V[:,c] and W[:,c] outer: shape (mp*pn,)
        VW = np.einsum('kc,rc->krc', V, W).reshape(mp * pn, r)
        # T[i,:,:] flattened: shape (mp*pn,)
        Tflat = Tf.reshape(nm, mp * pn)
        # Solve: VW^T @ U[i,:] = Tflat[i,:]
        U = np.linalg.lstsq(VW.T, Tflat.T, rcond=None)[0].T
        # Update V
        UW = np.einsum('ic,rc->ir c', U, W)  # (nm, pn, r)
        UW = np.einsum('ic,rc->irc', U, W).reshape(nm * pn, r)
        Tflat = Tf.reshape(nm, mp, pn).transpose(0, 2, 1).reshape(mp, nm * pn)
        V = np.linalg.lstsq(UW.T, Tflat.T, rcond=None)[0].T
        # Update W
        UV = np.einsum('ic,kc->ikc', U, V).reshape(nm * mp, r)
        Tflat = Tf.transpose(2, 0, 1).reshape(pn, nm * mp)
        W = np.linalg.lstsq(UV.T, Tflat.T, rcond=None)[0].T
        # Error
        rec = np.einsum('ic,kc,rc->ikr', U, V, W)
        err = np.max(np.abs(rec - Tf))
        if err < prev_err * (1 - 1e-6) and err < tol:
            break
        prev_err = err
    return U, V, W, prev_err

def _to_half_integer(M, tol=1e-6):
    """Round entries to nearest half-integer; return None if not close."""
    M2 = 2.0 * M
    M2r = np.round(M2)
    if np.max(np.abs(M2 - M2r)) > tol:
        return None
    return M2r / 2.0

def _try_rank(T, r, rng, max_trials=8, time_budget=10.0):
    nm, mp, pn = T.shape
    start = time.time()
    for trial in range(max_trials):
        if time.time() - start > time_budget:
            break
        U = rng.standard_normal((nm, r))
        V = rng.standard_normal((mp, r))
        W = rng.standard_normal((pn, r))
        U, V, W, err = _refine(U, V, W, T, iters=400)
        if err < 1e-9:
            # Check half-integer property
            Uh = _to_half_integer(U)
            Vh = _to_half_integer(V)
            Wh = _to_half_integer(W)
            if Uh is not None and Vh is not None and Wh is not None:
                # Final exact verification with half-integer values
                if _verify(Uh, Vh, Wh, T, tol=1e-9):
                    return Uh, Vh, Wh
    return None

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s
    n, m, p = DIMENSIONS
    T = _build_tensor()
    nm, mp, pn = T.shape
    # Lower bound: at least max(nm, mp, pn) is not a valid LB for tensor rank;
    # use known results: for 2x3x9, try from 46 upward
    # Actually rank >= max of any matricization rank.
    # Matricizations: (nm x mp*pn), (mp x pn*nm), (pn x nm*mp)
    T1 = T.reshape(nm, mp * pn)
    T2 = T.reshape(mp, pn * nm)
    T3 = T.reshape(pn, nm * mp)
    r_min = max(np.linalg.matrix_rank(T1), np.linalg.matrix_rank(T2), np.linalg.matrix_rank(T3))
    # Start from a reasonable lower bound
    r_start = max(r_min, 46)
    best = None
    rng = np.random.default_rng(42)
    for r in range(r_start, 55):
        if time.time() > deadline - 20.0:
            break
        res = _try_rank(T, r, rng, max_trials=6, time_budget=min(12.0, deadline - time.time() - 15.0))
        if res is not None:
            U, V, W = res
            best = (r, U, V, W)
            break
    if best is None:
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    r, U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
