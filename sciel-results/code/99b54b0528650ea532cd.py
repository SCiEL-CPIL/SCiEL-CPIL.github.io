"""Search for a rank-minimal exact tensor decomposition of the 2x3x4
matrix-multiplication tensor.  The schoolbook rank is 24.  We attempt a
structured search for lower-rank decompositions using random initialisation
and ALS-style refinement, with a hard deadline guard.  Falls back to the
guaranteed schoolbook rank-24 result."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]
_T0 = time.time()
_DEADLINE = _T0 + 110.0  # safety margin before the 150 s kill


def _mult_tensor(n, m, p):
    """The all-ones tensor T[i,j,k] = 1 for the 2x3x4 multiplication."""
    return np.ones((n, m, p), dtype=float)


def _snap(x):
    """Snap to nearest half-integer."""
    return np.rint(2.0 * x) / 2.0


def _make_factors(U, V, W):
    """Return dict with factor matrices as lists, snapped to half-integers."""
    def _clean(M):
        s = _snap(M)
        if np.allclose(s, np.rint(s), atol=1e-9):
            return np.rint(s).astype(int)
        return s
    return {"U": _clean(U).tolist(),
            "V": _clean(V).tolist(),
            "W": _clean(W).tolist()}


def _verify(n, m, p, U, V, W, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    T = _mult_tensor(n, m, p)
    rec = np.einsum('ic,rc,jc->ijr', U, V, W)
    return bool(np.allclose(rec, T, atol=tol, rtol=tol))


def _schoolbook(n, m, p):
    """Guaranteed valid rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return _make_factors(U, V, W)


def _als_step(n, m, p, U, V, W, rank, max_iter=200, tol=1e-10):
    """Alternating least squares refinement.  Returns (U,V,W, converged)."""
    T = _mult_tensor(n, m, p)
    for it in range(max_iter):
        # Update U: for each (i,j), solve  V^T W_{(i,j)} V  U_{(i,j)} = T[i,j,:]
        # Actually: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Fix V,W: for fixed (i,j), T[i,j,:] = U[i*m+j,:] * (V * W_{i,j})
        # where (V*W_{i,j})[k,c] = V[j*p+k,c]*W[k*n+i,c]
        # So U[i*m+j,:] solves:  sum_c U[c] * (V[j*p+k,c]*W[k*n+i,c]) = T[i,j,k]
        # => A U_row = b where A[k,c] = V[j*p+k,c]*W[k*n+i,c], b[k]=T[i,j,k]
        new_U = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build p x rank matrix A
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = T[i, j, :]  # all ones
                # Solve A^T A u = A^T b
                try:
                    new_U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    new_U[i * m + j, :] = U[i * m + j, :]

        # Update V: for fixed (j,k), T[:,j,k] = (U * W_{j,k}) V[j*p+k,:]
        # A[i,c] = U[i*m+j,c]*W[k*n+i,c], b[i]=T[i,j,k]
        new_V = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = T[:, j, k]  # all ones
                try:
                    new_V[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    new_V[j * p + k, :] = V[j * p + k, :]

        # Update W: for fixed (k,i), T[i,:,k] = (U * V_{k,i}) W[k*n+i,:]
        # A[j,c] = U[i*m+j,c]*V[j*p+k,c], b[j]=T[i,j,k]
        new_W = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = T[i, :, k]  # all ones
                try:
                    new_W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    new_W[k * n + i, :] = W[k * n + i, :]

        # Check convergence
        U, V, W = new_U, new_V, new_W
        if it % 10 == 0 or it == max_iter - 1:
            rec = np.einsum('ic,rc,jc->ijr', U, V, W)
            err = np.max(np.abs(rec - T))
            if err < tol:
                return U, V, W, True
    return U, V, W, False


def _try_rank(n, m, p, rank, n_trials=50, seed=0):
    """Try to find a rank-r decomposition via random init + ALS."""
    T = _mult_tensor(n, m, p)
    rng = np.random.RandomState(seed)
    for trial in range(n_trials):
        if time.time() > _DEADLINE:
            return None
        # Random initialisation
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        # Normalise columns to avoid scale issues
        for c in range(rank):
            nU = np.linalg.norm(U[:, c])
            nV = np.linalg.norm(V[:, c])
            nW = np.linalg.norm(W[:, c])
            if nU > 1e-12 and nV > 1e-12 and nW > 1e-12:
                scale = (nU * nV * nW) ** (1.0 / 3.0)
                U[:, c] /= nU / scale
                V[:, c] /= nV / scale
                W[:, c] /= nW / scale

        U, V, W, conv = _als_step(n, m, p, U, V, W, rank)
        if conv:
            # Verify
            rec = np.einsum('ic,rc,jc->ijr', U, V, W)
            if np.allclose(rec, T, atol=1e-6, rtol=1e-6):
                return _make_factors(U, V, W)
    return None


def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)  # guaranteed valid rank-24 baseline
    best_rank = n * m * p  # 24

    # Try ranks from 13 up to 23 (known lower bound for 2x3x4 is at least 13
    # by various bounds; we search a reasonable range).
    for rank in range(13, best_rank):
        if time.time() > _DEADLINE:
            break
        result = _try_rank(n, m, p, rank, n_trials=30, seed=rank)
        if result is not None:
            best = result
            best_rank = rank
            # Found a valid decomposition; continue searching for lower rank?
            # No, we go ascending, so this is the lowest found so far.
            # Actually we should break and try even lower ranks.
            # But we're going from 13 up, so if we find rank r, we should
            # check if r-1, ..., 13 also work.  Let's just break and try lower.
            break

    # If we found something at some rank, try to go lower
    if best_rank < n * m * p:
        for rank in range(13, best_rank):
            if time.time() > _DEADLINE:
                break
            result = _try_rank(n, m, p, rank, n_trials=30, seed=1000 + rank)
            if result is not None:
                best = result
                best_rank = rank

    if time.time() > _DEADLINE:
        return best
    return best


# EVOLVE-BLOCK-END
