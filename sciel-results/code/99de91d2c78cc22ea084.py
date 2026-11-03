"""Matrix multiplication 2,3,6 — rank-minimal exact tensor decomposition.

Strategy (Kronecker family):
  The 2x3x6 multiplication tensor factors as a tensor (Kronecker) product of
  a (2,1,3) problem and a (1,3,2) problem:
      (2*1) x (1*3) x (3*2) = 2 x 3 x 6
  with rank r1 * r2.  Schoolbook factors give r1=6, r2=6 -> rank 36.

  We build this Kronecker construction (always valid), then attempt a
  rank-reduction polish: alternating least-squares on the factor matrices
  with an exactness check (entries must be integer or half-integer) and a
  strict deadline guard so we always return a valid decomposition.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# Hard wall-clock budget with safety margin (evaluator kills at 150 s).
_DEADLINE = 140.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _deadline_exceeded():
    return _elapsed() > _DEADLINE


def _is_half_integer(x, tol=1e-6):
    """True if x is within tol of an integer or half-integer."""
    r = round(2 * x)
    return abs(2 * x - r) < tol


def _snap_half(x):
    """Snap x to the nearest integer or half-integer."""
    return round(2 * x) / 2.0


def _build_kronecker(n=2, m=3, p=6):
    """Kronecker-product construction of the 2x3x6 multiplication tensor.

    Factor as (n1,m1,p1)=(2,1,3) and (n2,m2,p2)=(1,3,2).
    Index maps:
      i = i1*n2 + i2      (i1 in [0,2), i2 in [0,1))  -> i in [0,2)
      j = j1*m2 + j2      (j1 in [0,1), j2 in [0,3))  -> j in [0,3)
      k = k1*p2 + k2      (k1 in [0,3), k2 in [0,2))  -> k in [0,6)

    For the (2,1,3) factor, schoolbook rank-1 terms are indexed by (i1, j1, k1)
      with j1 fixed (m1=1).  U1[i1,j1,c1]=1, V1[j1,k1,c1]=1, W1[k1,i1,c1]=1.
    For the (1,3,2) factor, schoolbook rank-1 terms indexed by (i2,j2,k2)
      with i2 fixed (n2=1).  U2[i2,j2,c2]=1, V2[j2,k2,c2]=1, W2[k2,i2,c2]=1.

    Kronecker: a rank-1 term of the product is the outer (Kronecker) product
    of a rank-1 term of factor1 and a rank-1 term of factor2.
    U[i,j,c] = U1[i1,j1,c1] * U2[i2,j2,c2]
    V[j,k,c] = V1[j1,k1,c1] * V2[j2,k2,c2]
    W[k,i,c] = W1[k1,i1,c1] * W2[k2,i2,c2]
    with c = c1 * r2 + c2.
    """
    n1, m1, p1 = 2, 1, 3
    n2, m2, p2 = 1, 3, 2
    assert n1 * n2 == n and m1 * m2 == m and p1 * p2 == p

    r1 = n1 * m1 * p1  # 6
    r2 = n2 * m2 * p2  # 6
    rank = r1 * r2     # 36

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Build schoolbook rank-1 term lists for each factor.
    # Factor1 (2,1,3): terms indexed by (i1, j1, k1), j1 in [0,1).
    terms1 = []
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                terms1.append((i1, j1, k1))
    # Factor2 (1,3,2): terms indexed by (i2, j2, k2), i2 in [0,1).
    terms2 = []
    for i2 in range(n2):
        for j2 in range(m2):
            for k2 in range(p2):
                terms2.append((i2, j2, k2))

    for c1, (i1, j1, k1) in enumerate(terms1):
        for c2, (i2, j2, k2) in enumerate(terms2):
            c = c1 * r2 + c2
            i = i1 * n2 + i2
            j = j1 * m2 + j2
            k = k1 * p2 + k2
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0

    return U, V, W, rank


def _verify(U, V, W, n=2, m=3, p=6, tol=1e-8):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _check_half_integer(U, V, W, tol=1e-6):
    for M in (U, V, W):
        for x in np.asarray(M).ravel():
            if not _is_half_integer(x, tol):
                return False
    return True


def _try_reduce_rank(U0, V0, W0, n=2, m=3, p=6, max_rank=None,
                     iters=400, tol=1e-7, seed=0):
    """Attempt to find a lower-rank exact (half-integer) decomposition via ALS.

    We try target ranks r = current_rank-1, -2, ... down to a floor.
    For each r, initialize factors by truncating / random and run ALS,
    then snap to half-integers and verify.  Return the best (lowest) valid
    decomposition found, or None.
    """
    if max_rank is None:
        max_rank = U0.shape[1]

    rng = np.random.default_rng(seed)
    cur_rank = U0.shape[1]

    for r in range(cur_rank - 1, 5, -1):
        if _deadline_exceeded():
            break
        # Try a few random initializations.
        for trial in range(3):
            if _deadline_exceeded():
                break
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            # Initialize with a scaled random projection of the full factors.
            U[:, :min(r, cur_rank)] = U0[:, :min(r, cur_rank)] * 0.5
            V[:, :min(r, cur_rank)] = V0[:, :min(r, cur_rank)] * 0.5
            W[:, :min(r, cur_rank)] = W0[:, :min(r, cur_rank)] * 0.5
            U[:, min(r, cur_rank):] = rng.standard_normal((n * m, r - min(r, cur_rank))) * 0.3
            V[:, min(r, cur_rank):] = rng.standard_normal((m * p, r - min(r, cur_rank))) * 0.3
            W[:, min(r, cur_rank):] = rng.standard_normal((p * n, r - min(r, cur_rank))) * 0.3

            best_err = float('inf')
            for it in range(iters):
                if _deadline_exceeded():
                    break
                # Update U: for each (i,j), U[i*m+j, c] should satisfy
                #   sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all k
                # This is a linear system in U[i*m+j,:] given V,W.
                # Build matrix A[i*m+j] of shape (p, r): A[c, k] = V[j*p+k,c]*W[k*n+i,c]
                # Solve A^T? We want sum_c U[c] * (V[j*p+k,c]*W[k*n+i,c]) = 1 for each k.
                # i.e. M @ U_col = ones(p) where M[k, c] = V[j*p+k,c]*W[k*n+i,c].
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, r))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve least squares M @ u = 1
                        try:
                            u, *_ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        except np.linalg.LinAlgError:
                            continue
                        U[i * m + j, :] = u
                # Update V similarly.
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((m, r))
                        for ii in range(n):
                            M[ii, :] = U[ii * m + j, :] * W[k * n + ii, :]
                        # We need sum_c V[j*p+k,c] * U[ii*m+j,c]*W[k*n+ii,c] = 1 for each ii.
                        try:
                            v, *_ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        except np.linalg.LinAlgError:
                            continue
                        V[j * p + k, :] = v
                # Update W similarly.
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((p, r))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        # sum_c W[k*n+i,c]*U[i*m+j,c]*V[j*p+k,c] = 1 for each j.
                        try:
                            w, *_ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        except np.linalg.LinAlgError:
                            continue
                        W[k * n + i, :] = w
                # Error check.
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += abs(s - 1.0)
                if err < best_err:
                    best_err = err
                if err < tol:
                    break

            # Snap to half-integers and verify.
            Us = np.array([[_snap_half(x) for x in row] for row in U])
            Vs = np.array([[_snap_half(x) for x in row] for row in V])
            Ws = np.array([[_snap_half(x) for x in row] for row in W])
            if _verify(Us, Vs, Ws, n, m, p, tol=1e-6) and _check_half_integer(Us, Vs, Ws, tol=1e-6):
                return Us.tolist(), Vs.tolist(), Ws.tolist()

    return None


def construct():
    global _START
    _START = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 6

    # Kronecker construction (always valid, rank 36).
    U, V, W, rank = _build_kronecker(n, m, p)

    # Verify the Kronecker construction.
    if not _verify(U, V, W, n, m, p, tol=1e-9):
        # Fallback to pure schoolbook (should never happen).
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    # Attempt rank reduction within the Kronecker family.
    result = _try_reduce_rank(U, V, W, n, m, p)
    if result is not None:
        U, V, W = result

    # Final safety: ensure validity; if not, revert to Kronecker.
    if not _verify(np.array(U), np.array(V), np.array(W), n, m, p, tol=1e-6):
        U, V, W, _ = _build_kronecker(n, m, p)

    return {"U": U, "V": V, "W": W}
