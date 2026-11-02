"""Matrix multiplication 2x4x6 rank-minimal exact tensor decomposition.

Structural change from the seed: instead of returning the schoolbook rank-48
decomposition directly, we search for a lower-rank exact decomposition using
alternating least squares in floating point, then verify the candidate EXACTLY
with integer arithmetic (common-denominator scaling) so the returned factors
satisfy the tensor identity to machine-exact precision. We iterate candidate
ranks from a linear-algebra lower bound up to the schoolbook rank and return
the smallest rank that passes exact verification. A wall-clock deadline guard
ensures we always return a valid (schoolbook) result before the limit.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Return the multiplication tensor mu as a 3-array.

    mu[(i*k), (i'*j), (j'*k')] = 1 iff i==i' and k==k' and j==j'.
    Shapes: (n*p), (n*m), (m*p).
    """
    np_ = n * p
    nm = n * m
    mp = m * p
    mu = np.zeros((np_, nm, mp), dtype=float)
    for i in range(n):
        for i2 in range(n):
            for k in range(p):
                for k2 in range(p):
                    if i != i2 or k != k2:
                        continue
                    for j in range(m):
                        for j2 in range(m):
                            if j != j2:
                                continue
                            mu[i * p + k, i2 * m + j, j2 * p + k2] = 1.0
    return mu


def _lower_bound(n, m, p):
    """Lower bound on rank = max of the three mode-flattening ranks."""
    mu = _build_target_tensor(n, m, p)
    np_, nm, mp = mu.shape
    # Flatten mode 1 vs (2,3)
    M1 = mu.reshape(np_, nm * mp)
    r1 = np.linalg.matrix_rank(M1)
    # Flatten mode 2 vs (1,3)
    M2 = mu.transpose(1, 0, 2).reshape(nm, np_ * mp)
    r2 = np.linalg.matrix_rank(M2)
    # Flatten mode 3 vs (1,2)
    M3 = mu.transpose(2, 0, 1).reshape(mp, np_ * nm)
    r3 = np.linalg.matrix_rank(M3)
    return int(max(r1, r2, r3))


def _als_decompose(mu, r, n_iter=200, n_restarts=8, seed=0):
    """Alternating least squares to find rank-r decomposition mu ~= A@B@C.

    Returns (A, B, C) with shapes (np_, r), (nm, r), (mp, r) or None on failure.
    """
    np_, nm, mp = mu.shape
    rng = np.random.default_rng(seed)
    best = None
    best_err = np.inf
    for restart in range(n_restarts):
        A = rng.standard_normal((np_, r))
        B = rng.standard_normal((nm, r))
        C = rng.standard_normal((mp, r))
        # normalize columns to avoid scale drift
        for _ in range(n_iter):
            # update A: for each i, A[i,:] such that (A[i,:] @ (B*C)) approx mu[i,:,:].vec
            # mu[i] is (nm, mp) = sum_c A[i,c] * B[:,c] outer C[:,c]
            # => mu[i].vec = (C ⊗ B) @ A[i,:].T  ... solve least squares
            # Build M = C ⊗ B of shape (nm*mp, r)
            # mu[i].reshape(-1) = M @ A[i,:]
            M = np.kron(C, B)  # (mp*nm, r) but ordering: kron(C,B)[ (k*nm + j), c ] = C[k,c]*B[j,c]
            # We need mu[i, j, k] ordering. mu[i] shape (nm, mp). vec with j-major? 
            # M[c-index] row = (k*nm + j) corresponds to C[k,c]*B[j,c].
            # mu[i, j, k] should equal sum_c A[i,c]*B[j,c]*C[k,c].
            # So mu[i].T (shape mp, nm) vec (k-major, j) : mu[i].T.reshape(-1)[k*nm+j] = mu[i,j,k].
            rhs = mu[i].T.reshape(-1)
            sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
            A[i, :] = sol
            # update B: mu[:, j, :] -> for each j, mu[:,j,:] (np_, mp) = sum_c A[:,c] outer C[:,c] * B[j,c]
            M2 = np.kron(C, A)  # row (k*np_+i) = C[k,c]*A[i,c]; shape (mp*np_, r)
            for j in range(nm):
                rhs = mu[:, j, :].T.reshape(-1)  # (k*np_+i)
                sol, *_ = np.linalg.lstsq(M2, rhs, rcond=None)
                B[j, :] = sol
            # update C: mu[:, :, k] -> for each k, mu[:,:,k] (np_, nm) = sum_c A[:,c] outer B[:,c] * C[k,c]
            M3 = np.kron(B, A)  # row (j*np_+i) = B[j,c]*A[i,c]; shape (nm*np_, r)
            for k in range(mp):
                rhs = mu[:, :, k].T.reshape(-1)  # (j*np_+i)
                sol, *_ = np.linalg.lstsq(M3, rhs, rcond=None)
                C[k, :] = sol
        # compute error
        recon = np.einsum('ic,jc,kc->ijk', A, B, C)
        err = np.max(np.abs(recon - mu))
        if err < best_err:
            best_err = err
            best = (A.copy(), B.copy(), C.copy())
    return best, best_err


def _to_exact(A, B, C, tol=1e-7):
    """Try to convert float factors to exact integer/half-integer entries.

    Find a common denominator D (power of 2 up to some limit, else integer gcd)
    such that all entries * D are integers within tol. Return (Ae, Be, Ce) as
    lists of lists of floats (integers or halves) or None.
    """
    allv = np.concatenate([A.ravel(), B.ravel(), C.ravel()])
    # Try to find scaling D so that D*allv are all near-integer.
    # First try: are entries already near integers?
    for D in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        scaled = allv * D
        rounded = np.round(scaled)
        if np.max(np.abs(scaled - rounded)) < tol:
            # verify reconstruction exactly with these rounded ints
            Ae = np.round(A * D).astype(int)
            Be = np.round(B * D).astype(int)
            Ce = np.round(C * D).astype(int)
            # check reconstruction: sum_c Ae[i,c] Be[j,c] Ce[k,c] / D^3 should equal mu
            # We'll verify in the caller with proper scaling. Return scaled ints and D.
            return Ae, Be, Ce, D
    return None


def _verify_exact(Ae, Be, Ce, D, mu, tol=1e-9):
    """Check that (1/D^3) * sum_c Ae[i,c] Be[j,c] Ce[k,c] == mu[i,j,k] exactly."""
    recon = np.einsum('ic,jc,kc->ijk', Ae.astype(float), Be.astype(float), Ce.astype(float)) / (D ** 3)
    return np.max(np.abs(recon - mu)) < tol


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def construct():
    start = time.time()
    DEADLINE = 115.0  # seconds, safety margin before 120s kill
    n, m, p = DIMENSIONS
    mu = _build_target_tensor(n, m, p)
    np_, nm, mp = mu.shape

    school = _schoolbook(n, m, p)

    try:
        lb = _lower_bound(n, m, p)
    except Exception:
        lb = 1

    upper = n * m * p  # schoolbook rank
    # Search ranks from lb up to upper-1 (we already have schoolbook for upper)
    for r in range(lb, upper):
        if time.time() - start > DEADLINE:
            break
        try:
            res, err = _als_decompose(mu, r, n_iter=120, n_restarts=6, seed=r)
        except Exception:
            continue
        if res is None:
            continue
        A, B, C = res
        if err > 1e-6:
            continue
        # try to make exact
        exact = _to_exact(A, B, C, tol=1e-6)
        if exact is None:
            continue
        Ae, Be, Ce, D = exact
        if _verify_exact(Ae, Be, Ce, D, mu, tol=1e-9):
            # Build output with proper scaling: U[i,c] = Ae[i,c]/D, etc.
            # Entries must be integer or half-integer. Ae/D may not be half-integer.
            # We need final factors such that product reconstructs mu.
            # U = Ae / D, V = Be / D, W = Ce / D gives (1/D^3) sum = mu. Good.
            # But entries must be integer or half-integer (per contract).
            # If D is a power of 2, Ae/D is a dyadic rational; may not be half-integer.
            # The contract says "integer/half-integer entries". To be safe, we can
            # instead absorb: let U = Ae, V = Be, W = Ce / D^3? No, W must be integer/half.
            # Alternative: scale so that each factor's entries are integer or half.
            # Use U = Ae, V = Be, W = Ce, and the product gives D^3 * mu. That's wrong.
            # We need the *product* to equal mu. So we must divide by D^3 somewhere.
            # Put all scaling in W: W = Ce / D^3. Ce/D^3 entries: if D=2, /8 -> not half-integer.
            # Instead, find D such that we can split. Simplest: require D=1 (integer factors)
            # or accept that we return float entries equal to integer/half-integer.
            # Let's just return the float values; if they are integer or half-integer, good.
            Uf = (Ae / D).tolist()
            Vf = (Be / D).tolist()
            Wf = (Ce / D).tolist()
            # Check all entries are integer or half-integer
            ok = True
            for mat in (Uf, Vf, Wf):
                for row in mat:
                    for x in row:
                        if abs(x * 2 - round(x * 2)) > 1e-9:
                            ok = False
            if ok:
                # snap to exact half-integers
                def snap(mat):
                    return [[ (2 * round(2 * x)) / 2.0 for x in row] for row in mat]
                return {"U": snap(Uf), "V": snap(Vf), "W": snap(Wf)}
    return school
# EVOLVE-BLOCK-END
