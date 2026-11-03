"""Rank-minimal exact tensor decomposition for 2x3x4 matrix multiplication.
Uses a rank-revealing SVD unfolding + integer lattice refinement search,
with a hard deadline guard. Falls back to the schoolbook rank-24 result."""
import time
import numpy as np
import scipy.optimize

DIMENSIONS = [2, 3, 4]
_T0 = time.time()
_DEADLINE = _T0 + 110.0  # safety margin before 150s kill


def _mult_tensor(n, m, p):
    """Matrix-multiplication tensor T[i, j, k] = 1 if valid product entry."""
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _build_target_tensor(n, m, p):
    """Build the full 3-way tensor with indices (i*m+j, j*p+k, k*n+i)."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _snap(x, tol=1e-6):
    """Snap near-integer / half-integer floats to exact rationals (as floats)."""
    r = np.rint(2.0 * x) / 2.0
    return r


def _verify_exact(U, V, W, n, m, p, T_target, tol=1e-9):
    """Verify U, V, W form a valid exact decomposition with integer/half-integer entries."""
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False

    # Check entries are close to integers or half-integers
    if not (np.allclose(U * 2, np.round(U * 2), atol=1e-6) and
            np.allclose(V * 2, np.round(V * 2), atol=1e-6) and
            np.allclose(W * 2, np.round(W * 2), atol=1e-6)):
        return False

    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2

    T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True


def _make_factors(n, m, p, U, V, W):
    """Return dict with factor matrices as integer/half-integer lists."""
    U = _snap(U).astype(int) if np.allclose(_snap(U), np.rint(_snap(U))) else _snap(U)
    V = _snap(V).astype(int) if np.allclose(_snap(V), np.rint(_snap(V))) else _snap(V)
    W = _snap(W).astype(int) if np.allclose(_snap(W), np.rint(_snap(W))) else _snap(W)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return _make_factors(n, m, p, U, V, W)


def _try_rank_lm(rank, n, m, p, T_target, max_seeds=30, seed_base=1000):
    """Try to find a rank-r decomposition via Levenberg-Marquardt optimization."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n

    def residual(x):
        U = x[:dim_U * rank].reshape(dim_U, rank)
        V = x[dim_U * rank: dim_U * rank + dim_V * rank].reshape(dim_V, rank)
        W = x[dim_U * rank + dim_V * rank:].reshape(dim_W, rank)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_recon - T_target).flatten()

    for seed in range(max_seeds):
        if time.time() > _DEADLINE:
            return None

        rng = np.random.RandomState(seed_base + seed)

        # Mix random and structured initializations
        if seed % 3 == 0:
            # Pure random
            U_init = rng.randn(dim_U, rank) * 0.5
            V_init = rng.randn(dim_V, rank) * 0.5
            W_init = rng.randn(dim_W, rank) * 0.5
        elif seed % 3 == 1:
            # Structured: schoolbook-like pattern perturbed
            U_init = np.zeros((dim_U, rank))
            V_init = np.zeros((dim_V, rank))
            W_init = np.zeros((dim_W, rank))
            count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if count >= rank:
                            break
                        U_init[i * m + j, count] = 1.0
                        V_init[j * p + k, count] = 1.0
                        W_init[k * n + i, count] = 1.0
                        count += 1
                    if count >= rank:
                        break
                if count >= rank:
                    break
            U_init += rng.randn(dim_U, rank) * 0.1
            V_init += rng.randn(dim_V, rank) * 0.1
            W_init += rng.randn(dim_W, rank) * 0.1
        else:
            # Integer lattice random
            U_init = rng.randint(-1, 2, size=(dim_U, rank)).astype(float)
            V_init = rng.randint(-1, 2, size=(dim_V, rank)).astype(float)
            W_init = rng.randint(-1, 2, size=(dim_W, rank)).astype(float)

        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

        try:
            res = scipy.optimize.least_squares(
                residual, x0, method='lm',
                ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=1500
            )
        except Exception:
            continue

        if res.cost < 1e-10:
            x_sol = res.x
            U = x_sol[:dim_U * rank].reshape(dim_U, rank)
            V = x_sol[dim_U * rank: dim_U * rank + dim_V * rank].reshape(dim_V, rank)
            W = x_sol[dim_U * rank + dim_V * rank:].reshape(dim_W, rank)

            if _verify_exact(U, V, W, n, m, p, T_target):
                return _make_factors(n, m, p, U, V, W)

    return None


def _svd_guided_init(n, m, p, T_target, rank):
    """Use SVD of a tensor unfolding to guide the initial guess for optimization."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n

    # Unfold along one axis: (n*m) x (m*p) matrix from fixing W index
    # Actually, let's use the (n*m) x p unfolding of the n,m,p tensor
    T_nmp = T_target.reshape(dim_U, dim_V, dim_W)
    # Contract over W to get a (dim_U) x (dim_V) matrix
    # M[a,b] = sum_c T[a,b,c]
    M = T_target.sum(axis=2)  # shape (dim_U, dim_V)

    try:
        Umat, s, Vt = np.linalg.svd(M, full_matrices=False)
        r = min(rank, len(s))
        U = Umat[:, :r] * np.sqrt(s[:r])
        V = (Vt[:r, :] * np.sqrt(s[:r])).T
        # W is initialized randomly since the unfolding doesn't capture it
        rng = np.random.RandomState(42)
        W = rng.randn(dim_W, rank) * 0.5
        return U, V, W
    except Exception:
        return None


def construct():
    n, m, p = DIMENSIONS
    T_target = _build_target_tensor(n, m, p)
    best = _schoolbook(n, m, p)  # guaranteed valid rank-24 baseline

    # Try to find lower-rank decompositions using LM optimization
    # with SVD-guided and structured initializations
    for rank in range(23, 12, -1):  # try ranks 23 down to 13
        if time.time() > _DEADLINE:
            break

        # First try with SVD-guided initialization
        init = _svd_guided_init(n, m, p, T_target, rank)
        if init is not None:
            U_init, V_init, W_init = init
            dim_U, dim_V, dim_W = n * m, m * p, p * n

            def residual(x):
                U = x[:dim_U * rank].reshape(dim_U, rank)
                V = x[dim_U * rank: dim_U * rank + dim_V * rank].reshape(dim_V, rank)
                W = x[dim_U * rank + dim_V * rank:].reshape(dim_W, rank)
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                return (T_recon - T_target).flatten()

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            try:
                res = scipy.optimize.least_squares(
                    residual, x0, method='lm',
                    ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000
                )
                if res.cost < 1e-10:
                    x_sol = res.x
                    U = x_sol[:dim_U * rank].reshape(dim_U, rank)
                    V = x_sol[dim_U * rank: dim_U * rank + dim_V * rank].reshape(dim_V, rank)
                    W = x_sol[dim_U * rank + dim_V * rank:].reshape(dim_W, rank)
                    if _verify_exact(U, V, W, n, m, p, T_target):
                        return _make_factors(n, m, p, U, V, W)
            except Exception:
                pass

        # Then try generic LM with multiple seeds
        result = _try_rank_lm(rank, n, m, p, T_target, max_seeds=20, seed_base=rank * 100)
        if result is not None:
            return result

    if time.time() > _DEADLINE:
        return best
    return best


# EVOLVE-BLOCK-END
