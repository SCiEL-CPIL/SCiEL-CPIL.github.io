"""Structural redesign: reduce the 400-dim search to a smooth low-dimensional
basis (windowed Zadoff-Chu / Chebyshev) and optimize with BFGS on that reduced
space. The objective is scale-invariant, so only the direction of h matters."""
import time
import numpy as np

N = 400
DEADLINE = 115.0  # wall-clock safety margin below the 120s evaluator limit

# ---------------------------------------------------------------------------
# Precompute the autoconvolution kernel so that
#   conv(h,h)[k] = sum_i h[i] * h[i + (k - (N-1))]
#                = sum_i h[i] * K[i,k] * h[i + (k-(N-1))]
# i.e. a quadratic form h^T A_k h for each lag k.
# ---------------------------------------------------------------------------
def _build_kernel():
    K = np.arange(N)
    A = np.empty((2 * N - 1, N, N), dtype=np.float64)
    for k in range(2 * N - 1):
        d = k - (N - 1)
        if d >= 0:
            i0 = 0
            i1 = N - d
        else:
            i0 = -d
            i1 = N
        idx = np.arange(i0, i1)
        A[k, idx, idx + d] = 1.0
    return A

_KERNEL = _build_kernel()
_M = 2 * N - 1


def _objective_and_grad(theta):
    """Objective = 2*N*max(conv)/sum(h)^2 and its gradient w.r.t. theta.

    theta is the reduced parameter vector; h = basis @ theta.
    """
    h = _BASIS @ theta
    S = float(np.sum(h))
    if S == 0.0:
        return 1e18, None

    # conv_k = sum_i A[k,i,j] h_i h_j  = h^T A_k h
    # For a symmetric-ish quadratic form we can compute all conv via einsum.
    conv = np.einsum('kij,i,j->k', _KERNEL, h, h)
    S2 = S * S
    obj = 2.0 * N * float(np.max(conv)) / S2

    # gradient w.r.t. h: d/dh [ 2N max(conv)/S^2 ]
    kmax = int(np.argmax(conv))
    Ak = _KERNEL[kmax]
    dconv = 2.0 * (Ak @ h)          # d(conv_kmax)/dh
    dS = np.full(N, 1.0)            # dS/dh
    grad_h = 2.0 * N * (dconv * S2 - obj * S * dS) / (S2 * S2)
    # obj/S^2 = 2N max(conv)/S^4 ... recompute cleanly:
    # f = (2N/S^2) * conv_kmax
    # df/dh = (2N/S^2) dconv - (4N/S^3) conv_kmax * dS
    grad_h = (2.0 * N / S2) * dconv - (4.0 * N * conv[kmax] / (S2 * S)) * dS
    grad_theta = _BASIS.T @ grad_h
    return obj, grad_theta


def _objective_only(theta):
    h = _BASIS @ theta
    S = float(np.sum(h))
    if S == 0.0:
        return 1e18
    conv = np.einsum('kij,i,j->k', _KERNEL, h, h)
    return 2.0 * N * float(np.max(conv)) / (S * S)


# ---------------------------------------------------------------------------
# Basis: smooth windowed Zadoff-Chu (near-zero autocorrelation core) blended
# with low-frequency Chebyshev modes (smooth global shape).
# ---------------------------------------------------------------------------
def _build_basis():
    n = N
    x = np.linspace(-1.0, 1.0, n)
    basis = []

    # Windowed Zadoff-Chu (real part) with a few window widths
    m = 7  # CAZAC order (odd)
    zc = np.exp(-1j * np.pi * m * np.arange(n) * (np.arange(n) + 1) / n)
    for w in (0.55, 0.75, 0.95):
        win = 0.5 * (1.0 + np.cos(np.pi * (x / w)))
        win = np.clip(win, 0.0, 1.0)
        basis.append(np.real(zc) * win)
        basis.append(np.imag(zc) * win)

    # Low-frequency Chebyshev modes (smooth global shape)
    for j in range(0, 9):
        basis.append(np.cos(j * np.arccos(np.clip(x, -1.0, 1.0))))

    B = np.column_stack(basis)
    # Orthonormalize columns (helps BFGS conditioning)
    Q, _ = np.linalg.qr(B)
    return Q


_BASIS = _build_basis()


def _best_from_theta(theta):
    h = _BASIS @ theta
    if not np.all(np.isfinite(h)):
        return None, 1e18
    S = float(np.sum(h))
    if S == 0.0:
        return None, 1e18
    conv = np.einsum('kij,i,j->k', _KERNEL, h, h)
    val = 2.0 * N * float(np.max(conv)) / (S * S)
    return h, val


def construct():
    t0 = time.time()
    best_h = None
    best_val = 1e18

    def _consider(theta):
        nonlocal best_h, best_val
        h, v = _best_from_theta(theta)
        if h is not None and v < best_val:
            best_h, best_val = h, v

    # ---- Seed candidates -------------------------------------------------
    rng = np.random.default_rng(0)
    npar = _BASIS.shape[1]
    starts = []
    # All-ones direction projected onto basis (reproduces seed value 2.0)
    ones = np.ones(N)
    starts.append(_BASIS.T @ ones)
    # Random smooth starts
    for _ in range(12):
        starts.append(rng.normal(size=npar))

    # ---- BFGS on each start with deadline guard --------------------------
    for s in starts:
        if time.time() > DEADLINE:
            break
        try:
            res = _scipy_minimize(_objective_and_grad, s, method="L-BFGS-B",
                                  options={"maxiter": 60, "maxls": 40})
            if res.success or res.fun < best_val:
                _consider(res.x)
        except Exception:
            pass
        if time.time() > DEADLINE:
            break

    # ---- Fallbacks -------------------------------------------------------
    if best_h is None:
        best_h = np.ones(N)
        best_val = 2.0

    # Safety: ensure finite, nonzero sum
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0.0:
        best_h = np.ones(N)

    return best_h


# Lazy import of scipy inside construct to avoid import-time cost surprises
from scipy.optimize import minimize as _scipy_minimize
