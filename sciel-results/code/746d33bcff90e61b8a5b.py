# EVOLVE-BLOCK-START
import time
import numpy as np

_K = 10
_NTH = 70   # Gauss-Legendre in cos(theta)
_NPHI = 130  # equally spaced phi


def _build_grid_and_harmonics():
    """Build the quadrature grid and real orthonormal degree-10 harmonics.

    Returns:
        Y : (N, 21) array, Y[i, j] = Y_{m=j-10}(point i), real orthonormal basis.
        w : (N,) quadrature weights summing to 4*pi (surface measure).
    """
    k = _K
    # Gauss-Legendre nodes/weights on [-1,1]
    x, wx = np.polynomial.legendre.leggauss(_NTH)  # x in [-1,1]
    # theta in [0, pi], cos(theta) = x. phi in [0, 2*pi)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    ct = x[:, None]            # (NTH, 1)
    st = np.sqrt(np.clip(1.0 - ct * ct, 0.0, 1.0))  # sin(theta)
    cphi = np.cos(phi)[None, :]  # (1, NPHI)
    sphi = np.sin(phi)[None, :]

    # meshgrid
    CT, CP = np.meshgrid(ct, cphi, indexing='ij')   # (NTH, NPHI)
    ST, SP = np.meshgrid(st, sphi, indexing='ij')
    # flatten
    CT = CT.ravel(); ST = ST.ravel(); CP = CP.ravel(); SP = SP.ravel()

    # weights: dOmega = dphi dcos(theta); phi spacing = 2pi/NPHI, theta weight wx
    dphi = 2.0 * np.pi / _NPHI
    # w = dphi * wx (broadcast)
    w = np.outer(wx, np.ones(_NPHI)).ravel() * dphi
    # total mass = 4*pi
    w = w * (4.0 * np.pi) / w.sum()

    # Associated Legendre P_l^m with Condon-Shortley phase, via scipy
    from scipy.special import lpmv
    N = CT.size
    Y = np.empty((N, 2 * k + 1), dtype=np.float64)
    for m in range(-k, k + 1):
        am = abs(m)
        # P_l^m(cos theta) with Condon-Shortley phase
        P = lpmv(am, k, CT)  # includes Condon-Shortley (-1)^m
        Nm = np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                     float(_factorial(k - am) / _factorial(k + am)))
        if m > 0:
            Y[:, m + k] = np.sqrt(2.0) * Nm * P * CP
        elif m < 0:
            Y[:, m + k] = np.sqrt(2.0) * Nm * P * SP
        else:
            Y[:, 0 + k] = Nm * P
    return Y, w


def _factorial(n):
    return float(np.math.factorial(n))


def _objective_and_grad(c, Y, w):
    """Return R = ||f||_4/||f||_2 and its gradient wrt c."""
    f = Y @ c
    # weighted sums (w sums to 4*pi)
    f2 = np.sum(w * f * f)            # ||f||_2^2
    f4 = np.sum(w * f * f * f * f)    # ||f||_4^4
    # R = (f4)^(1/4) / (f2)^(1/2)
    R = (f4 ** 0.25) / np.sqrt(f2)
    # gradient
    # dR/dc = (1/4) f4^(-3/4) * 4 sum(w f^3 (Y c)_m) * f2^(-1/2)
    #        - f4^(1/4) * f2^(-3/2) * 2 sum(w f Y_m)
    # = f4^(-3/4) f2^(-1/2) [ Y^T (w f^3) - 2 (f4/f2) Y^T (w f) ]
    wfm3 = w * f * f * f
    wfm1 = w * f
    term1 = Y.T @ wfm3
    term2 = (f4 / f2) * (Y.T @ wfm1)
    grad = (f4 ** (-0.75)) * (f2 ** (-0.5)) * (term1 - 2.0 * term2)
    return R, grad


def _optimize(Y, w, c0, maxiter, time_budget):
    """Local L-BFGS-B from c0. Returns best (R, c)."""
    from scipy.optimize import minimize
    start = time.time()

    def fun(c):
        R, _ = _objective_and_grad(c, Y, w)
        return R

    def jac(c):
        _, g = _objective_and_grad(c, Y, w)
        return g

    best_R = np.inf
    best_c = c0.copy()
    try:
        res = minimize(fun, c0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'gtol': 1e-12,
                                'ftol': 1e-14, 'maxls': 50,
                                'maxfun': maxiter * 12})
        if res.success or res.fun < best_R:
            best_R = float(res.fun)
            best_c = res.x.copy()
    except Exception:
        pass
    # verify
    Rv, _ = _objective_and_grad(best_c, Y, w)
    if np.isfinite(Rv):
        best_R = Rv
        best_c = best_c.copy()
    return best_R, best_c


def rudin_problem10() -> np.ndarray:
    k = _K
    Y, w = _build_grid_and_harmonics()

    deadline = time.time() + 600.0 - 30.0  # safety margin

    # candidate initializations
    rng = np.random.default_rng(12345)
    inits = []
    inits.append(np.ones(2 * k + 1, dtype=np.float64))
    # random
    for _ in range(12):
        inits.append(rng.normal(size=2 * k + 1))
    # single-mode
    for m in range(-k, k + 1):
        v = np.zeros(2 * k + 1)
        v[m + k] = 1.0
        inits.append(v)
    # even/odd symmetric patterns
    for _ in range(6):
        v = rng.normal(size=2 * k + 1)
        v = (v + v[::-1]) / 2.0  # symmetric in m
        inits.append(v)

    best_R = np.inf
    best_c = None
    for c0 in inits:
        if time.time() > deadline:
            break
        R, c = _optimize(Y, w, c0, maxiter=400, time_budget=deadline)
        if R < best_R:
            best_R = R
            best_c = c
    if best_c is None:
        best_c = np.ones(2 * k + 1, dtype=np.float64)

    # final polish with tighter tolerance
    if time.time() < deadline:
        R, c = _optimize(Y, w, best_c, maxiter=2000, time_budget=deadline)
        if R < best_R:
            best_R = R
            best_c = c

    # validity guard: scale into [1e-3, 1e5]
    c = np.asarray(best_c, dtype=np.float64)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    mx = np.max(np.abs(c))
    if mx < 1e-3:
        c = c * (1.0 / max(mx, 1e-12))
        c = c * 1e-3
    elif mx > 1e5:
        c = c * (1e5 / mx)
    return c


# EVOLVE-BLOCK-END
