# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 630.0
_MARGIN = 15.0
_K = 10
_NTH = 70          # Gauss-Legendre nodes in cos(theta)
_NPHI = 130        # equally spaced phi
_NPT = _NTH * _NPHI


def _build_basis():
    """Return (theta_grid, phi_grid, weights, B) where B is (21, NPT)."""
    xgl, wgl = np.polynomial.legendre.leggauss(_NTH)
    theta = np.arccos(xgl)
    phi = (np.arange(_NPHI) / _NPHI) * 2.0 * np.pi

    theta_g, phi_g = np.meshgrid(theta, phi, indexing='ij')
    ct = np.cos(theta_g)
    st = np.sin(theta_g)
    w = wgl * (2.0 / _NPHI)
    w = np.tile(w[:, None], (1, _NPHI)).ravel()

    def legendre_nm(n, m, x):
        m = abs(m)
        if m > n:
            return np.zeros_like(x)
        # P_{m}^{m} = (-1)^m (2m-1)!! (1-x^2)^{m/2}
        pm = np.ones_like(x)
        if m > 0:
            pm = ((-1.0) ** m) * (2.0 * m - 1.0) * np.prod(np.arange(1, 2 * m, 2))
            pm = pm * (1.0 - x * x) ** (0.5 * m)
        if m == n:
            return pm
        pm1 = (2.0 * m + 1.0) * x * pm
        for n_ in range(m + 1, n + 1):
            pn = ((2.0 * n_ - 1.0) * x * pm1 - (n_ + m - 1.0) * pm) / (n_ - m)
            pm, pm1 = pm1, pn
        return pm1

    B = np.empty((2 * _K + 1, _NPT), dtype=np.float64)
    for m in range(-_K, _K + 1):
        am = abs(m)
        N = np.sqrt((2 * _K + 1) / (4.0 * np.pi) *
                    np.math.factorial(_K - am) / np.math.factorial(_K + am))
        P = legendre_nm(_K, am, ct)
        if m == 0:
            B[m + _K] = N * P
        elif m > 0:
            B[m + _K] = np.sqrt(2.0) * N * P * np.cos(m * phi_g)
        else:
            B[m + _K] = np.sqrt(2.0) * N * P * np.sin(am * phi_g)
    return theta_g, phi_g, w, B


_THETA, _PHI, _W, _B = _build_basis()
_W2 = _W * _W
_B2T = (_B * _W) @ _W2[:, None]   # (21, 21)  second-moment matrix


def _R_and_grad(c):
    """Return (R, grad) for f = c @ B at the quadrature nodes."""
    f = _B @ c
    af = np.abs(f)
    af2 = af * af
    af4 = af2 * af2

    log_af2 = np.log(af2 + 1e-300)
    m2 = np.max(_W * log_af2)
    Z2 = np.exp(np.log(np.sum(_W * np.exp(log_af2 - m2))) + m2)
    l2 = np.sqrt(Z2)

    log_af4 = np.log(af4 + 1e-300)
    m4 = np.max(_W * log_af4)
    Z4 = np.exp(np.log(np.sum(_W * np.exp(log_af4 - m4))) + m4)
    l4 = np.sqrt(np.sqrt(Z4))

    R = l4 / l2

    # dR/dc = (l4/(2 l2)) * ( (1/Z4) dZ4/dc - (1/Z2) dZ2/dc )
    # dZ4/dc = 4 * sum(w |f|^2 * f * B),  dZ2/dc = 2 * sum(w f * B)
    dZ4 = 4.0 * (_W2 * af2 * f) @ _B
    dZ2 = 2.0 * (_W * f) @ _B
    grad = (l4 / (2.0 * l2)) * (dZ4 / Z4 - dZ2 / Z2)
    return R, grad


def _R_only(c):
    f = _B @ c
    af = np.abs(f)
    af2 = af * af
    af4 = af2 * af2
    l4 = np.sqrt(np.sqrt(np.sum(_W * af4)))
    l2 = np.sqrt(np.sum(_W * af2))
    return l4 / l2


def _rotated_monomial_seed(rng, axis=None):
    """Re[(a . r)^10] expressed in the real degree-10 basis."""
    if axis is None:
        v = rng.standard_normal(3)
        axis = v / (np.linalg.norm(v) + 1e-30)
    a = axis
    # coefficients of (a1 x + a2 y + a3 z)^10  ->  (x + b y + c z)^10 with b=a2/a1,c=a3/a1
    if abs(a[0]) < 1e-9:
        # use y or z as leading; handle by permuting
        # fall back: expand with symbolic-free approach using y as leading
        b, c = a[2] / a[1], 0.0
        # (y + b z)^10 ; map x->y, y->z, z->0
        def map_coeff(j, i, h):  # x^j y^i z^h ; x=y, y=z, z=0
            if h != 0:
                return 0.0
            return (j, i, 0)
        # build polynomial in (x', y', z') = (y, z, 0)
        # we need (x' + b y')^10 with x'=y, y'=z
        # coefficient of x'^j y'^(10-j)
        # then map x'^j y'^(10-j) -> y^j z^(10-j)
        coeff = {}
        for j in range(11):
            coeff[(j, 10 - j, 0)] = float(np.math.factorial(10) /
                                          (np.math.factorial(j) * np.math.factorial(10 - j))) * (b ** (10 - j))
        # map (x^j y^i z^h) with x=y,y=z,z=0 : y^j z^i -> x^j y^i z^0  => (j, i, 0)
        # but here j is power of y(=x), i power of z(=y)
        mapped = {}
        for (j, i, h), v in coeff.items():
            mapped[(j, i, 0)] = v
    else:
        b, c = a[1] / a[0], a[2] / a[0]
        mapped = {}
        for j in range(11):
            for i in range(11 - j):
                h = 10 - j - i
                v = (float(np.math.factorial(10) /
                           (np.math.factorial(j) * np.math.factorial(i) * np.math.factorial(h)))
                     * (b ** i) * (c ** h))
                mapped[(j, i, h)] = v

    # map (x^j y^i z^h) -> basis index
    def basis_index(j, i, h):
        if h > 0:
            return None
        if i > 0:
            if j < 0:
                return None
            return i + _K
        else:
            if j < 0:
                return None
            return -j + _K

    seed = np.zeros(2 * _K + 1)
    for (j, i, h), v in mapped.items():
        idx = basis_index(j, i, h)
        if idx is not None and 0 <= idx < 2 * _K + 1:
            seed[idx] += v
    nrm = np.linalg.norm(seed)
    if nrm > 1e-12:
        seed = seed / nrm
    return seed


def _random_seed(rng):
    c = rng.standard_normal(2 * _K + 1)
    nrm = np.linalg.norm(c)
    if nrm < 1e-12:
        return c
    return c / nrm


def _optimize_from(c0, time_budget):
    """L-BFGS-B from c0. Returns (best_c, best_R)."""
    from scipy.optimize import minimize
    c0 = np.array(c0, dtype=np.float64).copy()
    best_c, best_R = c0, _R_only(c0)

    def fun(x):
        R, _ = _R_and_grad(x)
        return R

    def jac(x):
        _, g = _R_and_grad(x)
        return g

    t_end = time.time() + time_budget
    # coarse
    try:
        res = minimize(fun, c0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-14, 'gtol': 1e-8})
        if res.x is not None:
            Rv = _R_only(res.x)
            if Rv < best_R:
                best_R, best_c = Rv, res.x
    except Exception:
        pass
    if time.time() > t_end:
        return best_c, best_R
    # polish
    try:
        res = minimize(fun, best_c, jac=jac, method='L-BFGS-B',
                       options={'maxiter': 400, 'maxfun': 800, 'ftol': 1e-16, 'gtol': 1e-10})
        if res.x is not None:
            Rv = _R_only(res.x)
            if Rv < best_R:
                best_R, best_c = Rv, res.x
    except Exception:
        pass
    return best_c, best_R


def rudin_problem10() -> np.ndarray:
    rng = np.random.default_rng(12345)
    t0 = time.time()

    seeds = []
    # all-ones (the seed baseline)
    ones = np.ones(2 * _K + 1)
    seeds.append(ones / np.linalg.norm(ones))
    # high-symmetry rotated monomial seeds
    for _ in range(8):
        seeds.append(_rotated_monomial_seed(rng))
    # random seeds
    for _ in range(12):
        seeds.append(_random_seed(rng))

    # give each seed a share of the time budget
    n_seed = len(seeds)
    base_budget = (_DEADLINE - _MARGIN - (time.time() - t0)) / n_seed
    base_budget = max(base_budget, 2.0)

    best_c, best_R = ones, _R_only(ones)
    for s in seeds:
        if time.time() > _DEADLINE - _MARGIN:
            break
        rem = _DEADLINE - _MARGIN - time.time()
        budget = min(base_budget, rem)
        if budget < 1.0:
            break
        try:
            c, R = _optimize_from(s, budget)
        except Exception:
            continue
        if R < best_R:
            best_R, best_c = R, c

    # final polish on the global best if time remains
    rem = _DEADLINE - _MARGIN - time.time()
    if rem > 2.0:
        try:
            c, R = _optimize_from(best_c, rem)
            if R < best_R:
                best_R, best_c = R, c
        except Exception:
            pass

    # scale to a safe, normalized magnitude (scale-invariant objective)
    mx = np.max(np.abs(best_c))
    if mx > 0:
        best_c = best_c * (1.0 / mx)
    # safety: ensure within [1e-3, 1e5]
    mx2 = np.max(np.abs(best_c))
    if not (np.isfinite(mx2)) or mx2 < 1e-3 or mx2 > 1e5:
        best_c = np.ones(2 * _K + 1)
    return best_c.astype(np.float64)


# EVOLVE-BLOCK-END
