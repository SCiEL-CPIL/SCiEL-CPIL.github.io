# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

_DEADLINE = 620.0  # safety margin under 630s wall-clock
_START = time.perf_counter()


def _elapsed():
    return time.perf_counter() - _START


def _deadline_ok(margin=10.0):
    return (_DEADLINE - _elapsed()) > margin


def _assoc_legendre(n, m, x):
    """P_n^m(x) with Condon-Shortley phase, for fixed n, m, vector x."""
    x = np.asarray(x, dtype=np.float64)
    if m < 0:
        raise ValueError("m must be >= 0")
    if m > n:
        raise ValueError("m must be <= n")
    # P_m^m
    p = np.ones_like(x)
    if m > 0:
        fact = 1.0
        for i in range(1, m + 1):
            fact *= (2 * i - 1)
        p = -np.sqrt(fact) * (1.0 - x * x) ** (m / 2.0)
    if m == n:
        return p
    pm1 = p
    if m + 1 <= n:
        # P_{m+1}^m
        pmm1 = x * (2 * m + 1) * p
        pm1 = p
        p = pmm1
    for l in range(m + 2, n + 1):
        pmm1 = ((2 * l - 1) * x * pm1 - (l + m - 1) * p) / (l - m)
        pm1 = p
        p = pmm1
    return p


def _precompute_gaunt():
    """Precompute G[a,b,c,d] = ∫ Y_a Y_b Y_c Y_d over sphere, a,b,c,d in -10..10."""
    k = 10
    n = 2 * k + 1
    # Gauss-Legendre quadrature in cos(theta), 70 points
    n_theta = 70
    cos_t, w_t = np.polynomial.legendre.leggauss(n_theta)
    cos_t = cos_t[::-1]  # ascending for readability
    w_t = w_t[::-1]
    # Equally spaced phi, 130 points
    n_phi = 130
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi
    # Build basis functions on the grid
    # Y_0 = N_0 P_10^0
    # Y_m (m>0) = sqrt(2) N_m P_10^m cos(m phi)
    # Y_m (m<0) = sqrt(2) N_|m| P_10^|m| sin(|m| phi)
    N = np.zeros(n, dtype=np.float64)
    Pabs = np.zeros((n, n_theta), dtype=np.float64)  # P_10^{|m|}(cos_t)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        N[idx] = np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                         float(np.math.factorial(k - am)) / float(np.math.factorial(k + am)))
        Pabs[idx] = _assoc_legendre(k, am, cos_t)
    # Basis values: shape (n, n_theta, n_phi)
    Y = np.zeros((n, n_theta, n_phi), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        if m == 0:
            Y[idx] = N[idx] * Pabs[idx][:, None]
        elif m > 0:
            Y[idx] = np.sqrt(2.0) * N[idx] * Pabs[idx][:, None] * np.cos(m * phi)[None, :]
        else:
            Y[idx] = np.sqrt(2.0) * N[idx] * Pabs[idx][:, None] * np.sin(am * phi)[None, :]
    # Measure weights
    weights = (w_t[:, None] * dphi)[None, :, :]  # (1, n_theta, n_phi)
    # G[a,b,c,d] = sum over grid of Y_a Y_b Y_c Y_d * weights
    # Vectorize: for each (a,b) compute M_ab[theta,phi] = Y_a*Y_b, then G[a,b,c,d] = sum M_ab * Y_c * Y_d * w
    G = np.zeros((n, n, n, n), dtype=np.float64)
    # Precompute YcYd for all (c,d)
    Ycd = np.zeros((n, n, n_theta, n_phi), dtype=np.float64)
    for c in range(n):
        for d in range(n):
            Ycd[c, d] = Y[c] * Y[d]
    for a in range(n):
        Ya = Y[a]
        for b in range(n):
            Mab = Ya * Y[b]
            Mabw = Mab * weights
            # G[a,b,c,d] = sum Mabw * Ycd[c,d]
            G[a, b, :, :] = np.einsum('ij,cdij->cd', Mabw, Ycd, optimize=True)
    return G


def _norm4_sq_and_grad(c, G):
    """Return (||f||_4^4, grad) where f = sum c_m Y_m. Exact via Gaunt."""
    c4 = c[:, None, None, None] * c[None, :, None, None] * c[None, None, :, None] * c[None, None, None, :]
    val = np.einsum('abcd,abcd->', G, c4, optimize=True)
    # grad_j = 4 * G[j,:,:,:] contracted with c,c,c
    grad = 4.0 * np.einsum('abcd,cde->b', G, c, c, c, optimize=True)
    # Wait: G[a,b,c,d] is symmetric in all indices, so
    # d/dc_j sum G[a,b,c,d] c_a c_b c_c c_d = 4 * sum G[j,b,c,d] c_b c_c c_d
    grad = 4.0 * np.einsum('jbcd,bcd->j', G, c, c, c, optimize=True)
    return val, grad


def _objective_and_grad(c, G):
    """Minimize R = ||f||_4 / ||f||_2. Returns (R, grad_R)."""
    n4, g4 = _norm4_sq_and_grad(c, G)
    n2 = np.dot(c, c)
    if n2 <= 1e-30:
        return 1e30, np.zeros_like(c)
    R = np.sqrt(np.sqrt(n4)) / np.sqrt(n2)
    # R = n4^{1/4} * n2^{-1/2}
    # dR/dc = (1/4) n4^{-3/4} g4 * n2^{-1/2} + n4^{1/4} * (-1/2) n2^{-3/2} * 2c
    n4_quarter = n4 ** 0.25
    n4_neg34 = n4 ** (-0.75)
    n2_sqrt = np.sqrt(n2)
    n2_neg32 = n2 ** (-1.5)
    grad = 0.25 * n4_neg34 * g4 / n2_sqrt - n4_quarter * n2_neg32 * c
    return R, grad


def _validate(c):
    if c is None or c.shape != (21,):
        return False
    if not np.all(np.isfinite(c)):
        return False
    m = np.max(np.abs(c))
    if m < 1e-3 or m > 1e5:
        return False
    return True


def _evaluate_R(c, G):
    R, _ = _objective_and_grad(c, G)
    return R


def rudin_problem10() -> np.ndarray:
    k = 10
    n = 2 * k + 1
    G = _precompute_gaunt()

    best_c = np.ones(n, dtype=np.float64)
    best_R = _evaluate_R(best_c, G)

    rng = np.random.default_rng(20240517)
    budget_per_restart = 35.0
    max_restarts = 12

    def _try_from(x0):
        nonlocal best_c, best_R
        x0 = np.asarray(x0, dtype=np.float64).copy()
        if not _validate(x0):
            return
        R0, _ = _objective_and_grad(x0, G)
        res = minimize(
            lambda c: _objective_and_grad(c, G)[0],
            x0,
            jac=lambda c: _objective_and_grad(c, G)[1],
            method='L-BFGS-B',
            options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50},
        )
        if res.success or res.fun < R0:
            c = np.asarray(res.x, dtype=np.float64)
            if _validate(c):
                R = _evaluate_R(c, G)
                if R < best_R:
                    best_R = R
                    best_c = c.copy()

    # Initial: all ones
    _try_from(np.ones(n))

    # Random restarts
    for r in range(max_restarts):
        if not _deadline_ok(margin=15.0):
            break
        # Random direction, normalize to unit L2 to avoid scale issues
        v = rng.standard_normal(n)
        v /= np.linalg.norm(v)
        # Scale to a reasonable magnitude
        scale = 1.0
        x0 = v * scale
        _try_from(x0)

    # Deterministic structured starts: single basis, pairs, cos-only, sin-only
    for m_idx in range(n):
        if not _deadline_ok(margin=15.0):
            break
        v = np.zeros(n); v[m_idx] = 1.0
        _try_from(v)
    # cos-only subspace (m >= 0)
    if _deadline_ok(margin=15.0):
        v = np.zeros(n)
        for idx in range(11, n):  # m = 0..10
            v[idx] = 1.0
        _try_from(v)
    # sin-only subspace (m < 0)
    if _deadline_ok(margin=15.0):
        v = np.zeros(n)
        for idx in range(0, 10):  # m = -10..-1
            v[idx] = 1.0
        _try_from(v)

    # Final polish: high-precision local refinement from best
    if _deadline_ok(margin=10.0):
        def _polish(c):
            nonlocal best_R
            res = minimize(
                lambda c: _objective_and_grad(c, G)[0],
                c,
                jac=lambda c: _objective_and_grad(c, G)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 100},
            )
            if res.success:
                cc = np.asarray(res.x, dtype=np.float64)
                if _validate(cc):
                    R = _evaluate_R(cc, G)
                    if R < best_R:
                        best_R = R
                        best_c = cc.copy()
        _polish(best_c.copy())

    # Ensure valid
    if not _validate(best_c):
        best_c = np.ones(n)

    return best_c


# EVOLVE-BLOCK-END
