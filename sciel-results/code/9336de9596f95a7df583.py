"""Chebyshev collocation approach for uncertainty bound minimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195

# Precompute Chebyshev nodes and weights for collocation on [-L, L]
_L = 6.0
_N = 200  # number of collocation points

# Chebyshev nodes: x_k = L * cos(pi * (k + 0.5) / N)
_cheb_x = _L * np.cos(np.pi * (np.arange(_N) + 0.5) / _N)
_cheb_w = np.full(_N, np.pi / _N)
_cheb_w[0] /= 2
_cheb_w[-1] /= 2
# Adjust weights for interval [-L, L]: multiply by L
_cheb_w *= _L


def _hermite_poly(x, n):
    """Compute probabilists' Hermite polynomial He_n(x) via recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    h0 = np.ones_like(x)
    h1 = x.copy()
    for k in range(2, n + 1):
        h2 = x * h1 - (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _compute_bound_and_grad(params):
    """
    Compute uncertainty product using Chebyshev collocation.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    
    Uses collocation to estimate integrals:
    A(f)^2 = <x^2>/<1> for |g|^2
    A(F)^2 = <xi^2>/<1> for |G|^2
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    x = _cheb_x
    w = _cheb_w

    # Compute Hermite polynomials at collocation points
    H0 = np.ones_like(x)
    H4 = _hermite_poly(x, 4)
    H8 = _hermite_poly(x, 8)
    H12 = _hermite_poly(x, 12)

    # Polynomial part
    P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12

    # g(x) = P(x) * exp(-x^2/2)
    g = P * np.exp(-0.5 * x * x)
    g2 = g * g

    # Position space integrals
    m0 = np.sum(w * g2)
    m2 = np.sum(w * x * x * g2)

    if m0 <= 1e-30 or not np.isfinite(m0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    # Fourier transform: G(xi) = sqrt(2*pi) * integral g(x) * exp(-i*xi*x) dx
    # For real even g, G is real even. Use collocation for the FT.
    # G(xi) = sqrt(2*pi) * sum_k w_k * g(x_k) * cos(xi * x_k)
    # |G(xi)|^2 = 2*pi * (sum_k w_k * g(x_k) * cos(xi * x_k))^2
    # We need n0 = integral |G|^2 dxi and n2 = integral xi^2 |G|^2 dxi
    # By Parseval: n0 = 2*pi * integral |g|^2 dx = 2*pi * m0
    # For n2, we need the second moment of |G|^2.
    # n2 = 2*pi * integral xi^2 |G(xi)|^2 dxi
    #     = 2*pi * integral |(-x^2 g(x))|^2 dx  (since xi^2 G = FT of -x^2 g)
    #     = 2*pi * integral x^4 |g(x)|^2 dx

    m4 = np.sum(w * x * x * x * x * g2)

    vf = m2 / m0
    vF = (2.0 * np.pi * m4) / (2.0 * np.pi * m0)
    vF = m4 / m0

    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # Analytic gradient via automatic differentiation of the collocation sums
    # dm0/da_c, dm2/da_c, dm4/da_c
    dP_da = np.zeros((3, len(x)))
    dP_da[0] = H0 + 945.0 * (-1.0/945.0) * H12
    dP_da[1] = H4 + 945.0 * (-3.0/945.0) * H12
    dP_da[2] = H8 + 945.0 * (-105.0/945.0) * H12

    dg_da = dP_da * np.exp(-0.5 * x * x)  # shape (3, N)

    # d(m_k)/da_c = sum w * x^(2k) * 2*g * dg/da_c
    dm0 = 2.0 * np.sum(w * g2[:, None] * dg_da, axis=1)
    # Actually: d(g^2)/da = 2*g*dg/da
    dm0 = 2.0 * np.sum(w * g * dg_da, axis=1)
    dm2 = 2.0 * np.sum(w * x * x * g * dg_da, axis=1)
    dm4 = 2.0 * np.sum(w * x * x * x * x * g * dg_da, axis=1)

    # d(vf)/da_c = (dm2/m0 - m2*dm0/m0^2)
    # d(vF)/da_c = (dm4/m0 - m4*dm0/m0^2)
    # d(bound)/da_c = (vF*d(vf) + vf*d(vF)) / (2*bound)

    dvf = (dm2 * m0 - m2 * dm0) / (m0 * m0)
    dvF = (dm4 * m0 - m4 * dm0) / (m0 * m0)

    grad = (vF * dvf + vf * dvF) / (2.0 * bound)

    return bound, grad


def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    return val, grad


def _time_remaining():
    return _DEADLINE - time.time()


def construct():
    # Multi-start with diverse initial points
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    
    rng = np.random.default_rng(12345)
    
    starts = [seed_params.copy()]
    
    # Add structured starts around the seed
    for scale in [0.5, 1.0, 2.0]:
        for _ in range(5):
            pert = rng.normal(0, scale * np.array([0.5, 0.02, 0.0002]), 3)
            starts.append(seed_params + pert)
    
    # Add some global starts
    for _ in range(10):
        starts.append(rng.normal(0, 1.0, 3))

    best_val = np.inf
    best_params = None

    for start in starts:
        if _time_remaining() < 30:
            break
        try:
            # Use SLSQP as suggested in the directive
            res = minimize(_objective_grad, start, method='SLSQP',
                          jac=True, options={'maxiter': 500, 'ftol': 1e-12})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        return seed_params.copy()

    # Refine with L-BFGS-B
    if _time_remaining() > 20:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final SLSQP polish
    if _time_remaining() > 10:
        try:
            res = minimize(_objective_grad, best_params, method='SLSQP',
                          jac=True, options={'maxiter': 200, 'ftol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
