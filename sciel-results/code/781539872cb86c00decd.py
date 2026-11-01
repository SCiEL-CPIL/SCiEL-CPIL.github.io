"""Chebyshev collocation approach to minimize uncertainty product upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 180


def _chebyshev_setup(N, L):
    """
    Set up Chebyshev collocation points, weights, and derivative matrices on [-L, L].
    """
    N = int(N)
    
    # Chebyshev points on [-1, 1]
    j = np.arange(N + 1)
    t = np.cos(np.pi * j / N)
    t = t[::-1]  # ascending order
    x = L * t
    
    # Weights for Chebyshev quadrature
    w = np.ones(N + 1) * np.pi / N
    w[0] *= 0.5
    w[-1] *= 0.5
    w = L * w  # scale to [-L, L]
    
    # Derivative matrices
    c = np.ones(N + 1)
    c[0] = 0.5
    c[-1] = 0.5
    c = c * (-1.0) ** np.arange(N + 1)
    
    X = np.tile(t, (N + 1, 1))
    dX = X - X.T
    dX[np.abs(dX) < 1e-15] = 1.0
    
    # Off-diagonal
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    # Diagonal
    D = D - np.diag(D.sum(axis=1))
    
    # Scale to [-L, L]
    D1 = D / L
    D2 = D1 @ D1
    
    return x, w, D1, D2


def _build_functional(N, L):
    """
    Build the quadratic forms for position and momentum variances.
    
    g(x) = P(x) * exp(-x^2/2) where P is a polynomial of degree <= 12
    with coefficients (a0, a4, a8, a12) where a12 is determined by g(0)=0.
    
    We parameterize by (a0, a4, a8) and compute a12 = -(a0 + 3*a4 + 105*a8)/945.
    
    g(x) = sum_{n in {0,4,8,12}} a_n H_n(x) exp(-x^2/2)
    """
    x, w, D1, D2 = _chebyshev_setup(N, L)
    
    # Hermite polynomials H_n(x) for n = 0, 4, 8, 12
    def H0(x): return np.ones_like(x)
    def H4(x): return x**4 - 6*x**2 + 3
    def H8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    def H12(x): return x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945
    
    # Basis functions: phi_n(x) = H_n(x) * exp(-x^2/2)
    exp_factor = np.exp(-x**2 / 2.0)
    phi0 = H0(x) * exp_factor
    phi4 = H4(x) * exp_factor
    phi8 = H8(x) * exp_factor
    phi12 = H12(x) * exp_factor
    
    phis = np.array([phi0, phi4, phi8, phi12])  # shape (4, N+1)
    
    # M_0[i,j] = <phi_i, phi_j>
    M_0 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M_0[i, j] = np.sum(w * phis[i] * phis[j])
    
    # M_x[i,j] = <phi_i, x * phi_j>
    M_x = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M_x[i, j] = np.sum(w * x * phis[i] * phis[j])
    
    # M_x2[i,j] = <phi_i, x^2 * phi_j>
    M_x2 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M_x2[i, j] = np.sum(w * x**2 * phis[i] * phis[j])
    
    # M_p2[i,j] = <phi_i, -D2 * phi_j>
    M_p2 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            D2_phi_j = D2 @ phis[j]
            M_p2[i, j] = -np.sum(w * phis[i] * D2_phi_j)
    
    return M_0, M_x, M_x2, M_p2


def _variance_from_matrices(params, M_0, M_x, M_x2, M_p2):
    """
    Compute the true variance of position and momentum.
    
    Var(x) = <x^2>/||g||^2 - (<x>/||g||)^2
    Var(p) = <p^2>/||g||^2 - (<p>/||g||)^2
    
    For even-n Hermite functions, <p> = 0 (since the wavefunction is even).
    So Var(p) = <p^2>/||g||^2.
    
    But <x> may not be zero in general. However, for our specific combination
    of even-n Hermite functions, <x> = 0 as well (since x*phi_n is odd for even n).
    So Var(x) = <x^2>/||g||^2.
    
    Actually, let me be more careful. The task says "A(f)*A(Fourier(f))" where
    A(f) is presumably the standard deviation (square root of variance).
    
    For even functions, <x> = 0 and <p> = 0, so A(f) = sqrt(<x^2>) and A(F) = sqrt(<p^2>).
    
    The uncertainty product is sqrt(<x^2> * <p^2>) where these are the variances
    (which equal the second moments since the means are zero).
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    a = np.array([a0, a4, a8, a12])
    
    # ||g||^2 = a^T M_0 a
    norm2 = a @ M_0 @ a
    if norm2 <= 1e-30 or not np.isfinite(norm2):
        return 1e18
    
    # <x^2> = a^T M_x2 a
    x2_moment = a @ M_x2 @ a
    if x2_moment <= 0 or not np.isfinite(x2_moment):
        return 1e18
    
    # <p^2> = a^T M_p2 a
    p2_moment = a @ M_p2 @ a
    if p2_moment <= 0 or not np.isfinite(p2_moment):
        return 1e18
    
    # Since <x> = 0 and <p> = 0 for even functions:
    var_x = x2_moment / norm2
    var_p = p2_moment / norm2
    
    if var_x <= 0 or var_p <= 0:
        return 1e18
    
    return np.sqrt(var_x * var_p)


def _bound_and_grad(params, M_0, M_x, M_x2, M_p2):
    """
    Compute bound and analytic gradient.
    
    bound = sqrt( (a^T M_x2 a) * (a^T M_p2 a) / (a^T M_0 a)^2 )
    
    Let N = a^T M_0 a, P = a^T M_x2 a, Q = a^T M_p2 a
    bound = sqrt(P * Q / N^2)
    
    d(bound)/da_c = (1/(2*bound)) * d(P*Q/N^2)/da_c
    d(P*Q/N^2)/da_c = (dP*Q + P*dQ)/N^2 - 2*P*Q*dN/N^3
    where dP/da_c = 2 * (M_x2 @ a)_c, etc.
    
    But a depends on params through a12.
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    a = np.array([a0, a4, a8, a12])
    
    N = a @ M_0 @ a
    P = a @ M_x2 @ a
    Q = a @ M_p2 @ a
    
    if N <= 1e-30 or P <= 0 or Q <= 0 or not np.isfinite(N) or not np.isfinite(P) or not np.isfinite(Q):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(P * Q / (N * N))
    
    # da/dparams (4x3)
    da_dparams = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0/945.0, -3.0/945.0, -105.0/945.0]
    ])
    
    # dN/dparams = 2 * (M_0 @ a)^T @ da_dparams
    M0a = M_0 @ a
    Mx2a = M_x2 @ a
    Mp2a = M_p2 @ a
    
    dN_dparams = 2.0 * (da_dparams @ M0a)  # shape (3,)
    dP_dparams = 2.0 * (da_dparams @ Mx2a)
    dQ_dparams = 2.0 * (da_dparams @ Mp2a)
    
    # d(P*Q/N^2)/dparams
    dPQN2 = (dP_dparams * Q + P * dQ_dparams) / (N * N) - 2.0 * P * Q * dN_dparams / (N * N * N)
    
    grad = dPQN2 / (2.0 * bound)
    
    if not np.all(np.isfinite(grad)):
        grad = np.array([1e18, 1e18, 1e18])
    
    return bound, grad


def construct():
    """
    Find the three coefficients (a0, a4, a8) that minimize the uncertainty product
    upper bound, with a12 determined by the constraint g(0) = 0.
    """
    # Use Chebyshev collocation with moderate N
    N = 300  # number of intervals, N+1 points
    L = 10.0  # domain half-width
    
    try:
        M_0, M_x, M_x2, M_p2 = _build_functional(N, L)
    except Exception:
        # Fallback to a simple approach
        return np.array([-1.13, 0.04, 0.000308642])
    
    def _obj(params):
        val, _ = _bound_and_grad(params, M_0, M_x, M_x2, M_p2)
        return val
    
    def _obj_grad(params):
        val, grad = _bound_and_grad(params, M_0, M_x, M_x2, M_p2)
        return val, grad
    
    # Multiple starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0001]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-2.5, 0.12, 0.0015]),
        np.array([-0.3, 0.01, 0.0001]),
        np.array([-1.8, 0.06, 0.0006]),
        np.array([0.5, -0.02, 0.0003]),
        np.array([-4.0, 0.2, 0.003]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(20):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_obj_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with SLSQP
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_obj_grad, best_params, method='SLSQP',
                          jac=_obj_grad, options={'maxiter': 5000, 'ftol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_obj_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
