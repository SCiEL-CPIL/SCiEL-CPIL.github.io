"""Chebyshev collocation approach to minimize uncertainty product upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 175


def _chebyshev_setup(N):
    """
    Set up Chebyshev collocation points, weights, and derivative matrices.
    We work on [-L, L] with N+1 Chebyshev points.
    
    Returns:
        x: collocation points (N+1,)
        w: quadrature weights (N+1,)
        D1: first derivative matrix (N+1, N+1)
        D2: second derivative matrix (N+1, N+1)
    """
    N = int(N)
    L = 10.0  # domain half-width - wider to capture more of the Gaussian tail
    
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
    # Standard Chebyshev differentiation matrix on [-1, 1]
    c = np.ones(N + 1)
    c[0] = 0.5
    c[-1] = 0.5
    c = c * (-1.0) ** np.arange(N + 1)
    
    X = np.tile(t, (N + 1, 1))
    dX = X - X.T
    dX[np.abs(dX) < 1e-15] = 1.0  # avoid division by zero
    
    # Off-diagonal
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    # Diagonal
    D = D - np.diag(D.sum(axis=1))
    
    # Scale to [-L, L]
    D1 = D / L
    
    # D2 = D1 @ D1
    D2 = D1 @ D1
    
    return x, w, D1, D2


def _build_functional(N, L=10.0):
    """
    Build the quadratic forms for position and momentum variances.
    
    g(x) = sum_{n in {0,4,8,12}} a_n H_n(x) exp(-x^2/2)
    with a12 = -(a0 + 3*a4 + 105*a8)/945 to ensure g(0) = 0.
    
    Uses Chebyshev collocation to build 4x4 matrices for:
    - M_0: <phi_i, phi_j>
    - M_pos: <phi_i, x^2 phi_j>  
    - M_mom: <phi_i, -d^2/dx^2 phi_j>
    """
    x, w, D1, D2 = _chebyshev_setup(N)
    
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
    
    # M_0: <phi_i, phi_j>
    M_0 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M_0[i, j] = np.sum(w * phis[i] * phis[j])
    
    # M_pos: <phi_i, x^2 phi_j>
    x2 = x**2
    M_pos = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M_pos[i, j] = np.sum(w * x2 * phis[i] * phis[j])
    
    # M_mom: <phi_i, -D2 phi_j>
    M_mom = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            D2_phi_j = D2 @ phis[j]
            M_mom[i, j] = -np.sum(w * phis[i] * D2_phi_j)
    
    return M_0, M_pos, M_mom


def _bound_from_matrices(params, M_0, M_pos, M_mom):
    """
    Compute the uncertainty bound from the precomputed matrices.
    params = (a0, a4, a8)
    a12 = -(a0 + 3*a4 + 105*a8)/945
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    a = np.array([a0, a4, a8, a12])
    
    # <g, g> = a^T M_0 a
    norm2 = a @ M_0 @ a
    if norm2 <= 1e-30 or not np.isfinite(norm2):
        return 1e18
    
    # <g, x^2 g> = a^T M_pos a
    pos_var = a @ M_pos @ a
    if pos_var <= 0 or not np.isfinite(pos_var):
        return 1e18
    
    # <g, p^2 g> = a^T M_mom a
    mom_var = a @ M_mom @ a
    if mom_var <= 0 or not np.isfinite(mom_var):
        return 1e18
    
    vf = pos_var / norm2
    vF = mom_var / norm2
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)


def _bound_and_grad_cheb(params, M_0, M_pos, M_mom):
    """
    Compute bound and analytic gradient.
    
    bound = sqrt( (a^T M_pos a) * (a^T M_mom a) / (a^T M_0 a)^2 )
    where a = [a0, a4, a8, a12] and a12 = -(a0 + 3*a4 + 105*a8)/945.
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    a = np.array([a0, a4, a8, a12])
    
    N = a @ M_0 @ a
    P = a @ M_pos @ a
    Q = a @ M_mom @ a
    
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
    
    M0a = M_0 @ a
    Mposa = M_pos @ a
    Mmoma = M_mom @ a
    
    dN_dparams = 2.0 * (da_dparams @ M0a)
    dP_dparams = 2.0 * (da_dparams @ Mposa)
    dQ_dparams = 2.0 * (da_dparams @ Mmoma)
    
    dPQN2 = (dP_dparams * Q + P * dQ_dparams) / (N * N) - 2.0 * P * Q * dN_dparams / (N * N * N)
    
    grad = dPQN2 / (2.0 * bound)
    
    if not np.all(np.isfinite(grad)):
        grad = np.array([1e18, 1e18, 1e18])
    
    return bound, grad


def construct():
    # Use Chebyshev collocation with higher N for better accuracy
    N = 300  # more points for improved accuracy
    
    try:
        M_0, M_pos, M_mom = _build_functional(N)
    except Exception:
        # Fallback to a simple approach
        return np.array([-1.13, 0.04, 0.000308642])
    
    def _obj(params):
        val, _ = _bound_and_grad_cheb(params, M_0, M_pos, M_mom)
        return val
    
    def _obj_grad(params):
        val, grad = _bound_and_grad_cheb(params, M_0, M_pos, M_mom)
        return val, grad
    
    # Multiple starting points - diverse and strategic
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
        np.array([-1.3, 0.06, 0.0005]),
        np.array([-0.9, 0.05, 0.0003]),
        np.array([-1.6, 0.07, 0.0006]),
        np.array([-2.5, 0.12, 0.0015]),
        np.array([-0.7, 0.04, 0.00025]),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(20):
        starts.append(rng.normal(0, 0.8, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_obj_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: SLSQP polish
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_obj, best_params, method='SLSQP',
                          options={'maxiter': 3000, 'ftol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_obj_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
