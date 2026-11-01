"""Chebyshev collocation approach to minimize uncertainty product upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 170


def _chebyshev_setup(N, L):
    """
    Set up Chebyshev collocation points, weights, and derivative matrices.
    We work on [-L, L] with N+1 Chebyshev points.
    """
    N = int(N)
    j = np.arange(N + 1)
    t = np.cos(np.pi * j / N)
    t = t[::-1]  # ascending order
    x = L * t
    
    # Weights for Chebyshev quadrature
    w = np.ones(N + 1) * np.pi / N
    w[0] *= 0.5
    w[-1] *= 0.5
    w = L * w
    
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
    
    D1 = D / L
    D2 = D1 @ D1
    
    return x, w, D1, D2


def _build_functional(N, L=10.0):
    """
    Build the quadratic forms for position and momentum variances.
    """
    x, w, D1, D2 = _chebyshev_setup(N, L)
    
    # Hermite polynomials H_n(x) for n = 0, 4, 8, 12
    def H0(x): return np.ones_like(x)
    def H4(x): return x**4 - 6*x**2 + 3
    def H8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    def H12(x): return x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945
    
    exp_factor = np.exp(-x**2 / 2.0)
    phi0 = H0(x) * exp_factor
    phi4 = H4(x) * exp_factor
    phi8 = H8(x) * exp_factor
    phi12 = H12(x) * exp_factor
    
    phis = np.array([phi0, phi4, phi8, phi12])
    
    M_0 = np.zeros((4, 4))
    M_pos = np.zeros((4, 4))
    M_mom = np.zeros((4, 4))
    
    x2 = x**2
    
    for i in range(4):
        for j in range(4):
            M_0[i, j] = np.sum(w * phis[i] * phis[j])
            M_pos[i, j] = np.sum(w * x2 * phis[i] * phis[j])
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
    
    norm2 = a @ M_0 @ a
    if norm2 <= 1e-30 or not np.isfinite(norm2):
        return 1e18
    
    pos_var = a @ M_pos @ a
    if pos_var <= 0 or not np.isfinite(pos_var):
        return 1e18
    
    mom_var = a @ M_mom @ a
    if mom_var <= 0 or not np.isfinite(mom_var):
        return 1e18
    
    vf = pos_var / norm2
    vF = mom_var / norm2
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)


def _bound_and_grad(params, M_0, M_pos, M_mom):
    """
    Compute bound and gradient analytically.
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
    """
    Main optimization routine. Uses Chebyshev collocation to build matrices,
    then optimizes the 3 free coefficients (a0, a4, a8) using SLSQP with analytic gradients.
    """
    N = 256
    L = 12.0
    
    try:
        M_0, M_pos, M_mom = _build_functional(N, L)
    except Exception:
        return np.array([-1.13, 0.04, 0.000308642])
    
    def _obj(params):
        val, _ = _bound_and_grad(params, M_0, M_pos, M_mom)
        return val
    
    def _obj_grad(params):
        val, grad = _bound_and_grad(params, M_0, M_pos, M_mom)
        return val, grad
    
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val = _obj(seed_params)
    
    # Generate diverse starting points
    starts = [seed_params]
    
    rng = np.random.default_rng(123)
    
    # Grid-based starts
    for a0 in [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
        for a4 in [-0.1, -0.05, 0.0, 0.02, 0.05, 0.1, 0.2]:
            for a8 in [-0.001, -0.0005, 0.0, 0.0001, 0.0005, 0.001, 0.002]:
                if time.time() > _DEADLINE - 60:
                    break
                starts.append(np.array([a0, a4, a8]))
        if time.time() > _DEADLINE - 60:
            break
    
    # Random starts
    for _ in range(50):
        if time.time() > _DEADLINE - 60:
            break
        a0 = rng.uniform(-5, 2)
        a4 = rng.uniform(-0.5, 0.5)
        a8 = rng.uniform(-0.01, 0.01)
        starts.append(np.array([a0, a4, a8]))
    
    # Deduplicate starts (roughly)
    unique_starts = []
    for s in starts:
        if not any(np.allclose(s, u, atol=1e-6) for u in unique_starts):
            unique_starts.append(s)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    # Evaluate all starts
    for start in unique_starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            val = _obj(start)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_params = start.copy()
        except Exception:
            pass
    
    # Local optimization from best starts
    top_starts = sorted(unique_starts, key=lambda s: _obj(s))[:20]
    
    for start in top_starts:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(_obj_grad, start, method='SLSQP',
                          jac=True, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish from the best
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_obj_grad, best_params, method='SLSQP',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None or not np.all(np.isfinite(best_params)):
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
