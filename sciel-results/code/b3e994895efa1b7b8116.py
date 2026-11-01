# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 175


def _phys_hermite(N, x):
    """Compute physicists' Hermite polynomials H_0(x), H_4(x), H_8(x), H_12(x)."""
    x = np.asarray(x, dtype=float)
    H0 = np.ones_like(x)
    H1 = 2.0 * x
    H2 = 4.0 * x**2 - 2.0
    H3 = 8.0 * x**3 - 12.0 * x
    H4 = 16.0 * x**4 - 48.0 * x**2 + 12.0
    H5 = 32.0 * x**5 - 160.0 * x**3 + 120.0 * x
    H6 = 64.0 * x**6 - 480.0 * x**4 + 720.0 * x**2 - 120.0
    H7 = 128.0 * x**7 - 1344.0 * x**5 + 3360.0 * x**3 - 1680.0 * x
    H8 = 256.0 * x**8 - 3584.0 * x**6 + 13440.0 * x**4 - 13440.0 * x**2 + 1680.0
    H9 = 512.0 * x**9 - 10752.0 * x**7 + 60480.0 * x**5 - 120960.0 * x**3 + 60480.0 * x
    H10 = 1024.0 * x**10 - 30720.0 * x**8 + 241920.0 * x**6 - 604800.0 * x**4 + 453600.0 * x**2 - 10080.0
    H11 = 2048.0 * x**11 - 81920.0 * x**9 + 887040.0 * x**7 - 3991680.0 * x**5 + 6652800.0 * x**3 - 3024000.0 * x
    H12 = 4096.0 * x**12 - 245760.0 * x**10 + 4128768.0 * x**8 - 30965760.0 * x**6 + 87264000.0 * x**4 - 87264000.0 * x**2 + 1088640.0
    return H0, H4, H8, H12


def _chebyshev_setup(N, L):
    """
    Set up Chebyshev collocation on [-L, L].
    Returns x (points), w (weights), D1 (first derivative matrix), D2 (second derivative matrix).
    """
    j = np.arange(N + 1)
    t = np.cos(np.pi * j / N)
    t = t[::-1]  # ascending
    x = L * t

    # Weights
    w = np.full(N + 1, np.pi / N)
    w[0] *= 0.5
    w[-1] *= 0.5
    w = L * w

    # Chebyshev differentiation matrix on [-1, 1]
    c = np.ones(N + 1)
    c[0] = 0.5
    c[-1] = 0.5
    c = c * (-1.0) ** np.arange(N + 1)

    X = np.tile(t, (N + 1, 1))
    dX = X - X.T
    dX[np.abs(dX) < 1e-15] = 1.0

    D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))

    D1 = D / L
    D2 = D1 @ D1

    return x, w, D1, D2


def _build_matrices(N, L):
    """
    Build the 4x4 matrices M_0, M_pos, M_mom for the basis {phi_0, phi_4, phi_8, phi_12}
    where phi_n(x) = H_n(x) * exp(-x^2/2).
    """
    x, w, D1, D2 = _chebyshev_setup(N, L)

    H0, H4, H8, H12 = _phys_hermite(12, x)
    exp_factor = np.exp(-x**2 / 2.0)

    phi0 = H0 * exp_factor
    phi4 = H4 * exp_factor
    phi8 = H8 * exp_factor
    phi12 = H12 * exp_factor

    phis = np.array([phi0, phi4, phi8, phi12])  # (4, N+1)

    M_0 = np.zeros((4, 4))
    M_pos = np.zeros((4, 4))
    M_mom = np.zeros((4, 4))

    x2 = x**2

    # Precompute D2 @ phi_j for each j
    D2_phis = np.zeros((4, N + 1))
    for j in range(4):
        D2_phis[j] = D2 @ phis[j]

    for i in range(4):
        for j in range(4):
            M_0[i, j] = np.sum(w * phis[i] * phis[j])
            M_pos[i, j] = np.sum(w * x2 * phis[i] * phis[j])
            M_mom[i, j] = -np.sum(w * phis[i] * D2_phis[j])

    return M_0, M_pos, M_mom


def _bound_and_grad(params, M_0, M_pos, M_mom):
    """
    Compute the uncertainty bound and its gradient w.r.t. (a0, a4, a8).
    a12 = -(a0 + 3*a4 + 105*a8)/945
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    a = np.array([a0, a4, a8, a12])

    N_val = a @ M_0 @ a
    P_val = a @ M_pos @ a
    Q_val = a @ M_mom @ a

    if N_val <= 1e-30 or P_val <= 0 or Q_val <= 0:
        if not (np.isfinite(N_val) and np.isfinite(P_val) and np.isfinite(Q_val)):
            return 1e18, np.array([1e18, 1e18, 1e18])
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(P_val * Q_val / (N_val * N_val))

    # da/dparams (4x3)
    da_dparams = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0]
    ])

    M0a = M_0 @ a
    Mposa = M_pos @ a
    Mmoma = M_mom @ a

    dN_dparams = 2.0 * (da_dparams @ M0a)
    dP_dparams = 2.0 * (da_dparams @ Mposa)
    dQ_dparams = 2.0 * (da_dparams @ Mmoma)

    dPQN2 = (dP_dparams * Q_val + P_val * dQ_dparams) / (N_val * N_val) \
            - 2.0 * P_val * Q_val * dN_dparams / (N_val * N_val * N_val)

    grad = dPQN2 / (2.0 * bound)

    if not np.all(np.isfinite(grad)):
        grad = np.array([1e18, 1e18, 1e18])

    return bound, grad


def _bound_only(params, M_0, M_pos, M_mom):
    val, _ = _bound_and_grad(params, M_0, M_pos, M_mom)
    return val


def construct():
    # Use higher resolution Chebyshev collocation
    N = 300
    L = 10.0

    try:
        M_0, M_pos, M_mom = _build_matrices(N, L)
    except Exception:
        return np.array([-1.13, 0.04, 0.000308642])

    def _obj(params):
        return _bound_only(params, M_0, M_pos, M_mom)

    def _obj_grad(params):
        return _bound_and_grad(params, M_0, M_pos, M_mom)

    # Diverse starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0001]),
        np.array([-1.2, 0.06, 0.0006]),
        np.array([-0.3, 0.01, 0.0001]),
        np.array([-2.5, 0.12, 0.0012]),
        np.array([0.5, -0.02, 0.0002]),
        np.array([-1.8, 0.09, 0.0009]),
    ]

    rng = np.random.default_rng(42)
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
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: SLSQP polish from best L-BFGS-B result
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_obj, best_params, method='SLSQP',
                          options={'maxiter': 10000, 'ftol': 1e-16, 'iprint': 0})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Final L-BFGS-B polish
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
