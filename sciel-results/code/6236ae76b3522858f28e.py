import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize

def _build_quadrature():
    mu, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    weights = np.outer(w, np.ones(n_phi)) * (2.0 * np.pi / n_phi)
    return mu, phi, weights


def _spherical_harmonics(mu, phi, l=10):
    n_mu, n_phi = len(mu), len(phi)
    Y = np.zeros((2 * l + 1, n_mu, n_phi), dtype=np.float64)
    for m in range(-l, l + 1):
        idx = m + l
        abs_m = abs(m)
        log_N = 0.5 * (np.log(2 * l + 1) - np.log(4.0 * np.pi)
                        + gammaln(l - abs_m + 1) - gammaln(l + abs_m + 1))
        N = np.exp(log_N)
        P = lpmv(abs_m, l, mu)
        if m == 0:
            Y[idx] = N * P[:, np.newaxis]
        elif m > 0:
            Y[idx] = np.sqrt(2.0) * N * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            Y[idx] = np.sqrt(2.0) * N * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    return Y


def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0

    l = 10
    mu, phi, weights = _build_quadrature()
    n_points = mu.size
    Y = _spherical_harmonics(mu, phi, l=l)
    Y_flat = Y.reshape(21, n_points).T
    weights_flat = weights.ravel()

    def objective_and_grad(c):
        f = Y_flat @ c
        f2 = f * f
        f3 = f2 * f
        f4 = f2 * f2
        A = np.dot(f4, weights_flat)
        B = np.dot(f2, weights_flat)
        if A <= 1e-30 or B <= 1e-30:
            return 1e10, np.zeros(21)
        R = (A ** 0.25) / (B ** 0.5)
        term1 = Y_flat.T @ (weights_flat * f3)
        term2 = Y_flat.T @ (weights_flat * f)
        grad = (A ** (-0.75)) * (B ** (-1.5)) * (B * term1 - A * term2)
        return R, grad

    best_c = None
    best_val = np.inf

    inits = [np.ones(21) / np.sqrt(21.0)]
    alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(21)])
    inits.append(alt / np.sqrt(21.0))
    c0 = np.zeros(21); c0[10] = 1.0; inits.append(c0)
    c10 = np.zeros(21); c10[0] = 1.0; c10[20] = 1.0
    inits.append(c10 / np.sqrt(2.0))
    for t in range(20):
        rng = np.random.RandomState(1000 + t)
        c = rng.randn(21)
        c /= np.linalg.norm(c)
        inits.append(c)

    for c0 in inits:
        if time.time() > deadline - 40.0:
            break
        try:
            result = minimize(objective_and_grad, c0, method='L-BFGS-B', jac=True,
                              options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass

    if best_c is not None and time.time() < deadline - 15.0:
        try:
            result = minimize(objective_and_grad, best_c, method='L-BFGS-B', jac=True,
                              options={'maxiter': 2000, 'ftol': 1e-17, 'gtol': 1e-15})
            if result.fun < best_val:
                best_c = result.x
                best_val = result.fun
        except Exception:
            pass

    if best_c is None:
        best_c = np.ones(21)
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        best_c = np.ones(21)
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1e-2 / max_abs)
    elif max_abs > 1e5:
        best_c = best_c * (1e4 / max_abs)
    return best_c.astype(np.float64)
