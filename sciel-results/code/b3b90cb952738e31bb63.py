"""Global-search uncertainty bound via differential evolution + log-space polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from math import factorial, sqrt, pi

_DEADLINE = 185.0

# ---------- Hermite helpers ----------
def _hermite_coeffs(n):
    if n == 0:
        return {0: 1.0}
    if n == 1:
        return {1: 2.0}
    h_prev = {0: 1.0}
    h_curr = {1: 2.0}
    for m in range(2, n + 1):
        h_new = {}
        for p, c in h_curr.items():
            h_new[p + 1] = h_new.get(p + 1, 0.0) + 2.0 * c
        for p, c in h_prev.items():
            h_new[p] = h_new.get(p, 0.0) - 2.0 * (m - 1) * c
        h_new = {p: c for p, c in h_new.items() if abs(c) > 1e-15}
        h_prev, h_curr = h_curr, h_new
    return h_curr

def _int_x2m(m):
    if m == 0:
        return sqrt(pi)
    val = sqrt(pi)
    for j in range(1, m + 1):
        val *= (2 * j - 1) / 2.0
    return val

# ---------- Precompute matrices (same quadratic form as before) ----------
DEGREES = [0, 4, 8, 12]
N_DEG = 4

H_POLYS = []
for d in DEGREES:
    hc = _hermite_coeffs(d)
    vec = np.zeros(d + 1)
    for p, c in hc.items():
        vec[p] = c
    H_POLYS.append(vec)

M0_mat = np.zeros((N_DEG, N_DEG))
for i, d in enumerate(DEGREES):
    M0_mat[i, i] = (2 ** d) * factorial(d)

M2_mat = np.zeros((N_DEG, N_DEG))
for i in range(N_DEG):
    for j in range(N_DEG):
        ci, cj = H_POLYS[i], H_POLYS[j]
        prod = np.convolve(ci, cj)
        prod_x2 = np.zeros(len(prod) + 2)
        prod_x2[2:] = prod
        val = 0.0
        for k in range(0, len(prod_x2), 2):
            val += prod_x2[k] * _int_x2m(k // 2)
        M2_mat[i, j] = val

Q_POLYS = []
for i, d in enumerate(DEGREES):
    q_vec = np.zeros(max(d + 1, 1))
    if d == 0:
        q_vec = np.zeros(2)
        q_vec[1] = -1.0
    else:
        h_minus = _hermite_coeffs(d - 1)
        h_plus = _hermite_coeffs(d + 1)
        q_vec = np.zeros(d + 2)
        for p, c in h_minus.items():
            q_vec[p] += d * c
        for p, c in h_plus.items():
            q_vec[p] -= 0.5 * c
    Q_POLYS.append(q_vec)

L_mat = np.zeros((N_DEG, N_DEG))
for i in range(N_DEG):
    for j in range(N_DEG):
        ci, cj = Q_POLYS[i], Q_POLYS[j]
        prod = np.convolve(ci, cj)
        val = 0.0
        for k in range(0, len(prod), 2):
            val += prod[k] * _int_x2m(k // 2)
        L_mat[i, j] = val

# B maps a3=[a0,a4,a8] -> a4=[a0,a4,a8,a12], with a12 fixed so g(0)=0
B = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0 / 27720.0, -12.0 / 27720.0, -2520.0 / 27720.0]
])

M0_eff = B.T @ M0_mat @ B
M2_eff = B.T @ M2_mat @ B
L_eff = B.T @ L_mat @ B


def _bound_and_grad(a3):
    """Return (bound, grad) of sqrt(M2*L)/M0 w.r.t a3, or (inf, 0) if invalid."""
    M0 = a3 @ M0_eff @ a3
    M2 = a3 @ M2_eff @ a3
    L = a3 @ L_eff @ a3
    if M0 <= 1e-12 or M2 <= 1e-12 or L <= 1e-12:
        return np.inf, np.zeros(3)
    bound = np.sqrt(M2 * L) / M0
    dM0 = 2.0 * (M0_eff @ a3)
    dM2 = 2.0 * (M2_eff @ a3)
    dL = 2.0 * (L_eff @ a3)
    s = np.sqrt(M2 * L)
    term1 = (L * dM2 + M2 * dL) / (2.0 * M0 * s)
    term2 = bound * dM0 / (M0 * M0)
    return bound, term1 - term2


def _log_obj(a3):
    b, _ = _bound_and_grad(a3)
    return np.log(b) if b < np.inf else 1e6


# ---------- Construct: global DE search + local polish ----------
def construct():
    start_time = time.time()

    # Broad, well-scaled box for the global search.
    bounds = [(-3.0, 3.0), (-0.5, 0.5), (-0.05, 0.05)]

    # A generous seed point (also a fallback).
    best_x = np.array([-1.13, 0.04, 0.0003])
    best_b, _ = _bound_and_grad(best_x)

    def _update(x, b):
        nonlocal best_x, best_b
        if b < best_b:
            best_b = b
            best_x = x.copy()

    # Global differential evolution (log objective for conditioning).
    if time.time() - start_time < _DEADLINE - 30.0:
        try:
            de = differential_evolution(
                _log_obj,
                bounds=bounds,
                seed=12345,
                maxiter=200,
                popsize=15,
                tol=1e-9,
                polish=False,
                init='sobol',
                updating='immediate',
                callback=None,
            )
            if de.x is not None and de.fun < 1e6:
                b, _ = _bound_and_grad(de.x)
                _update(de.x, b)
        except Exception:
            pass

    # Local polish from several diverse starts (DE result + a spread of seeds).
    if time.time() - start_time < _DEADLINE - 15.0:
        starts = [best_x.copy()]
        for s in [np.array([-1.13, 0.04, 0.0003]),
                  np.array([-1.0, 0.0, 0.0]),
                  np.array([-0.5, -0.5, 0.1]),
                  np.array([-2.0, 0.1, -0.01]),
                  np.array([-0.3, 0.2, -0.02])]:
            starts.append(s)

        for x0 in starts:
            if time.time() - start_time > _DEADLINE - 8.0:
                break
            try:
                res = minimize(_log_obj, x0, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-10})
                if res.fun < 1e6:
                    b, _ = _bound_and_grad(res.x)
                    _update(res.x, b)
            except Exception:
                pass

    # Final high-precision polish on the best point.
    if time.time() - start_time < _DEADLINE - 4.0:
        try:
            res = minimize(_log_obj, best_x, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12})
            if res.fun < 1e6:
                b, _ = _bound_and_grad(res.x)
                _update(res.x, b)
        except Exception:
            pass

    return best_x

# EVOLVE-BLOCK-END
