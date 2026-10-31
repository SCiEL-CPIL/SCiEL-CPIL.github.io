"""Uncertainty bound minimization: fast quadratic-form evaluation + multi-start polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from math import factorial, sqrt, pi

_DEADLINE = time.time() + 195.0  # 15s safety margin before 210s kill


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


DEGREES = [0, 4, 8, 12]
N_DEG = 4

H_POLYS = []
for d in DEGREES:
    hc = _hermite_coeffs(d)
    vec = np.zeros(d + 1)
    for p, c in hc.items():
        vec[p] = c
    H_POLYS.append(vec)

# M0 = <P,P>, M2 = <x^2 P,P>, L = <Q,Q>  (Q_k = (-1)^k P_k)
M0_mat = np.zeros((N_DEG, N_DEG))
for i, d in enumerate(DEGREES):
    M0_mat[i, i] = (2 ** d) * factorial(d)


def _quad_even(ci, cj, shift=0):
    prod = np.convolve(ci, cj)
    prod_shifted = np.zeros(len(prod) + shift)
    prod_shifted[shift:] = prod
    val = 0.0
    for k in range(0, len(prod_shifted), 2):
        m = k // 2
        val += prod_shifted[k] * _int_x2m(m)
    return val


M2_mat = np.zeros((N_DEG, N_DEG))
for i in range(N_DEG):
    for j in range(N_DEG):
        M2_mat[i, j] = _quad_even(H_POLYS[i], H_POLYS[j], shift=2)

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
        L_mat[i, j] = _quad_even(Q_POLYS[i], Q_POLYS[j])

# a12 = -(a0 + 12*a4 + 2520*a8)/27720
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
    """Exact bound and analytic gradient w.r.t. a3 = [a0, a4, a8]."""
    M0 = float(a3 @ M0_eff @ a3)
    M2 = float(a3 @ M2_eff @ a3)
    L = float(a3 @ L_eff @ a3)

    if M0 <= 1e-12 or L <= 1e-12 or M2 <= 1e-12:
        return np.inf, np.zeros(3)

    bound = np.sqrt(M2 * L) / M0

    dM0 = 2.0 * (M0_eff @ a3)
    dM2 = 2.0 * (M2_eff @ a3)
    dL = 2.0 * (L_eff @ a3)

    term1 = (L * dM2 + M2 * dL) / (2.0 * M0 * np.sqrt(M2 * L))
    term2 = bound * dM0 / (M0 * M0)

    grad = term1 - term2
    return bound, grad


def _objective(x):
    b, _ = _bound_and_grad(x)
    if not np.isfinite(b):
        return 1e10
    return b


def _objective_grad(x):
    return _bound_and_grad(x)


def construct():
    start_time = time.time()
    best_x = np.array([-1.13, 0.04, 0.000308642])
    best_bound = _objective(best_x)

    rng = np.random.default_rng(42)

    # ---- Phase 1: broad random sampling + structured grid ----
    starts = [best_x.copy()]
    for _ in range(25):
        starts.append(best_x + rng.normal(0, 0.15, 3))
    for _ in range(60):
        starts.append(np.array([rng.uniform(-3.5, 2.5), rng.uniform(-0.6, 0.6), rng.uniform(-0.015, 0.015)]))
    for a0 in np.linspace(-3.0, 1.5, 12):
        for a4 in np.linspace(-0.4, 0.4, 8):
            starts.append(np.array([a0, a4, 0.0008 * rng.normal()]))

    scored = []
    for s in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            v = _objective(s)
            if np.isfinite(v):
                scored.append((v, s.copy()))
                if v < best_bound:
                    best_bound = v
                    best_x = s.copy()
        except Exception:
            pass

    scored.sort(key=lambda t: t[0])

    # ---- Phase 2: L-BFGS-B from the top-N + best ----
    top = [s for _, s in scored[:15]]
    top.append(best_x.copy())
    for s in top:
        if time.time() > _DEADLINE - 50:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B', jac=True,
                           options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # ---- Phase 3: differential evolution global search ----
    if time.time() < _DEADLINE - 40:
        try:
            def _time_check(x):
                return time.time() - start_time > _DEADLINE - 45
            de_res = differential_evolution(_objective, bounds=[(-4.0, 4.0), (-1.5, 1.5), (-0.02, 0.02)],
                                            maxiter=200, popsize=25, tol=1e-12, seed=42,
                                            polish=False, init='sobol', callback=_time_check)
            if np.isfinite(de_res.fun) and de_res.fun < best_bound:
                best_bound = de_res.fun
                best_x = de_res.x.copy()
        except Exception:
            pass

    # ---- Phase 4: Nelder-Mead high-precision polish (epigraph safeguard) ----
    if time.time() < _DEADLINE - 10 and best_x is not None:
        # 4a: direct smooth polish
        try:
            nm = minimize(_objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 200000, 'xatol': 1e-16, 'fatol': 1e-18})
            if np.isfinite(nm.fun) and nm.fun < best_bound:
                best_bound = nm.fun
                best_x = nm.x.copy()
        except Exception:
            pass
        # 4b: t-epigraph (soft-min) safeguard against any non-smooth boundary
        if time.time() < _DEADLINE - 5:
            try:
                def _epi(z):
                    a3 = z[:3]
                    t = z[3]
                    b, _ = _bound_and_grad(a3)
                    if not np.isfinite(b):
                        return 1e10
                    return max(b, t)
                z0 = np.concatenate([best_x, [best_bound]])
                e = minimize(_epi, z0, method='Nelder-Mead',
                             options={'maxiter': 100000, 'xatol': 1e-15, 'fatol': 1e-18})
                if e.x is not None and e.x[3] < best_bound and np.isfinite(e.x[3]):
                    # refine the a3 part
                    b2, _ = _bound_and_grad(e.x[:3])
                    if np.isfinite(b2) and b2 < best_bound:
                        best_bound = b2
                        best_x = e.x[:3].copy()
            except Exception:
                pass

    if best_x is None or not np.all(np.isfinite(best_x)):
        return np.array([-1.13, 0.04, 0.000308642])
    return best_x


# EVOLVE-BLOCK-END
