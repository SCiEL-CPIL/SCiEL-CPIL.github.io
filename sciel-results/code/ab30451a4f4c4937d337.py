import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_T0 = time.time()
_HARD_LIMIT = 205.0   # safety margin before the 210s kill

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(26)
I[0] = np.sqrt(np.pi)
for n in range(1, 26):
    I[n] = I[n - 1] * (2.0 * n - 1.0) / (2.0 * n)

# Hermite polynomial coefficients in power basis x^{2k}
H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

# a12 = -(a0 + 3*a4 + 105*a8) / 945  =>  g(0)=0
c12 = np.array([-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0])
col0 = H0_c + c12[0] * H12_c
col1 = H4_c + c12[1] * H12_c
col2 = H8_c + c12[2] * H12_c
A_mat = np.column_stack((col0, col1, col2))
ALTN = np.array([1, -1, 1, -1, 1, -1, 1], dtype=float)

# Precompute Gram matrices so the objective is a fast quadratic form.
# Mx0 = p^T Gx0 p, Mx1 = p^T Gx1 p, Mxi0 = q^T Gxi0 q, Mxi1 = q^T Gxi1 q
# where q = p * ALTN.
Gx0 = np.zeros((7, 7))
Gx1 = np.zeros((7, 7))
Gxi0 = np.zeros((7, 7))
Gxi1 = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        Gx0[i, j] = I[i + j]
        Gx1[i, j] = I[i + j + 1]
        Gxi0[i, j] = ALTN[i] * ALTN[j] * I[i + j]
        Gxi1[i, j] = ALTN[i] * ALTN[j] * I[i + j + 1]
# Precompute dM contributions: dM0_m = 2 * A[:,m]^T Gx0 p, etc.
dGx0 = 2.0 * A_mat.T @ Gx0
dGx1 = 2.0 * A_mat.T @ Gx1
dGxi0 = 2.0 * A_mat.T @ Gxi0
dGxi1 = 2.0 * A_mat.T @ Gxi1


def _p_of(a):
    return A_mat @ a


def _obj_and_grad(a):
    p = A_mat @ a
    Mx0 = p @ Gx0 @ p
    Mx1 = p @ Gx1 @ p
    Mxi0 = p @ Gxi0 @ p
    Mxi1 = p @ Gxi1 @ p
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18, np.zeros(3)
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
    prod = vx * vxi
    bound = np.sqrt(prod)
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
    # gradients
    dMx0 = dGx0 @ p
    dMx1 = dGx1 @ p
    dMxi0 = dGxi0 @ p
    dMxi1 = dGxi1 @ p
    dvx = (dMx1 * Mx0 - Mx1 * dMx0) / (Mx0 * Mx0)
    dvxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) / (Mxi0 * Mxi0)
    factor = 0.5 / bound
    grad = factor * (vxi * dvx + vx * dvxi)
    if not np.all(np.isfinite(grad)):
        grad = np.zeros(3)
    return bound, grad


def _obj_only(a):
    return _obj_and_grad(a)[0]


def _polish(a, budget):
    """L-BFGS-B polish with deadline guard. Returns best (val, params)."""
    val0, _ = _obj_and_grad(a)
    if not np.isfinite(val0):
        return 1e18, a.copy()
    best_val = val0
    best_x = a.copy()
    try:
        res = minimize(_obj_only, a, jac=lambda x: _obj_and_grad(x)[1],
                       method='L-BFGS-B',
                       options={'gtol': 1e-12, 'maxiter': 400, 'maxfun': 4000})
        if np.isfinite(res.fun):
            res_val, res_grad = _obj_and_grad(res.x)
            if np.isfinite(res_val) and res_val < best_val:
                best_val = res_val
                best_x = res.x.copy()
    except Exception:
        pass
    return best_val, best_x


def construct():
    global _T0
    # baseline
    seed = np.array([-1.13, 0.04, 0.000308642])
    best_val, best_x = _polish(seed, 5.0)
    if not np.all(np.isfinite(best_x)):
        best_x = seed.copy()
        best_val = _obj_only(best_x)

    def time_left():
        return _HARD_LIMIT - (time.time() - _T0)

    # ---- Global exploration: differential evolution over a wide box ----
    if time_left() > 40:
        try:
            res_de = differential_evolution(
                _obj_only,
                bounds=[(-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)],
                seed=7, popsize=18, maxiter=12, tol=1e-8,
                polish=False, init='sobol',
                updating='deferred', workers=1,
                mutation=(0.3, 0.9), recombination=0.7,
            )
            if res_de.x is not None and np.all(np.isfinite(res_de.x)):
                v, x = _polish(res_de.x, 5.0)
                if np.isfinite(v) and v < best_val:
                    best_val, best_x = v, x
        except Exception:
            pass

    # ---- Multi-start L-BFGS-B from diverse points ----
    rng = np.random.default_rng(12345)
    starts = [best_x.copy(), seed.copy()]
    for r in [(0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (4.0, 4.0, 4.0)]:
        for s in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
                  (-1, 1, -1), (-1, -1, 1), (1, -1, -1), (-1, -1, -1)]:
            starts.append(np.array([r[0] * s[0], r[1] * s[1], r[2] * s[2]]))
    for _ in range(60):
        starts.append(rng.uniform(-5.0, 5.0, size=3))

    for st in starts:
        if time_left() < 8:
            break
        v, x = _polish(st, 5.0)
        if np.isfinite(v) and v < best_val:
            best_val, best_x = v, x

    # ---- High-precision final polish pass (tighter tolerance, more iters) ----
    if time_left() > 6:
        try:
            res = minimize(_obj_only, best_x, jac=lambda x: _obj_and_grad(x)[1],
                           method='L-BFGS-B',
                           options={'gtol': 1e-14, 'maxiter': 2000, 'maxfun': 20000})
            if np.isfinite(res.fun):
                res_val, _ = _obj_and_grad(res.x)
                if np.isfinite(res_val) and res_val < best_val:
                    best_val, best_x = res_val, res.x.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_x)):
        return np.array([-1.13, 0.04, 0.000308642])
    return np.array(best_x)
