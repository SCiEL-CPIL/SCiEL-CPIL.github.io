# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import factorial, factorial2
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 600.0
_K = 10


def _w3j(l1, l2, l3, m1, m2, m3):
    """Wigner 3j symbol (l1 l2 l3; m1 m2 m3) via Racah formula."""
    if l1 < 0 or l2 < 0 or l3 < 0:
        return 0.0
    if m1 < -l1 or m1 > l1 or m2 < -l2 or m2 > l2 or m3 < -l3 or m3 > l3:
        return 0.0
    if m1 + m2 + m3 != 0:
        return 0.0
    if (l1 + l2 + l3) % 2 != 0:
        return 0.0
    if abs(l1 - l2) > l3 or abs(l2 - l3) > l1 or abs(l3 - l1) > l2:
        return 0.0
    a = l1 + l2 - l3
    b = l1 - l2 + l3
    c = -l1 + l2 + l3
    if a < 0 or b < 0 or c < 0:
        return 0.0
    s = (a + b + c) // 2
    sign = (-1.0) ** (s - m1)
    pref = np.sqrt(
        factorial(a) * factorial(b) * factorial(c)
        * factorial(l1 + m1) * factorial(l1 - m1)
        * factorial(l2 + m2) * factorial(l2 - m2)
        * factorial(l3 + m3) * factorial(l3 - m3)
    )
    pref *= np.sqrt(2 * l3 + 1)
    inner = 0.0
    for t in range(0, s + 1):
        denom = (factorial(t) * factorial(s - t) * factorial(a + m1 - t)
                 * factorial(b + m2 - t) * factorial(l1 - l2 - m3 + t)
                 * factorial(l2 - l1 + m3 + t))
        inner += (-1.0) ** t / denom
    return sign * pref * inner


def _build():
    k = _K
    ms = np.arange(-k, k + 1)
    n = len(ms)

    # Normalization N_m (real orthonormal)
    Nm = np.zeros(n)
    for i, m in enumerate(ms):
        am = abs(m)
        Nm[i] = np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                        factorial(k - am) / factorial(k + am))

    # Condon-Shortley associated Legendre P_k^m(x)
    P = np.zeros((n, n))
    for i, m in enumerate(ms):
        am = abs(m)
        for j, mj in enumerate(ms):
            x = mj
            if x > 1.0 or x < -1.0:
                P[i, j] = 0.0
                continue
            # Compute P_k^m(x)
            if am == 0:
                # Legendre P_k(x)
                if k == 0:
                    val = 1.0
                elif k == 1:
                    val = x
                else:
                    p0 = 1.0
                    p1 = x
                    for ell in range(1, k):
                        p0, p1 = p1, ((2 * ell + 1) * x * p1 - ell * p0) / (ell + 1)
                    val = p1
            else:
                # Associated Legendre via recurrence
                # P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^(m/2)
                if x == 1.0 or x == -1.0:
                    val = 0.0
                else:
                    sign = (-1.0) ** am
                    double_fact = factorial2(2 * am - 1)
                    val = sign * double_fact * (1.0 - x * x) ** (am / 2.0)
                    if k > am:
                        # P_{am+1}^am
                        p_prev = val
                        p_cur = (2 * am + 1) * x * val
                        for ell in range(am + 1, k):
                            p_next = ((2 * ell + 1) * x * p_cur - (ell + am) * (ell - am) * p_prev) / (ell + am + 1)
                            p_prev, p_cur = p_cur, p_next
                        val = p_cur
            P[i, j] = val

    # Angular integrals C[i,j,l] = ∫ Y_i Y_j Y_l dOmega (using Y = N P)
    C = np.zeros((n, n, n))
    for i in range(n):
        mi = ms[i]
        for j in range(n):
            mj = ms[j]
            if mi + mj > k or mi + mj < -k:
                continue
            ml = -(mi + mj)
            if ml < -k or ml > k:
                continue
            l = ml + k  # index
            w = _w3j(k, k, k, mi, mj, ml)
            if w == 0.0:
                continue
            C[i, j, l] = Nm[i] * Nm[j] * Nm[l] * w

    # Quadratic form Q[i,j] = ∫ Y_i Y_j dOmega = δ_ij (orthonormal)
    # L2 = c^T c (with this normalization)

    return ms, Nm, P, C


def _objective(c, C, N):
    """Return (R, grad) where R = ||f||_4/||f||_2."""
    L2 = float(np.dot(c, c))
    if L2 < 1e-30:
        return np.inf, np.zeros_like(c)
    L4_4 = float(np.einsum('i,j,l,ijl->', c, c, c, C))
    if L4_4 <= 0:
        L4_4 = 1e-30
    L4 = np.sqrt(L4_4)
    R = L4 / np.sqrt(L2)
    # dR/dc_i = (1/(2 L4 sqrt(L2))) * 4 C[i,j,l] c_j c_l * sqrt(L2) - (L4/(2 L2^(3/2))) c_i
    # = (2 C[i,j,l] c_j c_l)/L4 - L4 c_i / L2^(3/2)
    term = np.einsum('ijl,j,l->i', C, c, c)
    grad = (2.0 * term) / L4 - (L4 / (L2 ** 1.5)) * c
    return R, grad


def _make_starts(N):
    starts = []
    # all ones
    starts.append(np.ones(2 * _K + 1))
    # cos-only (m >= 0 nonzero, m<0 zero)
    v = np.zeros(2 * _K + 1)
    v[10:] = 1.0
    starts.append(v)
    # sin-only
    v = np.zeros(2 * _K + 1)
    v[:10] = 1.0
    starts.append(v)
    # m=0 dominant
    v = np.zeros(2 * _K + 1)
    v[10] = 1.0
    starts.append(v)
    # symmetric cos with alternating sign
    v = np.zeros(2 * _K + 1)
    for m in range(0, 11):
        v[10 + m] = (-1.0) ** m
    starts.append(v)
    # random starts
    rng = np.random.default_rng(0)
    for _ in range(6):
        starts.append(rng.standard_normal(2 * _K + 1))
    return starts


def _refine(c0, C, N, deadline):
    best_c = c0.copy()
    best_R, _ = _objective(c0, C, N)
    for method in ['L-BFGS-B']:
        try:
            res = minimize(
                _objective, c0, jac=True, method=method,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12,
                         'disp': False}
            )
            if res.fun < best_R:
                best_R = res.fun
                best_c = res.x
        except Exception:
            pass
    # second pass from refined
    try:
        res = minimize(
            _objective, best_c, jac=True, method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14,
                     'disp': False}
        )
        if res.fun < best_R:
            best_R = res.fun
            best_c = res.x
    except Exception:
        pass
    return best_c, best_R


def rudin_problem10() -> np.ndarray:
    t0 = time.time()
    ms, Nm, P, C = _build()
    N = 2 * _K + 1

    starts = _make_starts(Nm)
    best_c = starts[0].copy()
    best_R, _ = _objective(best_c, C, N)

    for c0 in starts:
        if time.time() > _DEADLINE - 120:
            break
        c, R = _refine(c0, C, N, _DEADLINE)
        if R < best_R:
            best_R = R
            best_c = c

    # global differential evolution if time remains
    if time.time() < _DEADLINE - 180:
        try:
            def de_obj(x):
                r, _ = _objective(x, C, N)
                return r
            bounds = [(-10.0, 10.0)] * N
            de = differential_evolution(
                de_obj, bounds, seed=42, maxiter=300, tol=1e-10,
                popsize=12, polish=True, init='sobol',
                updating='deferred', x0=None
            )
            if de.fun < best_R:
                best_R = de.fun
                best_c = de.x
        except Exception:
            pass

    # final polish
    if time.time() < _DEADLINE - 30:
        try:
            res = minimize(
                _objective, best_c, jac=True, method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14,
                         'disp': False}
            )
            if res.fun < best_R:
                best_R = res.fun
                best_c = res.x
        except Exception:
            pass

    # normalize scale is irrelevant to R; ensure validity (max|c| in range)
    c = np.asarray(best_c, dtype=np.float64)
    mx = np.max(np.abs(c))
    if not np.isfinite(mx) or mx < 1e-3:
        c = np.ones(N, dtype=np.float64)
    elif mx > 1e5:
        c = c * (1e3 / mx)
    return c


# EVOLVE-BLOCK-END
