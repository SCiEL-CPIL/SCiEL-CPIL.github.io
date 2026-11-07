"""Optimize the uncertainty-inequality constant via continuous optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from scipy.special import eval_hermite
from scipy.optimize import differential_evolution, minimize, brentq


# ---- Precompute Hermite polynomial coefficients (physicists' convention) ----
# H_n(x) has integer coefficients; compute via recurrence with Fractions.
def _hermite_coeffs(n):
    """Return list of Fractions for H_n(x) = sum a_k x^k (k=0..n)."""
    if n == 0:
        return [Fraction(1)]
    if n == 1:
        return [Fraction(0), Fraction(2)]
    # H_n = 2x H_{n-1} - 2(n-1) H_{n-2}
    c1 = _hermite_coeffs(n - 1)
    c2 = _hermite_coeffs(n - 2)
    # 2x * c1
    a = [Fraction(0)] + [Fraction(2) * v for v in c1]
    # -2(n-1) * c2
    b = [Fraction(-2 * (n - 1)) * v for v in c2]
    m = max(len(a), len(b))
    res = [Fraction(0)] * m
    for i, v in enumerate(a):
        res[i] += v
    for i, v in enumerate(b):
        res[i] += v
    # trim trailing zeros
    while len(res) > 1 and res[-1] == 0:
        res.pop()
    return res


_HC = {n: _hermite_coeffs(n) for n in (0, 4, 8, 12)}
# Convert to float arrays for fast evaluation (high precision)
_HF = {n: np.array([float(v) for v in c], dtype=np.float64) for n, c in _HC.items()}
# H12(0)
_H12_0 = float(_HC[12][0])
_H0_0 = float(_HC[0][0])
_H4_0 = float(_HC[4][0])
_H8_0 = float(_HC[8][0])


def _build_q_coeffs(c1, c2):
    """
    Given c1, c2 (with c0=1), return the float coefficient array of q(x) = g(x)/x^2,
    ordered as [q_0, q_1, ..., q_d] so that q(x) = sum q_k x^k.
    g = H0 + c1 H4 + c2 H8 + c3 H12, c3 = -(H0_0 + c1 H4_0 + c2 H8_0)/H12_0.
    """
    c3 = -(_H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    # g coefficients (ascending)
    g = (1.0 * _HF[0] + c1 * _HF[4] + c2 * _HF[8] + c3 * _HF[12])
    # g must have zero constant term (by construction) and be divisible by x^2
    # q_k = g_{k+2}
    q = g[2:]
    # trim leading (high-degree) zeros
    while len(q) > 1 and q[-1] == 0.0:
        q = q[:-1]
    return q


def _largest_sign_change_root(q):
    """
    Find the largest real root of q at which q changes sign.
    q is a numpy array of coefficients ascending. Returns float or None.
    """
    if len(q) < 2:
        return None
    # Use numpy roots (descending order)
    try:
        roots = np.roots(q)
    except Exception:
        return None
    # Filter real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) <= 1e-9 * max(1.0, abs(r.real)):
            real_roots.append(float(r.real))
    if not real_roots:
        return None
    real_roots.sort(reverse=True)
    # Evaluate q at each candidate; find sign change
    # q(x) via Horner
    def qval(x):
        v = 0.0
        for c in q[::-1]:
            v = v * x + c
        return v
    # Check each real root from largest down; verify sign change by sampling around it
    for r in real_roots:
        h = 1e-6 * max(1.0, abs(r))
        left = qval(r - h)
        right = qval(r + h)
        if left == 0.0 or right == 0.0:
            # Degenerate; try smaller h
            h2 = 1e-9 * max(1.0, abs(r))
            left = qval(r - h2)
            right = qval(r + h2)
        if left * right < 0:
            # Refine with brentq on a small interval
            try:
                lo = r - h
                hi = r + h
                flo = qval(lo)
                fhi = qval(hi)
                if flo * fhi < 0:
                    r_refined = brentq(qval, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=100)
                    return r_refined
            except Exception:
                pass
            return r
    return None


def _value_from(c1, c2):
    """Compute the objective value = r^2/(2*pi) for coefficients (c0=1, c1, c2)."""
    try:
        q = _build_q_coeffs(c1, c2)
        r = _largest_sign_change_root(q)
        if r is None:
            return np.inf
        return r * r / (2.0 * np.pi)
    except Exception:
        return np.inf


def _objective(params):
    """Vectorized-safe objective for scipy (params = [c1, c2])."""
    c1, c2 = params
    return _value_from(float(c1), float(c2))


def construct():
    """Return three coefficients [c0, c1, c2] minimizing r^2/(2*pi)."""
    start = time.time()
    DEADLINE = 180.0  # safety margin before the 210s kill
    MARGIN = 25.0     # keep 25s of slack

    # Seed point: c0=-113/100, c1=1/25, c2=1/3240.
    # Rescale to c0=1: divide everything by c0.
    seed_c0 = -113.0 / 100.0
    seed_c1 = 1.0 / 25.0
    seed_c2 = 1.0 / 3240.0
    seed_c1r = seed_c1 / seed_c0   # = -25/113
    seed_c2r = seed_c2 / seed_c0   # = -100/(113*3240)

    best_params = np.array([seed_c1r, seed_c2r])
    best_val = _value_from(seed_c1r, seed_c2r)

    def time_left():
        return DEADLINE - MARGIN - (time.time() - start)

    # ---- Global search: differential evolution over a reasonable box ----
    # The seed has c1 ~ -0.22, c2 ~ -0.000275. Explore a wide box.
    bounds = [(-5.0, 5.0), (-0.05, 0.05)]
    try:
        if time_left() > 30:
            res_de = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.7,
                seed=42,
                polish=True,
                init='sobol',
                workers=-1,
            )
            if res_de is not None and np.isfinite(res_de.fun):
                if res_de.fun < best_val:
                    best_val = float(res_de.fun)
                    best_params = np.array(res_de.x, dtype=np.float64)
    except Exception:
        pass

    # ---- Local polish from the best point ----
    try:
        if time_left() > 10:
            res_l = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12},
            )
            if res_l is not None and np.isfinite(res_l.fun):
                if res_l.fun < best_val:
                    best_val = float(res_l.fun)
                    best_params = np.array(res_l.x, dtype=np.float64)
    except Exception:
        pass

    # ---- Final high-precision polish with BFGS using finite differences ----
    try:
        if time_left() > 5:
            res_b = minimize(
                _objective,
                best_params,
                method='BFGS',
                options={'maxiter': 2000, 'gtol': 1e-10},
            )
            if res_b is not None and np.isfinite(res_b.fun):
                if res_b.fun < best_val:
                    best_val = float(res_b.fun)
                    best_params = np.array(res_b.x, dtype=np.float64)
    except Exception:
        pass

    # ---- Convert back to (c0, c1, c2) with c0 = 1 ----
    # The evaluator is scale-invariant in value (r scales with 1/sqrt(|c0|) ... actually
    # value = r^2/(2*pi) is NOT scale invariant in general because c3 depends on c0.
    # But the problem says "choose three finite coefficients c0,c1,c2" - the value depends
    # on the actual polynomial, not its scale. So we must return a valid triple.
    # Returning c0=1 is valid. The evaluator will compute c3 from our c0,c1,c2.
    c0_out = 1.0
    c1_out = float(best_params[0])
    c2_out = float(best_params[1])

    # Sanity: if best_val is inf (no valid root found), fall back to seed.
    if not np.isfinite(best_val):
        return np.array([seed_c0, seed_c1, seed_c2])

    return np.array([c0_out, c1_out, c2_out])
# EVOLVE-BLOCK-END
