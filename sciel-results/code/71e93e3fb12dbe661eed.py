import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _get_hermite_coeffs(n):
    """Get coefficients of physicists' Hermite polynomial Hn(x) as list [c0, c1, ..., cn]."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    H = [[1.0], [0.0, 2.0]]
    for k in range(2, n + 1):
        prev1 = H[k - 1]
        prev2 = H[k - 2]
        curr = [0.0] * (k + 1)
        for i in range(k):
            curr[i + 1] += 2.0 * prev1[i]
        for i in range(k - 1):
            curr[i] += -2.0 * (k - 1) * prev2[i]
        H.append(curr)
    return H[n]

_H0 = _get_hermite_coeffs(0)
_H4 = _get_hermite_coeffs(4)
_H8 = _get_hermite_coeffs(8)
_H12 = _get_hermite_coeffs(12)
_H0_0 = _H0[0]
_H4_0 = _H4[0]
_H8_0 = _H8[0]
_H12_0 = _H12[0]

def _build_g(c0, c1, c2):
    """Build g coefficients and return them (with sign fix)."""
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g = [0.0] * 13
    for i in range(len(_H0)):
        g[i] += c0 * _H0[i]
    for i in range(len(_H4)):
        g[i] += c1 * _H4[i]
    for i in range(len(_H8)):
        g[i] += c2 * _H8[i]
    for i in range(len(_H12)):
        g[i] += c3 * _H12[i]
    if g[12] < 0:
        g = [-c for c in g]
    return g

def _get_q_coeffs(g):
    """Return q = g/x^2 coefficients, or None if not divisible."""
    if abs(g[0]) > 1e-14 or abs(g[1]) > 1e-14:
        return None
    return g[2:]

def _find_largest_sign_changing_root(q_coeffs):
    """Find largest positive real root where q changes sign."""
    if q_coeffs is None or len(q_coeffs) == 0:
        return None
    coeffs_desc = list(reversed(q_coeffs))
    # Remove leading zeros
    while len(coeffs_desc) > 1 and abs(coeffs_desc[0]) < 1e-15:
        coeffs_desc = coeffs_desc[1:]
    if len(coeffs_desc) == 0:
        return None
    try:
        roots = np.roots(coeffs_desc)
    except:
        return None
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
    if not real_roots:
        return None
    poly = np.poly1d(coeffs_desc)
    largest = None
    for r in real_roots:
        eps = 1e-7 * max(1.0, abs(r))
        val_left = poly(r - eps)
        val_right = poly(r + eps)
        if val_left * val_right < 0:
            if largest is None or r > largest:
                largest = r
    return largest

def _compute_value(c0, c1, c2):
    """Compute the exact value for given coefficients."""
    g = _build_g(c0, c1, c2)
    q = _get_q_coeffs(g)
    if q is None:
        return float('inf')
    r = _find_largest_sign_changing_root(q)
    if r is None:
        return float('inf')
    return r ** 2 / (2 * np.pi)

def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate: for each positive real root of q, compute a soft-max
    over roots that are sign-changing. We use log-sum-exp over log(r^2) 
    weighted by whether the root is sign-changing (using a smooth indicator).
    """
    try:
        g = _build_g(x[0], x[1], x[2])
        q = _get_q_coeffs(g)
        if q is None:
            return 1e10
        coeffs_desc = list(reversed(q))
        while len(coeffs_desc) > 1 and abs(coeffs_desc[0]) < 1e-15:
            coeffs_desc = coeffs_desc[1:]
        if len(coeffs_desc) == 0:
            return 1e10
        try:
            roots = np.roots(coeffs_desc)
        except:
            return 1e10
        poly = np.poly1d(coeffs_desc)
        # Collect positive real roots
        candidates = []
        for r in roots:
            if abs(r.imag) < 1e-8 and r.real > 0:
                candidates.append(r.real)
        if not candidates:
            return 1e10
        # For smooth surrogate, compute soft-max over r^2/(2*pi) for all
        # positive real roots, but weight by sign-change indicator.
        # Use a smooth approximation: for each root, compute sign change score
        scores = []
        for r in candidates:
            eps = 1e-6 * max(1.0, abs(r))
            vl = poly(r - eps)
            vr = poly(r + eps)
            # Smooth sign-change indicator: 1 if vl*vr < 0, 0 otherwise
            # Use sigmoid: 1/(1+exp(k*vl*vr)) where k is large
            prod = vl * vr
            k = 50.0
            if prod > 0:
                indicator = 1.0 / (1.0 + np.exp(k * min(prod, 50.0)))
            else:
                indicator = 1.0 / (1.0 + np.exp(-k * min(-prod, 50.0)))
            if indicator > 0.5:
                scores.append(np.log(r ** 2 / (2 * np.pi) + 1e-30))
        if not scores:
            return 1e10
        # Log-sum-exp for soft max
        m = max(scores)
        lse = m + np.log(sum(np.exp(s - m) for s in scores))
        return np.exp(lse)
    except:
        return 1e10

def _objective_exact(x):
    """Exact nonsmooth objective."""
    try:
        return _compute_value(x[0], x[1], x[2])
    except:
        return 1e10

def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s

    # Seed
    seed_c = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240.0])
    best_c = seed_c.copy()
    best_val = _compute_value(best_c[0], best_c[1], best_c[2])

    def elapsed():
        return time.time() - start_time

    # Phase 1: Broad exploration with differential evolution on smooth surrogate
    if elapsed() < deadline - 70:
        try:
            bounds = [(-3, 3), (-3, 3), (-3, 3)]
            de_result = differential_evolution(
                lambda x: _smooth_objective(x), bounds,
                maxiter=80, popsize=20, tol=1e-6, seed=42, polish=False
            )
            if de_result.fun < best_val * 1.5:
                v = _compute_value(de_result.x[0], de_result.x[1], de_result.x[2])
                if v < best_val:
                    best_val = v
                    best_c = de_result.x.copy()
        except:
            pass

    # Phase 2: Multi-start local optimization
    np.random.seed(123)
    num_starts = 40
    for i in range(num_starts):
        if elapsed() > deadline - 30:
            break

        if i == 0:
            x0 = best_c.copy()
        elif i % 4 == 1:
            x0 = best_c + np.random.normal(0, 0.3, size=3)
        elif i % 4 == 2:
            x0 = best_c + np.random.normal(0, 0.1, size=3)
        elif i % 4 == 3:
            x0 = np.random.uniform(-2, 2, size=3)
        else:
            x0 = best_c + np.random.normal(0, 0.05, size=3)

        # First try smooth surrogate for a warm start
        try:
            result_smooth = minimize(
                lambda x: _smooth_objective(x), x0, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            x0 = result_smooth.x
        except:
            pass

        # Then refine with exact objective
        try:
            result = minimize(
                _objective_exact, x0, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except:
            pass

        # Also try COBYLA which handles nonsmooth better
        if elapsed() < deadline - 30:
            try:
                result2 = minimize(
                    _objective_exact, x0, method='COBYLA',
                    options={'maxiter': 500, 'rhobeg': 0.1}
                )
                if result2.fun < best_val:
                    best_val = result2.fun
                    best_c = result2.x.copy()
            except:
                pass

    # Phase 3: High-precision polish
    if elapsed() < deadline - 10:
        try:
            result = minimize(
                _objective_exact, best_c, method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except:
            pass

        # Tiny perturbations to escape potential local minima
        if elapsed() < deadline - 5:
            for scale in [1e-4, 1e-5, 1e-6]:
                if elapsed() > deadline - 5:
                    break
                try:
                    perturbed = best_c + np.random.normal(0, scale, size=3)
                    result = minimize(
                        _objective_exact, perturbed, method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                    )
                    if result.fun < best_val:
                        best_val = result.fun
                        best_c = result.x.copy()
                except:
                    pass

    # Final verification
    final_val = _compute_value(best_c[0], best_c[1], best_c[2])
    if final_val < best_val:
        best_val = final_val

    return best_c
