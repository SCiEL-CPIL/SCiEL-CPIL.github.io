import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# Precomputed Hermite polynomial coefficients in terms of y = x^2
# H0(y) = 1
# H4(y) = y^2 - 6y + 3
# H8(y) = y^4 - 28y^3 + 280y^2 - 840y + 105
# H12(y) = y^6 - 60y^5 + 1260y^4 - 12600y^3 + 45360y^2 - 30240y + 10395

H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 10395.0


def _build_P_coeffs(c0, c1, c2):
    """Build coefficients of P(y) = g(sqrt(y))/y as a degree-5 polynomial in y.
    Returns [a5, a4, a3, a2, a1, a0] for a5*y^5 + a4*y^4 + ... + a0."""
    c3 = -(c0 + 3.0 * c1 + 105.0 * c2) / 10395.0

    a5 = c3
    a4 = -60.0 * c3
    a3 = 1260.0 * c3 + c2
    a2 = -12600.0 * c3 - 28.0 * c2
    a1 = 45360.0 * c3 + 280.0 * c2 - 6.0 * c1
    a0 = -30240.0 * c3 - 840.0 * c2 - 6.0 * c1

    return np.array([a5, a4, a3, a2, a1, a0])


def _eval_P(coeffs, y):
    """Evaluate P(y) using Horner's method for speed and stability."""
    a5, a4, a3, a2, a1, a0 = coeffs
    return (((a5 * y + a4) * y + a3) * y + a2) * y * y + (a1 * y + a0)


def _compute_value(c0, c1, c2):
    """Compute r^2 / (2*pi) where r is the largest sign-changing positive root of q(x)=g(x)/x^2."""
    coeffs = _build_P_coeffs(c0, c1, c2)

    # Strip leading near-zeros
    tol = 1e-14
    idx = 0
    while idx < len(coeffs) - 1 and abs(coeffs[idx]) < tol:
        idx += 1
    coeffs = coeffs[idx:]

    if len(coeffs) == 1:
        # Constant polynomial, no roots
        return 1e100

    try:
        roots = np.roots(coeffs)
    except Exception:
        return 1e100

    # Filter positive real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9:
            r_real = r.real
            if r_real > 1e-10:
                real_roots.append(r_real)

    if not real_roots:
        return 1e100

    # Find the largest root where P changes sign (odd multiplicity)
    max_y = 0.0
    eps_factor = 1e-7
    for r in sorted(real_roots, reverse=True):
        eps = eps_factor * max(1.0, r)
        y_left = max(0.0, r - eps)
        y_right = r + eps

        val_left = _eval_P(coeffs, y_left)
        val_right = _eval_P(coeffs, y_right)

        if val_left * val_right < 0:
            max_y = r
            break  # Already sorted descending, first sign-changing is the largest

    if max_y == 0.0:
        return 1e100

    return max_y / (2.0 * np.pi)


def _smooth_objective(x, temperature=0.01):
    """Smooth surrogate: soft-max over candidate root contributions.
    This helps gradient-based methods navigate the nonsmooth landscape."""
    c0, c1, c2 = x
    coeffs = _build_P_coeffs(c0, c1, c2)

    idx = 0
    while idx < len(coeffs) - 1 and abs(coeffs[idx]) < 1e-14:
        idx += 1
    coeffs = coeffs[idx:]

    if len(coeffs) == 1:
        return 1e100

    try:
        roots = np.roots(coeffs)
    except Exception:
        return 1e100

    # Collect positive real roots with sign-change check
    candidates = []
    eps_factor = 1e-7
    for r in roots:
        if abs(r.imag) < 1e-9:
            r_real = r.real
            if r_real > 1e-10:
                eps = eps_factor * max(1.0, r_real)
                y_left = max(0.0, r_real - eps)
                y_right = r_real + eps
                val_left = _eval_P(coeffs, y_left)
                val_right = _eval_P(coeffs, y_right)
                if val_left * val_right < 0:
                    candidates.append(r_real / (2.0 * np.pi))

    if not candidates:
        return 1e100

    # Soft-max: approximately max but smooth
    cands = np.array(candidates)
    max_c = np.max(cands)
    if max_c > 1e50:
        return max_c
    log_sum = np.log(np.sum(np.exp((cands - max_c) / temperature)))
    return max_c + temperature * log_sum


def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s limit

    # Seed
    seed_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    seed_value = _compute_value(*seed_coeffs)

    best_coeffs = seed_coeffs.copy()
    best_value = seed_value

    # Phase 1: Global search with differential evolution
    elapsed = time.time() - start_time
    if elapsed < 100.0:
        try:
            bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]

            def objective(x):
                return _compute_value(x[0], x[1], x[2])

            result = differential_evolution(
                objective, bounds,
                maxiter=150, popsize=18,
                tol=1e-10, seed=42,
                polish=True,
                updating='deferred'
            )
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Phase 2: Local refinement with multiple starts using smooth objective
    elapsed = time.time() - start_time
    if elapsed < 170.0:
        rng = np.random.default_rng(42)

        starts = [best_coeffs.copy()]
        # Add structured random starts
        for _ in range(30):
            s = seed_coeffs + rng.normal(0, 0.8, 3)
            starts.append(s)

        for s in starts:
            if time.time() - start_time > 175.0:
                break
            try:
                # First try with smooth objective to navigate nonsmoothness
                res_smooth = minimize(
                    _smooth_objective, s,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if res_smooth.fun < best_value:
                    # Polish with exact objective
                    res_exact = minimize(
                        lambda x: _compute_value(x[0], x[1], x[2]),
                        res_smooth.x,
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
                    )
                    if res_exact.fun < best_value:
                        best_value = res_exact.fun
                        best_coeffs = res_exact.x.copy()
                else:
                    # Try direct exact objective from this start
                    res_exact = minimize(
                        lambda x: _compute_value(x[0], x[1], x[2]),
                        s,
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
                    )
                    if res_exact.fun < best_value:
                        best_value = res_exact.fun
                        best_coeffs = res_exact.x.copy()
            except Exception:
                pass

    # Phase 3: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < 190.0 and best_value < 1e50:
        try:
            res_polish = minimize(
                lambda x: _compute_value(x[0], x[1], x[2]),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res_polish.fun < best_value:
                best_value = res_polish.fun
                best_coeffs = res_polish.x.copy()
        except Exception:
            pass

    return best_coeffs
