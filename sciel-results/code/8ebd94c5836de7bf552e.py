import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Precomputed Hermite polynomial values at x=0
H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 10395.0

# Hermite polynomials in y = x^2:
# H0(y) = 1
# H4(y) = y^2 - 6y + 3
# H8(y) = y^4 - 28y^3 + 280y^2 - 840y + 105
# H12(y) = y^6 - 60y^5 + 1260y^4 - 12600y^3 + 45360y^2 - 30240y + 10395


def _compute_value_from_coeffs(c0, c1, c2):
    """Compute r^2/(2*pi) where r is the largest sign-changing positive root of q(x)=g(x)/x^2."""
    c3 = -(c0 + 3.0 * c1 + 105.0 * c2) / 10395.0

    # P(y) = g(sqrt(y))/y has degree 5 (or less) in y
    # P(y) = c3*y^5 + (-60*c3)*y^4 + (1260*c3+c2)*y^3 + (-12600*c3-28*c2)*y^2
    #        + (45360*c3+280*c2-6*c1)*y + (-30240*c3-840*c2-6*c1)
    a5 = c3
    a4 = -60.0 * c3
    a3 = 1260.0 * c3 + c2
    a2 = -12600.0 * c3 - 28.0 * c2
    a1 = 45360.0 * c3 + 280.0 * c2 - 6.0 * c1
    a0 = -30240.0 * c3 - 840.0 * c2 - 6.0 * c1

    coeffs = np.array([a5, a4, a3, a2, a1, a0])

    # Strip leading near-zeros
    tol = 1e-14
    idx = 0
    while idx < len(coeffs) - 1 and abs(coeffs[idx]) < tol:
        idx += 1
    coeffs = coeffs[idx:]

    if len(coeffs) == 1:
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

    # Find the largest root where P changes sign
    max_y = 0.0
    eps_factor = 1e-7
    for r in sorted(real_roots, reverse=True):
        eps = eps_factor * max(1.0, r)
        y_left = max(0.0, r - eps)
        y_right = r + eps

        # Evaluate P at y_left and y_right using Horner
        val_left = (((((a5 * y_left + a4) * y_left + a3) * y_left + a2) * y_left + a1) * y_left + a0)
        val_right = (((((a5 * y_right + a4) * y_right + a3) * y_right + a2) * y_right + a1) * y_right + a0)

        if val_left * val_right < 0:
            max_y = r
            break

    if max_y == 0.0:
        return 1e100

    return max_y / (2.0 * np.pi)


def _coeffs_from_roots_and_scale(roots_y, scale):
    """Given roots of P(y) in y-space and a scale, compute (c0,c1,c2).
    P(y) = scale * prod(y - r_i) for the chosen roots.
    We need P to match the form derived from Hermite combination.
    This is a linear system: given P's coefficients, solve for c0,c1,c2 (c3 determined).
    
    Actually, P is degree 5 with specific structure. Let's think differently:
    P(y) = c3*y^5 + a4*y^4 + a3*y^3 + a2*y^2 + a1*y + a0
    where a4=-60*c3, a3=1260*c3+c2, a2=-12600*c3-28*c2, a1=45360*c3+280*c2-6*c1, a0=-30240*c3-840*c2-6*c1
    
    So P is determined by 3 free params (c0,c1,c2) with c3 fixed by g(0)=0.
    The space of P is 3-dimensional within the 6-dimensional space of degree-5 polynomials.
    
    Alternative: parameterize by 3 roots + 1 scale = 4 params, but P is only 3-dim.
    So we can't freely choose roots. Let's stick with coefficient-space but use a better strategy.
    """
    pass


def _objective(x):
    """Objective: minimize the value."""
    c0, c1, c2 = x
    return _compute_value_from_coeffs(c0, c1, c2)


def _smooth_objective(x, temp=0.05):
    """Smooth surrogate for the nonsmooth objective."""
    c0, c1, c2 = x
    c3 = -(c0 + 3.0 * c1 + 105.0 * c2) / 10395.0

    a5 = c3
    a4 = -60.0 * c3
    a3 = 1260.0 * c3 + c2
    a2 = -12600.0 * c3 - 28.0 * c2
    a1 = 45360.0 * c3 + 280.0 * c2 - 6.0 * c1
    a0 = -30240.0 * c3 - 840.0 * c2 - 6.0 * c1

    coeffs = np.array([a5, a4, a3, a2, a1, a0])
    tol = 1e-14
    idx = 0
    while idx < len(coeffs) - 1 and abs(coeffs[idx]) < tol:
        idx += 1
    coeffs = coeffs[idx:]

    if len(coeffs) == 1:
        return 1e100

    try:
        roots = np.roots(coeffs)
    except Exception:
        return 1e100

    candidates = []
    eps_factor = 1e-7
    for r in roots:
        if abs(r.imag) < 1e-9:
            r_real = r.real
            if r_real > 1e-10:
                eps = eps_factor * max(1.0, r_real)
                y_left = max(0.0, r_real - eps)
                y_right = r_real + eps
                val_left = (((((a5 * y_left + a4) * y_left + a3) * y_left + a2) * y_left + a1) * y_left + a0)
                val_right = (((((a5 * y_right + a4) * y_right + a3) * y_right + a2) * y_right + a1) * y_right + a0)
                if val_left * val_right < 0:
                    candidates.append(r_real / (2.0 * np.pi))

    if not candidates:
        return 1e100

    cands = np.array(candidates)
    max_c = np.max(cands)
    if max_c > 1e50:
        return max_c
    # Soft max
    log_sum = np.log(np.sum(np.exp((cands - max_c) / temp)))
    return max_c + temp * log_sum


def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s limit

    seed_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    seed_value = _compute_value_from_coeffs(*seed_coeffs)

    best_coeffs = seed_coeffs.copy()
    best_value = seed_value

    # Phase 1: Global search with differential evolution
    elapsed = time.time() - start_time
    if elapsed < 80.0:
        try:
            bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
            result = differential_evolution(
                _objective, bounds,
                maxiter=200, popsize=20,
                tol=1e-12, seed=42,
                polish=True,
                updating='deferred'
            )
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Phase 2: Multi-start local refinement
    elapsed = time.time() - start_time
    if elapsed < 185.0:
        rng = np.random.default_rng(42)

        starts = [best_coeffs.copy()]
        # Structured random starts around seed and best
        for _ in range(40):
            s = seed_coeffs + rng.normal(0, 1.0, 3)
            starts.append(s)
        for _ in range(20):
            s = best_coeffs + rng.normal(0, 0.5, 3)
            starts.append(s)

        for s in starts:
            if time.time() - start_time > 190.0:
                break
            try:
                # Smooth objective first
                res_smooth = minimize(
                    _smooth_objective, s,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if res_smooth.fun < best_value:
                    res_exact = minimize(
                        _objective, res_smooth.x,
                        method='Nelder-Mead',
                        options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
                    )
                    if res_exact.fun < best_value:
                        best_value = res_exact.fun
                        best_coeffs = res_exact.x.copy()
                else:
                    res_exact = minimize(
                        _objective, s,
                        method='Nelder-Mead',
                        options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
                    )
                    if res_exact.fun < best_value:
                        best_value = res_exact.fun
                        best_coeffs = res_exact.x.copy()
            except Exception:
                pass

    # Phase 3: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < 195.0 and best_value < 1e50:
        try:
            res_polish = minimize(
                _objective, best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
            )
            if res_polish.fun < best_value:
                best_value = res_polish.fun
                best_coeffs = res_polish.x.copy()
        except Exception:
            pass

    return best_coeffs
