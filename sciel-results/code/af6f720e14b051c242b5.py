"""Structural approach: Newton root-finding on double-root constraint in coefficient space."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

_DEADLINE = 175.0
_t_start = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _t_start)


def _smooth_min_roots(roots, alpha=50.0):
    """Smooth approximation of min of positive real roots using log-sum-exp."""
    pos = [r for r in roots if r > 1e-12]
    if not pos:
        return np.inf
    pos = np.array(pos)
    m = np.min(pos)
    return m + np.log(np.mean(np.exp(alpha * (pos - m)))) / alpha


def _q_coefficients(c0, c1, c2):
    """Return [b0..b5] where q(y) = b0 + b1*y + ... + b5*y^5, y=x^2."""
    c3 = -(c0 + 12.0 * c1 + 168.0 * c2) / 18480.0
    b0 = -48.0 * c1 - 40320.0 * c2 - 5109120.0 * c3
    b1 = 16.0 * c1 + 26880.0 * c2 + 9072000.0 * c3
    b2 = -4608.0 * c2 - 6005760.0 * c3
    b3 = 256.0 * c2 + 1424640.0 * c3
    b4 = -142464.0 * c3
    b5 = 4096.0 * c3
    return np.array([b0, b1, b2, b3, b4, b5])


def _q_eval(coeffs, y):
    """Horner evaluation of q(y)."""
    r = 0.0
    for b in coeffs:
        r = r * y + b
    return r


def _q_deriv(coeffs, y):
    """Derivative q'(y) via Horner."""
    r = 0.0
    for i in range(len(coeffs) - 1, 0, -1):
        r = r * y + i * coeffs[i]
    return r


def _q_second(coeffs, y):
    """Second derivative q''(y)."""
    r = 0.0
    for i in range(len(coeffs) - 1, 1, -1):
        r = r * y + i * (i - 1) * coeffs[i]
    return r


def _find_largest_sign_change_root(coeffs):
    """Find largest real positive root where q changes sign. Returns y or None."""
    poly_coeffs = np.array(coeffs[::-1])
    roots = np.roots(poly_coeffs)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9 and r.real > 1e-12:
            real_roots.append(r.real)
    if not real_roots:
        return None
    real_roots.sort(reverse=True)
    for y in real_roots:
        eps = max(1e-7, abs(y) * 1e-7)
        qb = _q_eval(coeffs, y - eps)
        qa = _q_eval(coeffs, y + eps)
        if qb * qa < 0:
            return y
    return None


def _smooth_objective(c0, c1, c2, alpha=50.0):
    """
    Smooth surrogate: approximate min of sign-changing roots squared / (2*pi).
    Uses log-sum-exp to smoothly approximate the minimum over all positive roots.
    Only counts roots where sign change is likely (q(y-eps)*q(y+eps)<0).
    """
    coeffs = _q_coefficients(c0, c1, c2)
    poly_coeffs = np.array(coeffs[::-1])
    roots = np.roots(poly_coeffs)
    valid_roots = []
    for r in roots:
        if abs(r.imag) < 1e-8 and r.real > 1e-10:
            y = r.real
            eps = max(1e-6, abs(y) * 1e-6)
            qb = _q_eval(coeffs, y - eps)
            qa = _q_eval(coeffs, y + eps)
            if qb * qa < 0:
                valid_roots.append(y)
    if not valid_roots:
        return 1e10  # large penalty
    valid_roots = np.array(valid_roots)
    # Smooth min via log-sum-exp
    m = np.min(valid_roots)
    smooth_min = m + np.log(np.mean(np.exp(alpha * (valid_roots - m)))) / alpha
    return smooth_min / (2.0 * np.pi)


def _evaluate_value(c0, c1, c2):
    """Compute value = r^2/(2*pi)."""
    coeffs = _q_coefficients(c0, c1, c2)
    y_root = _find_largest_sign_change_root(coeffs)
    if y_root is None:
        return np.inf
    return y_root / (2.0 * np.pi)


# Seed
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_seed_value = _evaluate_value(_SEED[0], _SEED[1], _SEED[2])


def _newton_double_root(c0, c1, c2, y_target, max_iter=50, tol=1e-14):
    """
    Newton's method to find (c1, c2) such that q(y_target)=0 and q'(y_target)=0,
    holding c0 fixed. Returns (c1, c2, success).
    """
    c1_cur = c1
    c2_cur = c2
    dc3_dc1 = -12.0 / 18480.0
    dc3_dc2 = -168.0 / 18480.0
    
    db0_dc1 = -48.0 + (-5109120.0) * dc3_dc1
    db1_dc1 = 16.0 + 9072000.0 * dc3_dc1
    db2_dc1 = 0.0 + (-6005760.0) * dc3_dc1
    db3_dc1 = 0.0 + 1424640.0 * dc3_dc1
    db4_dc1 = 0.0 + (-142464.0) * dc3_dc1
    db5_dc1 = 0.0 + 4096.0 * dc3_dc1
    
    db0_dc2 = -40320.0 + (-5109120.0) * dc3_dc2
    db1_dc2 = 26880.0 + 9072000.0 * dc3_dc2
    db2_dc2 = -4608.0 + (-6005760.0) * dc3_dc2
    db3_dc2 = 256.0 + 1424640.0 * dc3_dc2
    db4_dc2 = 0.0 + (-142464.0) * dc3_dc2
    db5_dc2 = 0.0 + 4096.0 * dc3_dc2
    
    y = y_target
    y2 = y * y
    y3 = y2 * y
    y4 = y2 * y2
    y5 = y3 * y2
    
    df1_dc1 = db0_dc1 + db1_dc1 * y + db2_dc1 * y2 + db3_dc1 * y3 + db4_dc1 * y4 + db5_dc1 * y5
    df1_dc2 = db0_dc2 + db1_dc2 * y + db2_dc2 * y2 + db3_dc2 * y3 + db4_dc2 * y4 + db5_dc2 * y5
    df2_dc1 = db1_dc1 + 2.0 * db2_dc1 * y + 3.0 * db3_dc1 * y2 + 4.0 * db4_dc1 * y3 + 5.0 * db5_dc1 * y4
    df2_dc2 = db1_dc2 + 2.0 * db2_dc2 * y + 3.0 * db3_dc2 * y2 + 4.0 * db4_dc2 * y3 + 5.0 * db5_dc2 * y4
    
    J = np.array([[df1_dc1, df1_dc2],
                  [df2_dc1, df2_dc2]])
    det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    if abs(det) < 1e-30:
        return c1_cur, c2_cur, False
    J_inv = np.array([[J[1, 1], -J[0, 1]],
                      [-J[1, 0], J[0, 0]]]) / det
    
    best_c1, best_c2, best_err = c1_cur, c2_cur, np.inf
    for _ in range(max_iter):
        coeffs = _q_coefficients(c0, c1_cur, c2_cur)
        f1 = _q_eval(coeffs, y_target)
        f2 = _q_deriv(coeffs, y_target)
        err = abs(f1) + abs(f2)
        if err < best_err:
            best_err = err
            best_c1, best_c2 = c1_cur, c2_cur
        if abs(f1) < tol and abs(f2) < tol:
            return c1_cur, c2_cur, True
        delta = J_inv @ np.array([f1, f2])
        step_size = 1.0
        for _ in range(10):
            c1_new = c1_cur - step_size * delta[0]
            c2_new = c2_cur - step_size * delta[1]
            coeffs_new = _q_coefficients(c0, c1_new, c2_new)
            f1_new = _q_eval(coeffs_new, y_target)
            f2_new = _q_deriv(coeffs_new, y_target)
            if abs(f1_new) + abs(f2_new) < err:
                c1_cur, c2_cur = c1_new, c2_new
                break
            step_size *= 0.5
        else:
            break
    coeffs = _q_coefficients(c0, best_c1, best_c2)
    f1 = _q_eval(coeffs, y_target)
    f2 = _q_deriv(coeffs, y_target)
    return best_c1, best_c2, (abs(f1) < 1e-10 and abs(f2) < 1e-10)


def construct():
    """
    Search for coefficients [c0, c1, c2] minimizing value = r^2/(2*pi).
    Uses Newton's method on double-root constraint with bisection over target y*.
    """
    global _t_start
    _t_start = time.time()

    best_params = _SEED.copy()
    best_value = _seed_value

    # Phase 1: Direct Newton search with improved damping
    c0_values = []
    for dc0 in np.linspace(-1.0, 1.0, 21):
        c0_values.append(_SEED[0] + dc0)
    for c0 in np.linspace(-5.0, 5.0, 21):
        c0_values.append(c0)
    c0_values = list(set(round(c, 6) for c in c0_values))
    c0_values.sort()
    
    y_targets = np.logspace(-5, 2.0, 40)
    
    for c0 in c0_values:
        if _remaining() < 5:
            break
        for y_t in y_targets:
            if _remaining() < 3:
                break
            for c1_init in [-1.0, -0.5, 0.0, 0.5, 1.0, _SEED[1]]:
                for c2_init in [-0.5, -0.1, 0.0, 0.1, 0.5, _SEED[2]]:
                    if _remaining() < 1:
                        break
                    c1_new, c2_new, success = _newton_double_root(
                        c0, c1_init, c2_init, y_t, max_iter=50, tol=1e-13)
                    if success:
                        val = _evaluate_value(c0, c1_new, c2_new)
                        if np.isfinite(val) and val < best_value:
                            best_value = val
                            best_params = np.array([c0, c1_new, c2_new])
    
    # Phase 2: Smooth surrogate optimization via coordinate descent
    if _remaining() > 15:
        params = best_params.copy()
        step = 0.1
        for _ in range(200):
            if _remaining() < 5:
                break
            improved = False
            for i in range(3):
                for sign in [1, -1]:
                    trial = params.copy()
                    trial[i] += sign * step
                    obj = _smooth_objective(trial[0], trial[1], trial[2], alpha=30.0)
                    if obj < _smooth_objective(params[0], params[1], params[2], alpha=30.0) - 1e-8:
                        params = trial
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                step *= 0.5
                if step < 1e-8:
                    break
            val = _evaluate_value(params[0], params[1], params[2])
            if np.isfinite(val) and val < best_value:
                best_value = val
                best_params = params.copy()
    
    # Phase 3: Local refinement around best found point
    if _remaining() > 10 and best_value < np.inf:
        for dc0 in np.linspace(-0.05, 0.05, 5):
            for dc1 in np.linspace(-0.05, 0.05, 5):
                for dc2 in np.linspace(-0.05, 0.05, 5):
                    if _remaining() < 2:
                        break
                    c0_p = best_params[0] + dc0
                    c1_p = best_params[1] + dc1
                    c2_p = best_params[2] + dc2
                    y_t = best_value * 2.0 * np.pi * 0.98
                    c1_new, c2_new, success = _newton_double_root(
                        c0_p, c1_p, c2_p, y_t, max_iter=60, tol=1e-14)
                    if success:
                        val = _evaluate_value(c0_p, c1_new, c2_new)
                        if np.isfinite(val) and val < best_value:
                            best_value = val
                            best_params = np.array([c0_p, c1_new, c2_new])
    
    # Phase 4: Final polish - try to push y even lower from best point
    if _remaining() > 5 and best_value < np.inf:
        y_current = best_value * 2.0 * np.pi
        for factor in [0.999, 0.998, 0.995, 0.99, 0.98, 0.97]:
            y_t = y_current * factor
            c1_new, c2_new, success = _newton_double_root(
                best_params[0], best_params[1], best_params[2], y_t, max_iter=80, tol=1e-15)
            if success:
                val = _evaluate_value(best_params[0], c1_new, c2_new)
                if np.isfinite(val) and val < best_value:
                    best_value = val
                    best_params = np.array([best_params[0], c1_new, c2_new])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
