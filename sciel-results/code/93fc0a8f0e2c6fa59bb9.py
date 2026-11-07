"""
Combined approach: parametric root-construction for the uncertainty inequality.

The key structural insight is that we can parametrize the problem by the roots
of q(t) = g(x)/x^2, which is a degree-5 polynomial in t = x^2. The coefficients
of q are linear in (c0, c1, c2, c3) with c3 determined by the constraint g(0)=0.
This gives us 3 free parameters that must match 6 coefficients of q, but with
constraints from the Hermite structure.

Instead of optimizing over (c0, c1, c2) directly, we can:
1. Use the fact that q(t) = a5*t^5 + a4*t^4 + a3*t^3 + a2*t^2 + a1*t + a0
   where a4/a5 = -30 and a3/a5 = 315 are FIXED (independent of c0,c1,c2,c3).
2. The remaining coefficients a2/a5, a1/a5, a0/a5 are free parameters.
3. We want to minimize r^2/(2*pi) where r is the largest real root of q at which
   q changes sign.

New structural approach: Directly parametrize q(t) by its roots and use the
fixed ratio constraints to solve for valid (c0, c1, c2).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import product


def _compute_value_fast(c0, c1, c2):
    """
    Fast computation of the value r^2/(2*pi) for given coefficients.
    Uses the explicit form of q(t) = g(x)/x^2.
    """
    H4_0 = 12.0
    H8_0 = 1344.0
    H12_0 = 1032192.0

    c3 = -(c0 + H4_0 * c1 + H8_0 * c2) / H12_0

    # If c3 < 0, g -> -inf as x -> inf, so we flip sign
    if c3 < 0:
        c0, c1, c2, c3 = -c0, -c1, -c2, -c3

    # Coefficients of Q(t) = a5*t^5 + a4*t^4 + a3*t^3 + a2*t^2 + a1*t + a0
    a5 = 4096.0 * c3
    a4 = -122880.0 * c3
    a3 = 256.0 * c2 + 1290240.0 * c3
    a2 = -3072.0 * c2 - 5806080.0 * c3
    a1 = 16.0 * c1 + 11520.0 * c2 + 9953280.0 * c3
    a0 = -48.0 * c1 - 13440.0 * c2 - 5505024.0 * c3

    coeffs = np.array([a5, a4, a3, a2, a1, a0])

    try:
        roots_t = np.roots(coeffs)
    except Exception:
        return 1e10

    # Find positive real roots where Q changes sign
    best_r = 0.0
    for rt in roots_t:
        if abs(rt.imag) < 1e-10:
            tr = rt.real
            if tr > 1e-12:
                # Check sign change
                eps = 1e-7 * max(1.0, abs(tr))
                Q_minus = np.polyval(coeffs, tr - eps)
                Q_plus = np.polyval(coeffs, tr + eps)
                if Q_minus * Q_plus < 0:
                    if tr > best_r:
                        best_r = tr

    if best_r <= 0:
        return 1e10

    return best_r ** 2 / (2.0 * np.pi)


def _compute_value_from_q_coeffs(a2_ratio, a1_ratio, a0_ratio):
    """
    Given the ratios a2/a5, a1/a5, a0/a5, compute the value.
    a4/a5 = -30 and a3/a5 = 315 are fixed.
    We normalize by setting a5 = 1, so:
    Q(t) = t^5 - 30t^4 + 315t^3 + a2_ratio*t^2 + a1_ratio*t + a0_ratio
    """
    coeffs = np.array([1.0, -30.0, 315.0, a2_ratio, a1_ratio, a0_ratio])

    try:
        roots_t = np.roots(coeffs)
    except Exception:
        return 1e10

    best_r = 0.0
    for rt in roots_t:
        if abs(rt.imag) < 1e-10:
            tr = rt.real
            if tr > 1e-12:
                eps = 1e-7 * max(1.0, abs(tr))
                Q_minus = np.polyval(coeffs, tr - eps)
                Q_plus = np.polyval(coeffs, tr + eps)
                if Q_minus * Q_plus < 0:
                    if tr > best_r:
                        best_r = tr

    if best_r <= 0:
        return 1e10

    return best_r ** 2 / (2.0 * np.pi)


def _q_to_coefficients(a2_ratio, a1_ratio, a0_ratio):
    """
    Convert q-coefficient ratios to (c0, c1, c2).
    
    With a5 = 4096*c3 = 1 (normalized), so c3 = 1/4096.
    Then:
    a3 = 256*c2 + 1290240*c3 = 315  =>  256*c2 = 315 - 1290240/4096
    a2 = -3072*c2 - 5806080*c3 = a2_ratio
    a1 = 16*c1 + 11520*c2 + 9953280*c3 = a1_ratio
    a0 = -48*c1 - 13440*c2 - 5505024*c3 = a0_ratio
    
    From a3: c2 = (315 - 1290240/4096) / 256
    From a2: check consistency (this is a constraint, not a free parameter!)
    Actually, a2_ratio is determined by c2, so it's NOT free.
    
    Let me re-derive:
    c3 = 1/4096
    a3 = 256*c2 + 1290240*(1/4096) = 256*c2 + 315
    So a3/a5 = 315 + 256*c2 => this means a3/a5 is NOT fixed at 315!
    
    Wait, let me recheck. a3 = 256*c2 + 1290240*c3
    a3/a5 = (256*c2 + 1290240*c3)/(4096*c3) = 256*c2/(4096*c3) + 1290240/4096
    = c2/(16*c3) + 315
    
    So a3/a5 = 315 + c2/(16*c3), which is NOT fixed!
    
    Let me redo this properly. We have 3 free params (c0, c1, c2) and c3 is determined.
    The 6 coefficients of Q are:
    a5 = 4096*c3
    a4 = -122880*c3  =>  a4/a5 = -30  (FIXED)
    a3 = 256*c2 + 1290240*c3  =>  a3/a5 = 256*c2/(4096*c3) + 315 = c2/(16*c3) + 315
    a2 = -3072*c2 - 5806080*c3  =>  a2/a5 = -3072*c2/(4096*c3) - 5806080/4096 = -3*c2/(4*c3) - 1417.5
    a1 = 16*c1 + 11520*c2 + 9953280*c3  =>  a1/a5 = 16*c1/(4096*c3) + 11520*c2/(4096*c3) + 9953280/4096
       = c1/(256*c3) + 45*c2/(16*c3) + 2430
    a0 = -48*c1 - 13440*c2 - 5505024*c3  =>  a0/a5 = -48*c1/(4096*c3) - 13440*c2/(4096*c3) - 5505024/4096
       = -3*c1/(256*c3) - 105*c2/(32*c3) - 1344
    
    Let u = c1/c3, v = c2/c3. Then:
    a4/a5 = -30
    a3/a5 = 315 + v/16
    a2/a5 = -1417.5 - 3v/4
    a1/a5 = 2430 + u/256 + 45v/16
    a0/a5 = -1344 - 3u/256 - 105v/32
    
    And c3 = -(c0 + 12*c1 + 1344*c2)/1032192
    => c0 = -1032192*c3 - 12*c1 - 1344*c2 = c3*(-1032192 - 12u - 1344v)
    
    So the 3 free params can be (u, v, c3) or equivalently (c0, c1, c2).
    The q-polynomial is determined by (u, v) up to scaling by c3.
    
    So we can parametrize by (u, v) = (c1/c3, c2/c3) and the shape of Q is:
    Q(t) = t^5 - 30t^4 + (315 + v/16)t^3 + (-1417.5 - 3v/4)t^2 + (2430 + u/256 + 45v/16)t + (-1344 - 3u/256 - 105v/32)
    
    This is a 2-parameter family! Much easier to optimize.
    """
    c3 = 1.0  # Normalization, doesn't affect roots
    u = a2_ratio  # This is actually v = c2/c3, mislabeled
    v = a1_ratio  # This is actually u = c1/c3, mislabeled
    
    # Actually let me just use (u, v) directly
    return None  # This function is not used; see construct()


def construct():
    """Return three coefficients [c0, c1, c2] that minimize the value."""
    t0 = time.time()
    deadline = t0 + 195  # Safety margin before 210s kill

    H4_0 = 12.0
    H8_0 = 1344.0
    H12_0 = 1032192.0

    def objective_uv(u, v):
        """
        Objective function in terms of u = c1/c3 and v = c2/c3.
        The normalized Q(t) (with a5=1) is:
        Q(t) = t^5 - 30t^4 + (315 + v/16)t^3 + (-1417.5 - 3v/4)t^2 
               + (2430 + u/256 + 45v/16)t + (-1344 - 3u/256 - 105v/32)
        """
        a3 = 315.0 + v / 16.0
        a2 = -1417.5 - 3.0 * v / 4.0
        a1 = 2430.0 + u / 256.0 + 45.0 * v / 16.0
        a0 = -1344.0 - 3.0 * u / 256.0 - 105.0 * v / 32.0

        coeffs = np.array([1.0, -30.0, a3, a2, a1, a0])

        try:
            roots_t = np.roots(coeffs)
        except Exception:
            return 1e10

        best_r = 0.0
        for rt in roots_t:
            if abs(rt.imag) < 1e-10:
                tr = rt.real
                if tr > 1e-12:
                    eps = 1e-7 * max(1.0, abs(tr))
                    Q_minus = np.polyval(coeffs, tr - eps)
                    Q_plus = np.polyval(coeffs, tr + eps)
                    if Q_minus * Q_plus < 0:
                        if tr > best_r:
                            best_r = tr

        if best_r <= 0:
            return 1e10

        return best_r ** 2 / (2.0 * np.pi)

    def uv_to_coefficients(u, v):
        """Convert (u, v) = (c1/c3, c2/c3) to (c0, c1, c2)."""
        c3 = 1.0  # Arbitrary scaling
        c1 = u * c3
        c2 = v * c3
        c0 = -H12_0 * c3 - H4_0 * c1 - H8_0 * c2
        return np.array([c0, c1, c2])

    # Seed in (u, v) space
    seed_c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    c3_seed = -(seed_c[0] + H4_0 * seed_c[1] + H8_0 * seed_c[2]) / H12_0
    u_seed = seed_c[1] / c3_seed
    v_seed = seed_c[2] / c3_seed

    best_uv = np.array([u_seed, v_seed])
    best_val = objective_uv(u_seed, v_seed)
    best_coeffs = seed_c.copy()

    # Strategy 1: Differential evolution in (u, v) space
    if time.time() < deadline - 60:
        try:
            bounds = [(-100.0, 100.0), (-100.0, 100.0)]
            de_result = differential_evolution(
                lambda x: objective_uv(x[0], x[1]),
                bounds, maxiter=150, popsize=30, seed=42,
                tol=1e-12, polish=False, init='latinhypercube'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_uv = de_result.x.copy()
                best_coeffs = uv_to_coefficients(best_uv[0], best_uv[1])
        except Exception:
            pass

    # Strategy 2: Multiple Nelder-Mead restarts in (u, v) space
    if time.time() < deadline - 20:
        rng = np.random.default_rng(12345)
        for i in range(40):
            if time.time() > deadline - 15:
                break
            if i == 0:
                x0 = best_uv.copy()
            else:
                scale = 10.0 * (1.0 - i / 40.0) + 0.5
                x0 = best_uv + scale * rng.standard_normal(2)

            try:
                nm_result = minimize(
                    lambda x: objective_uv(x[0], x[1]),
                    x0, method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                )
                if nm_result.fun < best_val:
                    best_val = nm_result.fun
                    best_uv = nm_result.x.copy()
                    best_coeffs = uv_to_coefficients(best_uv[0], best_uv[1])
            except Exception:
                pass

    # Strategy 3: Final refinement
    if time.time() < deadline - 5:
        try:
            nm_final = minimize(
                lambda x: objective_uv(x[0], x[1]),
                best_uv, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True}
            )
            if nm_final.fun < best_val:
                best_val = nm_final.fun
                best_uv = nm_final.x.copy()
                best_coeffs = uv_to_coefficients(best_uv[0], best_uv[1])
        except Exception:
            pass

    # Verify the result
    final_val = _compute_value_fast(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    if final_val < best_val:
        best_val = final_val

    return best_coeffs
# EVOLVE-BLOCK-END
