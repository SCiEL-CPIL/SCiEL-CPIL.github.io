# EVOLVE-BLOCK-START
import time
import numpy as np

# Precompute Hermite polynomial coefficients (physicists' convention)
# H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# H_0=1, H_1=2x, H_2=4x^2-2, H_4=16x^4-48x^2+12,
# H_8=256x^8-3072x^6+11520x^4-13440x^2+3360,
# H_12=4096x^12-98304x^10+1290240x^8-9031680x^6+35380480x^4-76807680x^2+54432000

_H0 = np.array([1.0])
_H1 = np.array([2.0, 0.0])
_H2 = np.array([4.0, -2.0])
_H4 = np.array([16.0, 0.0, -48.0, 0.0, 12.0])
_H8 = np.array([256.0, 0.0, -3072.0, 0.0, 11520.0, 0.0, -13440.0, 0.0, 3360.0])
_H12 = np.array([4096.0, 0.0, -98304.0, 0.0, 1290240.0, 0.0, -9031680.0, 0.0,
                 35380480.0, 0.0, -76807680.0, 0.0, 54432000.0])

# q = g/x^2: shift down by 2 in coefficient array
# q_coeffs[i] corresponds to x^i
_Q0 = np.array([1.0])                       # H0/x^2 = 1/x^2 -> not polynomial!
# Wait: H0 = 1, so H0/x^2 = x^{-2}. That's not a polynomial.
# But the task says "q = g/x^2, which must divide exactly."
# H0(0)=1, H4(0)=12, H8(0)=3360, H12(0)=54432000
# g = c0*1 + c1*H4 + c2*H8 + c3*H12
# g/x^2: H0/x^2 = 1/x^2 is NOT a polynomial.
# So the constraint is that the x^{-2} and x^{-1} terms vanish.
# H0 = 1 -> contributes c0*x^{-2}
# H4 = 16x^4 - 48x^2 + 12 -> /x^2 = 16x^2 - 48 + 12/x^2
# H8 = 256x^8-3072x^6+11520x^4-13440x^2+3360 -> /x^2 = 256x^6-3072x^4+11520x^2-13440+3360/x^2
# H12 = 4096x^12-...+54432000 -> /x^2 = 4096x^10-...+54432000/x^2

# For q to be a polynomial, the x^{-2} coefficient must be 0:
# c0*1 + c1*12 + c2*3360 + c3*54432000 = 0
# This is exactly the g(0)=0 condition! Since g(0) = c0*1 + c1*12 + c2*3360 + c3*54432000.
# And there's no x^{-1} term since all H_n are even.
# So g(0)=0 ensures q is a polynomial.

# q polynomial coefficients (x^0 to x^10):
# H0/x^2: no polynomial part (just -c0 contribution to x^{-2}, cancelled)
# Actually let me just build q directly:
# q(x) = c0*(1/x^2) + c1*(16x^2-48+12/x^2) + c2*(256x^6-3072x^4+11520x^2-13440+3360/x^2)
#        + c3*(4096x^10-98304x^8+1290240x^6-9031680x^4+35380480x^2-76807680+54432000/x^2)
# The x^{-2} terms: c0 + 12c1 + 3360c2 + 54432000c3 = g(0) = 0. Good.
# Polynomial part of q (x^0 to x^10):
# x^0: -48*c1 - 13440*c2 - 76807680*c3
# x^2: 16*c1 + 11520*c2 + 35380480*c3
# x^4: -3072*c2 - 9031680*c3
# x^6: 256*c2 + 1290240*c3
# x^8: -98304*c3
# x^10: 4096*c3

def _compute_q_coeffs(c0, c1, c2):
    """Return coefficients of q(x) = g/x^2 as a polynomial [a0, a2, a4, a6, a8, a10]
    (only even powers)."""
    c3 = -(c0 + 12.0*c1 + 3360.0*c2) / 54432000.0
    a0  = -48.0*c1 - 13440.0*c2 - 76807680.0*c3
    a2  = 16.0*c1 + 11520.0*c2 + 35380480.0*c3
    a4  = -3072.0*c2 - 9031680.0*c3
    a6  = 256.0*c2 + 1290240.0*c3
    a8  = -98304.0*c3
    a10 = 4096.0*c3
    return np.array([a0, a2, a4, a6, a8, a10]), c3

def _largest_sign_change_root(q_coeffs):
    """Find the largest real root of q at which q changes sign.
    q_coeffs = [a0, a2, a4, a6, a8, a10] for even polynomial."""
    # Build full polynomial coefficients for np.roots: [a10, a8, a6, a4, a2, a0, 0, 0, 0, 0, 0, 0]
    # Actually np.roots takes [highest, ..., lowest]
    full = np.array([q_coeffs[5], 0, q_coeffs[4], 0, q_coeffs[3], 0,
                     q_coeffs[2], 0, q_coeffs[1], 0, q_coeffs[0]])
    roots = np.roots(full)
    # Filter real roots
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    # For each real root, check sign change by evaluating q on both sides
    best_r = 0.0
    for r in real_roots:
        if r < 0:
            continue
        eps = 1e-6 * max(1.0, abs(r))
        qm = _eval_q(q_coeffs, r - eps)
        qp = _eval_q(q_coeffs, r + eps)
        if qm * qp < 0:
            if r > best_r:
                best_r = r
    return best_r

def _eval_q(q_coeffs, x):
    """Evaluate q at x using only even coefficients."""
    # Horner on y = x^2
    y = x * x
    result = q_coeffs[5]
    for i in range(4, -1, -1):
        result = result * y + q_coeffs[i]
    return result

def _compute_value(c0, c1, c2):
    q_coeffs, c3 = _compute_q_coeffs(c0, c1, c2)
    if c3 >= 0:
        # g tends to +infinity, replace g by -g, so q -> -q
        q_coeffs = -q_coeffs
        c3 = -c3
    r = _largest_sign_change_root(q_coeffs)
    return r * r / (2.0 * np.pi)

def _compute_value_fast(c0, c1, c2):
    """Fast version for scanning: use np.roots on the even polynomial via substitution."""
    q_coeffs, c3 = _compute_q_coeffs(c0, c1, c2)
    if c3 >= 0:
        q_coeffs = -q_coeffs
    # Find roots of even polynomial by substituting y = x^2
    # P(y) = a10*y^5 + a8*y^4 + a6*y^3 + a4*y^2 + a2*y + a0
    y_roots = np.roots(np.array([q_coeffs[5], q_coeffs[4], q_coeffs[3],
                                  q_coeffs[2], q_coeffs[1], q_coeffs[0]]))
    # Real positive y roots give x = sqrt(y)
    real_y = y_roots[np.abs(y_roots.imag) < 1e-10].real
    pos_y = real_y[real_y > 0]
    candidates = np.sqrt(pos_y)
    best_r = 0.0
    for r in candidates:
        eps = 1e-6 * max(1.0, abs(r))
        qm = _eval_q(q_coeffs, r - eps)
        qp = _eval_q(q_coeffs, r + eps)
        if qm * qp < 0:
            if r > best_r:
                best_r = r
    return best_r * best_r / (2.0 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # safety margin before 210s kill

    # Seed
    c0_s, c1_s, c2_s = -113.0/100, 1.0/25, 1.0/3240
    best_val = _compute_value_fast(c0_s, c1_s, c2_s)
    best_c = np.array([c0_s, c1_s, c2_s])

    # Dense 2D scan over (c0, c1) with c2 fixed at seed value
    c2_fixed = c2_s
    n_scan = 80
    c0_range = np.linspace(-2.0, 0.5, n_scan)
    c1_range = np.linspace(-0.5, 0.5, n_scan)

    scan_best_val = best_val
    scan_best_c = best_c.copy()

    for i, c0 in enumerate(c0_range):
        if time.time() > deadline - 60:
            break
        for j, c1 in enumerate(c1_range):
            if time.time() > deadline - 60:
                break
            try:
                v = _compute_value_fast(c0, c1, c2_fixed)
                if v < scan_best_val:
                    scan_best_val = v
                    scan_best_c = np.array([c0, c1, c2_fixed])
            except Exception:
                pass

    # Refine around the best point found
    if scan_best_val < best_val:
        best_val = scan_best_val
        best_c = scan_best_c.copy()

    # 3D local refinement around best point
    center = best_c.copy()
    step = 0.1
    for iteration in range(200):
        if time.time() > deadline - 30:
            break
        improved = False
        for dim in range(3):
            for direction in [-1, 1]:
                trial = center.copy()
                trial[dim] += direction * step
                try:
                    v = _compute_value_fast(trial[0], trial[1], trial[2])
                    if v < best_val - 1e-12:
                        best_val = v
                        center = trial.copy()
                        improved = True
                except Exception:
                    pass
        if not improved:
            step *= 0.5
            if step < 1e-10:
                break

    # Final high-precision check
    try:
        v_final = _compute_value(center[0], center[1], center[2])
        if v_final < best_val:
            best_val = v_final
    except Exception:
        pass

    return np.array([center[0], center[1], center[2]])
# EVOLVE-BLOCK-END
