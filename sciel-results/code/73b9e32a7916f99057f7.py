"""Search for coefficients minimizing the outermost sign-changing root of q = g/x^2."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

# Hermite polynomials H0, H4, H8, H12 in physicists' convention (monic leading coeff).
# H0 = 1
# H4 = 16 x^4 - 48 x^2 + 12
# H8 = 65536 x^8 - 294912 x^6 + 491520 x^4 - 241920 x^2 + 20160
# H12 = 4096 x^12 - 245760 x^10 + 5535360 x^8 - 49612800 x^6 + 190515840 x^4 - 267411840 x^2 + 103783680

_H12 = np.array([
    4096.0, 0.0,
    -245760.0, 0.0,
    5535360.0, 0.0,
    -49612800.0, 0.0,
    190515840.0, 0.0,
    -267411840.0, 0.0,
    103783680.0,
])  # degree 12, ascending powers

_H8 = np.array([
    65536.0, 0.0,
    -294912.0, 0.0,
    491520.0, 0.0,
    -241920.0, 0.0,
    20160.0,
])  # degree 8

_H4 = np.array([
    16.0, 0.0,
    -48.0, 0.0,
    12.0,
])  # degree 4

_H0 = np.array([1.0])  # degree 0

_H12_0 = _H12[-1]  # H12(0) = 103783680
_H8_0 = _H8[-1]    # H8(0) = 20160
_H4_0 = _H4[-1]    # H4(0) = 12
_H0_0 = _H0[-1]    # H0(0) = 1


def _g_coeffs(c0, c1, c2):
    """Return coefficients of g(x) = c0 H0 + c1 H4 + c2 H8 + c3 H12 (ascending powers)."""
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    # Build g as array of degree 12
    g = np.zeros(13)
    # add c0*H0
    g[0] += c0 * _H0_0
    # add c1*H4
    for i in range(len(_H4)):
        g[i] += c1 * _H4[i]
    # add c2*H8
    for i in range(len(_H8)):
        g[i] += c2 * _H8[i]
    # add c3*H12
    for i in range(len(_H12)):
        g[i] += c3 * _H12[i]
    return g


def _q_coeffs(c0, c1, c2):
    """q(x) = g(x)/x^2. Since g(0)=0, g = x * (degree 11 poly), so q = degree 10 poly.
    g[0] should be 0 (g(0)=0). q coefficients = g[1:13] shifted down by 1."""
    g = _g_coeffs(c0, c1, c2)
    # g[0] is g(0) = 0, so q = g/x^2 means g = x^2 * q, so q[i] = g[i+2]
    q = g[2:13]  # degree 10
    return q


def _largest_sign_changing_root(q):
    """Find the largest real root of q at which q changes sign.
    Return the root value, or None if no sign-changing root."""
    # Find all real roots
    try:
        roots = np.roots(q)
    except Exception:
        return None
    real_roots = roots[np.abs(roots.imag) < 1e-10]
    real_roots = real_roots.real
    if len(real_roots) == 0:
        return None
    # For each real root, check if q changes sign there.
    # q changes sign at root r if the multiplicity is odd, or equivalently
    # q(r - eps) and q(r + eps) have opposite signs.
    best = None
    for r in real_roots:
        # Check sign change
        eps = 1e-8 * max(1.0, abs(r))
        q_minus = np.polyval(q, r - eps)
        q_plus = np.polyval(q, r + eps)
        if q_minus * q_plus < 0:
            # Sign change at this root
            if best is None or r > best:
                best = r
    return best


def _compute_value(c0, c1, c2):
    """Compute value = r^2 / (2*pi). Return inf if invalid."""
    q = _q_coeffs(c0, c1, c2)
    # Check leading coefficient is nonzero (q must be degree 10)
    if abs(q[0]) < 1e-30:
        return float('inf')
    r = _largest_sign_changing_root(q)
    if r is None:
        return float('inf')
    return r * r / (2.0 * math.pi)


def _seed_value():
    return _compute_value(-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0)


def construct():
    deadline = time.time() + 180.0  # safety margin before 210s kill

    # Start from seed
    best = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _seed_value()

    # Local search with random perturbations and adaptive step size
    rng = np.random.default_rng(42)
    step = 0.05

    for iteration in range(100000):
        if time.time() > deadline - 10.0:
            break

        # Generate a random perturbation
        direction = rng.standard_normal(3)
        direction /= (np.linalg.norm(direction) + 1e-15)
        candidate = best + step * direction

        val = _compute_value(candidate[0], candidate[1], candidate[2])

        if val < best_val:
            best = candidate
            best_val = val
            # Slightly increase step to explore
            step = min(step * 1.05, 0.5)
        else:
            # Decrease step
            step = max(step * 0.95, 1e-6)

    return best
# EVOLVE-BLOCK-END
