"""Search for coefficients minimizing r^2/(2*pi) where r is the largest
sign-changing real root of q = g/x^2, subject to g >= 0."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import hermite


def _hermite_coeffs(n):
    """Return coefficients of physicists' Hermite polynomial H_n(x) in ascending powers."""
    # Use recurrence: H_0=1, H_1=2x, H_{n+1}=2x*H_n - 2n*H_{n-1}
    coeffs = np.zeros(n + 1)
    if n == 0:
        coeffs[0] = 1.0
        return coeffs
    if n == 1:
        coeffs[1] = 2.0
        return coeffs
    h_prev = np.array([1.0])  # H_0
    h_curr = np.array([0.0, 2.0])  # H_1
    for k in range(2, n + 1):
        h_next = np.zeros(k + 1)
        # 2x * h_curr
        h_next[1:] += 2.0 * h_curr
        # -2k * h_prev
        h_next[:len(h_prev)] -= 2.0 * k * h_prev
        h_prev, h_curr = h_curr, h_next
    return h_curr


def _build_q(c0, c1, c2):
    """Return coefficients of q(x)=g(x)/x^2 (ascending powers) as a float array."""
    c0 = float(c0); c1 = float(c1); c2 = float(c2)
    H0 = _hermite_coeffs(0)
    H4 = _hermite_coeffs(4)
    H8 = _hermite_coeffs(8)
    H12 = _hermite_coeffs(12)
    
    # g(0) = c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
    H0_0 = H0[0]
    H4_0 = H4[0]
    H8_0 = H8[0]
    H12_0 = H12[0]
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # q = g / x^2, so we need g to be divisible by x^2
    # Build g coefficients (degree up to 12)
    g = np.zeros(13)
    g[:len(H0)] += c0 * H0
    g[:len(H4)] += c1 * H4
    g[:len(H8)] += c2 * H8
    g[:len(H12)] += c3 * H12
    
    # q = g / x^2, so q[i] = g[i+2]
    q = np.zeros(11)
    q[:] = g[2:]
    return q


def _signchange_root(q):
    """Largest real root of q where q changes sign (simple root)."""
    # Remove leading zeros
    nz = np.nonzero(q)[0]
    if len(nz) == 0:
        return None
    q_trimmed = q[nz[0]:]
    roots = np.roots(q_trimmed)
    real = [z.real for z in roots if abs(z.imag) < 1e-10 * max(1.0, abs(z))]
    if not real:
        return None
    best = None
    for r in real:
        # Check sign change: evaluate q at r-eps and r+eps
        eps = 1e-8
        q_left = np.polyval(q, r - eps)
        q_right = np.polyval(q, r + eps)
        if q_left * q_right < 0:  # sign change
            if best is None or r > best:
                best = r
    return best


def _objective(c):
    """Smooth objective: (signchange root)^2 + penalty for g<0 or negative limit."""
    q = _build_q(c[0], c[1], c[2])
    H12 = 256.0 * c[0] + 16.0 * c[1] + 4.0 * c[2] + 1.0
    c3 = -(c[0] + c[1] * (16.0 * c[0] + 1.0) + c[2] * (64.0 * c[0] + 4.0 * c[1] + 1.0)) / H12
    # leading coefficient of g = c3*256 ; if g-> -inf at +inf, evaluator flips g
    g_lead = c3 * 256.0
    if g_lead < 0:
        # -g is used; its leading coeff is -g_lead > 0
        g_lead_eff = -g_lead
    else:
        g_lead_eff = g_lead

    # candidate roots (sign-changing)
    r = _signchange_root(q)
    if r is None:
        r = 0.0
    obj = r * r

    # penalty: g must be >= 0 (or its sign-flipped version) on a grid
    xs = np.linspace(0.0, 16.0, 400)
    g = np.polyval(q, xs) * xs * xs  # g = q * x^2
    if g_lead < 0:
        g = -g
    mn = float(np.min(g))
    if mn < 0:
        obj += 10.0 * (mn / max(1.0, abs(r))) ** 2
    # keep leading coefficient positive (so g -> +inf, no flip needed)
    if g_lead < 0:
        obj += 5.0 * (g_lead / 1.0) ** 2
    return obj


def _value(c):
    """Exact value = r^2/(2*pi) for the sign-changing root; None if invalid."""
    q = _build_q(c[0], c[1], c[2])
    H12 = 256.0 * c[0] + 16.0 * c[1] + 4.0 * c[2] + 1.0
    c3 = -(c[0] + c[1] * (16.0 * c[0] + 1.0) + c[2] * (64.0 * c[0] + 4.0 * c[1] + 1.0)) / H12
    g_lead = c3 * 256.0
    r = _signchange_root(q)
    if r is None:
        return None
    xs = np.linspace(0.0, 16.0, 600)
    g = np.polyval(q, xs) * xs * xs
    if g_lead < 0:
        g = -g
    if float(np.min(g)) < -1e-9:
        return None
    return r * r / (2.0 * np.pi)


def construct():
    t0 = time.time()
    deadline = t0 + 185.0  # safety margin under 210s kill

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_val = _value(seed)
    if seed_val is None:
        seed_val = 1.0 / (2.0 * np.pi)  # r=1
    best_c = seed.copy()
    best_val = seed_val

    def _try_update(c):
        nonlocal best_c, best_val
        v = _value(c)
        if v is not None and v < best_val:
            best_c = c.copy()
            best_val = v

    # Local optimization from seed and several perturbations
    starts = [seed.copy()]
    rng = np.random.default_rng(12345)
    for _ in range(24):
        starts.append(seed + rng.normal(0.0, 0.15, size=3))
    # a few structured starts near the non-negative boundary
    for s in [0.5, 1.0, 2.0, 3.0, 5.0]:
        starts.append(np.array([-0.1, s, 0.02]))
        starts.append(np.array([0.1, -s, 0.02]))

    for x0 in starts:
        if time.time() > deadline - 15.0:
            break
        try:
            res = minimize(_objective, x0, method="Nelder-Mead",
                           options={"maxiter": 800, "xatol": 1e-9, "fatol": 1e-11,
                                    "adaptive": True})
            _try_update(res.x)
            # polish with BFGS if finite
            if time.time() > deadline - 15.0:
                break
            try:
                res2 = minimize(_objective, res.x, method="L-BFGS-B",
                                options={"maxiter": 300, "gtol": 1e-12})
                _try_update(res2.x)
            except Exception:
                pass
        except Exception:
            continue

    # If nothing improved, return seed
    if best_val >= seed_val:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    return np.array(best_c, dtype=float)
# EVOLVE-BLOCK-END
