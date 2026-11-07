"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

# ---- Seed coefficients and its (known) value, used as a safe fallback ----
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])


def _q_coefficients(c0, c1, c2):
    """Return coefficients [b1..b6] of q(y) = b1 + b2 y + ... + b6 y^5, y = x^2.

    Uses physicists' Hermite polynomials:
      H0 = 1
      H4 = 16 x^4 - 48 x^2 + 12
      H8 = 256 x^8 - 4608 x^6 + 26880 x^4 - 40320 x^2 + 168
      H12= 4096 x^12 - 142464 x^10 + 1424640 x^8 - 6005760 x^6
           + 9072000 x^4 - 5109120 x^2 + 18480
    g = c0 H0 + c1 H4 + c2 H8 + c3 H12,  c3 = -(c0 + 12 c1 + 168 c2)/18480.
    q = g / x^2 = g(y)/y  with y = x^2.
    """
    c3 = -(c0 + 12.0 * c1 + 168.0 * c2) / 18480.0
    b1 = -48.0 * c1 - 40320.0 * c2 - 5109120.0 * c3      # y^0
    b2 = 16.0 * c1 + 26880.0 * c2 + 9072000.0 * c3       # y^1
    b3 = -4608.0 * c2 - 6005760.0 * c3                   # y^2
    b4 = 256.0 * c2 + 1424640.0 * c3                     # y^3
    b5 = -142464.0 * c3                                  # y^4
    b6 = 4096.0 * c3                                     # y^5
    return np.array([b1, b2, b3, b4, b5, b6], dtype=np.float64)


def _q_eval(qc, y):
    """Evaluate q(y) using Horner's method."""
    return (((((qc[5] * y + qc[4]) * y + qc[3]) * y + qc[2]) * y + qc[1]) * y + qc[0])


def _is_sign_change_root(qc, r):
    """True if r is a real root of q at which q changes sign (odd multiplicity)."""
    scale = max(1.0, abs(r))
    delta = 1e-6 * scale
    lo = _q_eval(qc, r - delta)
    hi = _q_eval(qc, r + delta)
    return lo * hi < 0.0


def _value_from_coeffs(c):
    """Compute value = r^2 / (2*pi) for the largest sign-changing real root of q.
    Falls back to the seed value on any numerical difficulty."""
    qc = _q_coefficients(c[0], c[1], c[2])
    if abs(qc[5]) < 1e-300:
        return _SEED_VALUE
    coeffs_desc = qc[::-1]
    try:
        roots = np.roots(coeffs_desc)
    except Exception:
        return _SEED_VALUE
    best_r = None
    tol = 1e-7
    for rt in roots:
        if abs(rt.imag) > tol * max(1.0, abs(rt.real)):
            continue
        r = float(rt.real)
        if r < -1e-9:
            continue
        if _is_sign_change_root(qc, r):
            if best_r is None or r > best_r:
                best_r = r
    if best_r is None:
        return _SEED_VALUE
    return (best_r * best_r) / (2.0 * np.pi)


_SEED_VALUE = _value_from_coeffs(_SEED)


def _objective(c):
    try:
        v = _value_from_coeffs(c)
    except Exception:
        v = _SEED_VALUE
    if not np.isfinite(v):
        v = _SEED_VALUE
    return v


def construct():
    deadline = time.time() + 180.0
    best_c = _SEED.copy()
    best_v = _SEED_VALUE

    def _improve(candidate):
        nonlocal best_c, best_v
        v = _objective(candidate)
        if v < best_v - 1e-15:
            best_v = v
            best_c = np.array(candidate, dtype=np.float64)

    # Strategy 1: Differential evolution for global search
    # Wider bounds to explore more of the coefficient space
    bounds = [(-2.0, 2.0), (-1.0, 1.0), (-0.05, 0.05)]

    if time.time() < deadline - 100.0:
        try:
            de_result = differential_evolution(
                _objective, bounds, seed=42, maxiter=300,
                popsize=20, tol=1e-10, mutation=(0.5, 1.0),
                recombination=0.8, polish=False, init='latinhypercube',
                updating='deferred'
            )
            if de_result.x is not None:
                _improve(de_result.x)
        except Exception:
            pass

    # Strategy 2: Local refinement with Nelder-Mead from multiple diverse starts
    rng = np.random.default_rng(123)
    starts = [best_c.copy()]
    # Add starts around the seed with various scales
    for scale in (0.01, 0.05, 0.1, 0.2, 0.5):
        for _ in range(4):
            starts.append(_SEED + scale * rng.standard_normal(3))
    # Add starts around the current best with various scales
    for scale in (0.01, 0.05, 0.1, 0.2):
        for _ in range(4):
            starts.append(best_c + scale * rng.standard_normal(3))

    for s in starts:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(_objective, s, method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-14,
                                    "fatol": 1e-16, "adaptive": True})
        except Exception:
            continue
        if time.time() > deadline - 20.0:
            break
        if res.x is not None:
            _improve(res.x)
            # Second refinement pass with tighter tolerance
            if time.time() < deadline - 15.0:
                try:
                    res2 = minimize(_objective, res.x, method="Nelder-Mead",
                                    options={"maxiter": 5000, "xatol": 1e-15,
                                             "fatol": 1e-17, "adaptive": True})
                except Exception:
                    res2 = None
                if res2 is not None and res2.x is not None:
                    _improve(res2.x)
                    # Third refinement pass
                    if time.time() < deadline - 10.0:
                        try:
                            res3 = minimize(_objective, res2.x, method="Nelder-Mead",
                                            options={"maxiter": 5000, "xatol": 1e-16,
                                                     "fatol": 1e-18, "adaptive": True})
                        except Exception:
                            res3 = None
                        if res3 is not None and res3.x is not None:
                            _improve(res3.x)

    # Final safety check
    if best_v > _SEED_VALUE + 1e-15:
        best_c = _SEED.copy()

    return np.array(best_c, dtype=np.float64)
# EVOLVE-BLOCK-END
