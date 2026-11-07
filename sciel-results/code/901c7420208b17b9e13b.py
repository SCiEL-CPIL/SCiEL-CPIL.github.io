"""Structural redesign: optimize (c1,c2,c3) directly with a global + local
optimizer over the analytic q(y)=g(y)/y, minimizing the largest sign-change root."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 185.0  # safety margin under the 210s kill
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_PI = np.pi


def _q_coeffs(c1, c2, c3):
    """Return [b1..b5] of q(y) = b1 + b2 y + b3 y^2 + b4 y^3 + b5 y^4 + b6 y^5."""
    b1 = -48.0 * c1 - 40320.0 * c2 - 5109120.0 * c3
    b2 = 16.0 * c1 + 26880.0 * c2 + 9072000.0 * c3
    b3 = -4608.0 * c2 - 6005760.0 * c3
    b4 = 256.0 * c2 + 1424640.0 * c3
    b5 = -142464.0 * c3
    b6 = 4096.0 * c3
    return np.array([b1, b2, b3, b4, b5, b6])


def _largest_sign_change_root(coeffs, y_max=1e12):
    """Largest real root of q(y) at which q changes sign, else None.
    coeffs = [b1..b6] (ascending), q(y)=sum_{k=1..6} b_k y^{k-1}."""
    # Find all real roots via companion matrix (robust for degree 5).
    a = coeffs[::-1]  # descending powers: [b6,b5,b4,b3,b2,b1]
    # Build companion matrix for monic polynomial.
    lead = a[0]
    if abs(lead) < 1e-300:
        # drop leading zeros
        while len(a) > 1 and abs(a[0]) < 1e-300:
            a = a[1:]
        lead = a[0]
        if abs(lead) < 1e-300:
            return None
    m = len(a) - 1
    if m == 0:
        return None
    # monic coefficients
    cm = a[1:] / lead  # [b5,b4,b3,b2,b1]/b6
    # companion: eigenvalues of
    # [[0,0,...,0,-cm[0]],
    #  [1,0,...,0,-cm[1]],
    #  [0,1,...,0,-cm[2]],
    #  ...
    #  [0,...,1,-cm[m-1]]]
    comp = np.zeros((m, m))
    comp[np.arange(1, m), np.arange(0, m - 1)] = 1.0
    comp[:, m - 1] = -cm
    try:
        roots = np.linalg.eigvals(comp)
    except np.linalg.LinAlgError:
        return None
    # keep real roots
    real_roots = []
    for rt in roots:
        if abs(rt.imag) <= 1e-7 * max(1.0, abs(rt.real)):
            real_roots.append(rt.real)
    if not real_roots:
        return None
    real_roots = np.array(real_roots)
    # For each candidate root, check sign change using high-precision-ish eval
    # via numpy poly on the original (non-monic) ascending coeffs.
    def qeval(y):
        return np.polyval(a, y)  # a descending
    best = None
    for rt in real_roots:
        if rt <= 0:
            continue
        # sample around the root to detect sign change
        h = max(1e-6 * max(1.0, rt), 1e-9)
        lo = qeval(rt - h)
        hi = qeval(rt + h)
        if lo == 0.0:
            lo = qeval(rt - 2 * h)
        if hi == 0.0:
            hi = qeval(rt + 2 * h)
        if lo == 0.0 or hi == 0.0:
            # refine: try smaller offset
            for hh in (1e-4, 1e-3, 1e-2):
                lo2 = qeval(rt - hh * max(1.0, rt))
                hi2 = qeval(rt + hh * max(1.0, rt))
                if lo2 * hi2 < 0:
                    lo, hi = lo2, hi2
                    break
        if lo * hi < 0:
            # refine root with bisection for accuracy
            a0, b0 = rt - h, rt + h
            fa, fb = lo, hi
            if fa * fb > 0:
                continue
            for _ in range(60):
                mid = 0.5 * (a0 + b0)
                fm = qeval(mid)
                if fm == 0.0:
                    a0 = b0 = mid
                    break
                if fa * fm < 0:
                    b0 = mid
                    fb = fm
                else:
                    a0 = mid
                    fa = fm
            rt_ref = 0.5 * (a0 + b0)
            if best is None or rt_ref > best:
                best = rt_ref
    return best


def _value_from_c(c1, c2, c3):
    coeffs = _q_coeffs(c1, c2, c3)
    r = _largest_sign_change_root(coeffs)
    if r is None:
        return np.inf
    return r / (2.0 * _PI)


def _seed_value():
    c1, c2 = _SEED[1], _SEED[2]
    c0 = _SEED[0]
    c3 = -(c0 + 12.0 * c1 + 168.0 * c2) / 18480.0
    return _value_from_c(c1, c2, c3)


def _objective(v):
    c1, c2, c3 = v
    return _value_from_c(c1, c2, c3)


def construct():
    global _T0
    start = time.time()
    # Seed baseline
    best_v = _seed_value()
    best_coeffs = _SEED.copy()

    c1s, c2s = _SEED[1], _SEED[2]
    c0s = _SEED[0]
    c3s = -(c0s + 12.0 * c1s + 168.0 * c2s) / 18480.0

    # Local Nelder-Mead around the seed (in (c1,c2,c3) space)
    from scipy.optimize import minimize, differential_evolution

    try:
        res = minimize(_objective, x0=np.array([c1s, c2s, c3s]),
                       method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-12, 'fatol': 1e-14,
                                'adaptive': True})
        if res.success and np.isfinite(res.fun) and res.fun < best_v:
            best_v = res.fun
            v = res.x
            c1, c2, c3 = v
            # reconstruct full coefficients with arbitrary c0 (set 0)
            c0 = 0.0
            best_coeffs = np.array([c0, c1, c2])
    except Exception:
        pass

    # Global differential evolution over a wide box
    if time.time() - start < _DEADLINE - 30:
        bounds = [(-50.0, 50.0), (-50.0, 50.0), (-0.05, 0.05)]
        try:
            de = differential_evolution(_objective, bounds,
                                        seed=12345,
                                        maxiter=200,
                                        popsize=25,
                                        tol=1e-10,
                                        mutation=(0.3, 1.0),
                                        recombination=0.8,
                                        polish=True,
                                        init='sobol',
                                        updating='deferred',
                                        seed=12345)
            if de.success and np.isfinite(de.fun) and de.fun < best_v:
                best_v = de.fun
                c1, c2, c3 = de.x
                c0 = 0.0
                best_coeffs = np.array([c0, c1, c2])
        except Exception:
            pass

    # Final polish with Nelder-Mead from the best found
    if time.time() - start < _DEADLINE - 5:
        try:
            c1, c2 = best_coeffs[1], best_coeffs[2]
            c0 = best_coeffs[0]
            c3 = -(c0 + 12.0 * c1 + 168.0 * c2) / 18480.0
            res2 = minimize(_objective, x0=np.array([c1, c2, c3]),
                            method='Nelder-Mead',
                            options={'maxiter': 6000, 'xatol': 1e-14, 'fatol': 1e-16,
                                     'adaptive': True})
            if np.isfinite(res2.fun) and res2.fun < best_v:
                best_v = res2.fun
                c1, c2, c3 = res2.x
                c0 = 0.0
                best_coeffs = np.array([c0, c1, c2])
        except Exception:
            pass

    return best_coeffs
# EVOLVE-BLOCK-END
