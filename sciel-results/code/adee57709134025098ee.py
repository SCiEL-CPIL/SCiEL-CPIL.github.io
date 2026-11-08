"""Rewritten program: construct 12 positive double-root locations for the
Laguerre combination (m=12, n=1) and return them so the C4 upper bound is
minimized. The seed's equal-spaced roots are invalid; the valid Laguerre
construction uses the zeros of the Laguerre polynomial of degree m."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = 600.0  # wall-clock guard; evaluator kills at 630s, keep margin


def _laguerre_zeros(m):
    """Return the m positive zeros of the Laguerre polynomial L_m, sorted."""
    try:
        from scipy.special import roots_laguerre
        # roots_laguerre(m) returns (x, w) where x are the m roots of L_m
        x, _ = roots_laguerre(m)
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        x = x[x > 0.0]
        x = np.sort(x)
        if x.size == m:
            return x
    except Exception:
        pass
    # Fallback: build L_m via numpy and take its roots
    try:
        from numpy.polynomial.laguerre import laggauss, lagval
        # Use the recurrence to form L_m coefficients in monomial basis is
        # unstable; instead use numpy's laguerre namespace.
        from numpy.polynomial import laguerre as _lag
        # _lag.lag2coef gives monomial coefficients of L_m
        c = _lag.lag2coef([0.0] * (m - 1) + [1.0])
        r = np.roots(c)
        r = np.real(r[np.abs(np.imag(r)) < 1e-8])
        r = r[np.isfinite(r)]
        r = r[r > 0.0]
        r = np.sort(r)
        if r.size == m:
            return r
    except Exception:
        pass
    return None


def construct():
    """Return a list of 12 positive double-root locations (m=12, n=1)."""
    m = 12
    best = None
    # Primary: exact Laguerre zeros (high precision, fast, deterministic).
    x = _laguerre_zeros(m)
    if x is not None and x.size == m:
        best = [float(v) for v in x]

    # Optional refinement / polish within the same family: the Laguerre zeros
    # already satisfy the root constraints exactly; a Newton polish on the
    # Laguerre polynomial keeps them valid while reducing any residual.
    if best is not None:
        try:
            from scipy.special import eval_laguerre, eval_genlaguerre
            # L_m'(x) = -L_{m+1}^{(1)}(x)  (derivative identity)
            def f(xv):
                return float(eval_laguerre(m, xv))
            def fp(xv):
                return float(-eval_genlaguerre(m, 1.0, xv))
            polished = []
            for v in best:
                x0 = float(v)
                for _ in range(8):
                    fv = f(x0)
                    if not np.isfinite(fv):
                        break
                    if abs(fv) < 1e-15:
                        break
                    d = fp(x0)
                    if not np.isfinite(d) or abs(d) < 1e-300:
                        break
                    step = fv / d
                    x1 = x0 - step
                    if not np.isfinite(x1) or x1 <= 0.0:
                        break
                    x0 = x1
                if np.isfinite(x0) and x0 > 0.0:
                    polished.append(x0)
            if len(polished) == m:
                polished.sort()
                best = polished
        except Exception:
            pass

    if best is None:
        # Last-resort fallback: a strictly positive, increasing set that is at
        # least structurally valid (positive, distinct). Prefer Laguerre-like
        # spacing to keep the C4 bound small.
        best = [float((i + 1) * (i + 1) * 0.5 + 1.0) for i in range(m)]

    # Deadline guard: ensure we return well before the kill.
    if time.time() - _START > _DEADLINE - 30.0:
        pass  # we already have a result; just return it

    return best
# EVOLVE-BLOCK-END
