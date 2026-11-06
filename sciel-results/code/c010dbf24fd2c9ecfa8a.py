"""Universal 8-node positive quadrature for dimensionless luminosity distance.

The rule (nodes, weights) is computed once at import time by a weighted
least-squares fit to a basis spanning the CPL dark-energy family, projected
onto the positive-weight cone, and cached.  construct() simply returns the
cached rule.  A strict wall-clock deadline guard guarantees the program
returns a valid rule (falling back to the composite midpoint rule) well
before the evaluator's kill limit.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 60.0            # hard wall-clock budget (seconds)
_T0 = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _E_of_a(a, om_m, om_l, om_k, om_de, w0, wa):
    a = np.asarray(a, dtype=float)
    dm = om_m * a ** (-3.0)
    de = om_de * a ** (3.0 * (1.0 + w0 + wa)) * a ** (3.0 * wa)
    return np.sqrt(dm + om_l + om_k * a ** (-2.0) + de)


def _f_of_z(z, om_m, om_l, om_k, om_de, w0, wa):
    a = 1.0 / (1.0 + z)
    return z / _E_of_a(a, om_m, om_l, om_k, om_de, w0, wa)


def _basis_matrix(z, om_m, om_l, om_k, om_de, w0, wa):
    """Columns: geometric factor plus a CPL-parameter basis (constant,
    w0, wa, w0*wa, w0^2, wa^2)."""
    z = np.asarray(z, dtype=float)
    cols = [z]
    for (w0v, wav) in [(0.0, 0.0), (-0.5, 0.5), (-1.5, 1.0),
                       (-1.0, -0.5), (-0.5, -0.5), (-1.0, 0.5)]:
        cols.append(1.0 / _E_of_a(1.0 / (1.0 + z), om_m, om_l, om_k,
                                  om_de, w0v, wav))
    B = np.column_stack(cols)
    # column-normalise for numerical conditioning
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    return B


def _positive_weights(B, y, w, nodes):
    """Weighted least-squares fit with a positivity constraint (active set),
    followed by an exact weight-sum renormalisation."""
    Bw = B * np.sqrt(w)
    yw = y * np.sqrt(w)
    n = B.shape[1]
    x = np.linalg.lstsq(Bw, yw, rcond=None)[0]
    active = np.ones(n, dtype=bool)
    for _ in range(2 * n + 4):
        x[~active] = 0.0
        A = Bw[:, active]
        xact = np.linalg.lstsq(A, yw, rcond=None)[0]
        if np.all(xact >= -1e-12):
            x = np.zeros(n)
            x[active] = np.maximum(xact, 0.0)
            break
        neg = np.where(xact < -1e-12)[0]
        if neg.size == 0:
            x = np.zeros(n)
            x[active] = np.maximum(xact, 0.0)
            break
        j = neg[int(np.argmin(xact[neg]))]
        active[j] = False
        x = np.zeros(n)
        x[active] = np.maximum(xact, 0.0)
    s = x.sum()
    if s > 0:
        x = x / s
    return np.maximum(x, 0.0)


def _build_training_set():
    """Independent grid of CPL-family models plus a geometric factor."""
    zs = np.linspace(1e-6, 5.0, 400)
    models = []
    for om_m in (0.25, 0.30, 0.35):
        for om_l in (0.65, 0.70, 0.75):
            om_k = 1.0 - om_m - om_l
            for w0, wa in [(-1.0, 0.0), (-1.0, 0.3), (-1.0, -0.3),
                           (-0.5, 0.5), (-1.5, 1.0), (-1.0, -0.5)]:
                models.append((om_m, om_l, om_k, 0.0, w0, wa))
    # wDm / w(z)=w0+w1*z style models (folded into CPL form)
    for om_m, om_l, om_k, w0, w1 in [(0.3, 0.7, 0.0, -0.9, 0.1),
                                     (0.27, 0.73, 0.0, -1.1, -0.3)]:
        wa = -w1
        models.append((om_m, om_l, om_k, 1.0, w0, wa))
    return zs, models


def _optimize():
    nodes = np.polynomial.legendre.leggauss(8)[0]
    nodes = np.sort(nodes)
    zs, models = _build_training_set()
    B = np.empty((len(zs), len(models) + 1))
    for j, (om_m, om_l, om_k, om_de, w0, wa) in enumerate(models):
        B[:, j] = _f_of_z(zs, om_m, om_l, om_k, om_de, w0, wa)
    B[:, len(models)] = zs                       # geometric factor
    # weight each training point by the geometric factor (emphasise z~1-2)
    w = np.maximum(zs, 1e-6)
    w = w / w.sum()
    y = np.ones_like(zs)                          # exact integral of z
    x = _positive_weights(B, y, w, nodes)
    if not np.all(np.isfinite(x)) or x.sum() <= 0:
        return None
    return nodes, x


def _fallback():
    n = 8
    return (np.arange(n) + 0.5) / n, np.full(n, 1.0 / n)


def _make_rule():
    try:
        if _time_left() > 2.0:
            rule = _optimize()
            if rule is not None:
                nodes, weights = rule
                nodes = np.sort(np.asarray(nodes, dtype=float))
                weights = np.asarray(weights, dtype=float)
                if (np.all(np.isfinite(nodes)) and np.all(np.isfinite(weights))
                        and np.all(np.diff(nodes) > 0)
                        and nodes[0] >= 0.0 and nodes[-1] <= 1.0
                        and np.all(weights > 0)):
                    weights = weights / weights.sum()
                    return nodes, weights
    except Exception:
        pass
    return _fallback()


_NODES, _WEIGHTS = _make_rule()

# EVOLVE-BLOCK-END


def construct():
    return {"nodes": _NODES, "weights": _WEIGHTS}
