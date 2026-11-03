"""Three hundred equal circles in a circle: soft-min smooth objective + L-BFGS-B + polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300
_T0 = time.time()
_DEADLINE = _T0 + 115.0   # safety margin before the 120s (150s hard) limit

# Precompute pair indices for vectorized distance computation
_iu, _ju = np.triu_indices(N, k=1)
_PAIR_I = _iu
_PAIR_J = _ju
_NPAIRS = len(_iu)


def _pair_diffs(P):
    return P[_iu] - P[_ju]


def _pair_dists(P):
    d = _pair_diffs(P)
    return np.einsum('ij,ij->i', d, d)


def _objective_and_grad(P, t, alpha):
    """Smooth soft-min objective and its analytic gradient.

    F = log( sum_i exp(2*max(0, 1-r_i)/t)  +  sum_{i<j} exp(2*max(0, 2r - d_ij)/t) )
    """
    r = np.linalg.norm(P, axis=1)
    viol_c = np.maximum(0.0, 1.0 - r)
    e_c = np.exp(2.0 * viol_c / t)          # (N,)

    d2 = _pair_dists(P)
    d = np.sqrt(d2)
    viol_p = np.maximum(0.0, 2.0 - d)
    e_p = np.exp(2.0 * viol_p / t)          # (NPAIRS,)

    total = e_c.sum() + e_p.sum()
    F = np.log(total + 1e-300)

    # gradient w.r.t. P
    g = np.zeros_like(P)
    # container term: dF/dP_i = (2/t) * (e_c_i/total) * (P_i / r_i)  (for r_i>0)
    w_c = (2.0 / t) * (e_c / total)         # (N,)
    safe_r = np.where(r > 1e-12, r, 1.0)
    g += w_c[:, None] * (P / safe_r[:, None])

    # pair term: for pair (i,j) with viol>0, weight w = (2/t)*(e_p/total)
    #   dF/dP_i += w * (P_i - P_j)/d ;  dF/dP_j += w * (P_j - P_i)/d
    w_p = (2.0 / t) * (e_p / total)         # (NPAIRS,)
    dd = np.where(d > 1e-12, d, 1.0)
    diff = _pair_diffs(P)                   # P_i - P_j
    contrib = (w_p / dd)[:, None] * diff    # (NPAIRS,2)
    np.add.at(g, _iu, contrib)
    np.add.at(g, _ju, -contrib)

    # scale by alpha (soft-min of radius ~ (1 - F/2)/alpha)
    return alpha * F, alpha * g


def _hex_init(jitter=0.0):
    """Hexagonal lattice cut to the unit disk, take N closest to centre."""
    rng = np.random.default_rng(12345)
    # choose spacing so we get >= N points inside
    for spacing in (0.11, 0.10, 0.095, 0.09, 0.085):
        k = int(1.0 / spacing) + 2
        pts = np.array([((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                        for i in range(-k, k + 1) for j in range(-k, k + 1)])
        rad = np.linalg.norm(pts, axis=1)
        inside = rad <= 1.0 - spacing / 2.0
        pts = pts[inside]
        if len(pts) >= N:
            if jitter > 0:
                pts = pts + jitter * rng.standard_normal(pts.shape)
            order = np.argsort(rad, kind="stable")
            return pts[order[:N]].copy()
    # fallback: jittered hex
    k = 12
    pts = np.array([((i + 0.5 * (j % 2)) * 0.08, j * 0.08 * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)])
    rad = np.linalg.norm(pts, axis=1)
    inside = rad <= 0.99
    pts = pts[inside] + 0.01 * rng.standard_normal(pts.shape)
    order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
    return pts[order[:N]].copy()


def _ring_init():
    """Rings of points from centre outward."""
    pts = [np.array([0.0, 0.0])]
    r = 0.10
    while len(pts) < N:
        m = max(1, int(2 * np.pi * r / 0.09))
        ang = np.linspace(0, 2 * np.pi, m, endpoint=False) + r
        ring = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
        pts.append(ring)
        r += 0.09
    P = np.vstack(pts)[:N]
    return P


def _score(P):
    """Raw packing radius (the value the evaluator uses)."""
    r = np.linalg.norm(P, axis=1)
    d2 = _pair_dists(P)
    dmin = np.sqrt(np.min(d2)) if _NPAIRS else np.inf
    return min(dmin / 2.0, 1.0 - np.max(r))


def _polish(P, t, alpha):
    """High-precision polish: tighter tolerance, more iterations, exact local solve."""
    best = P
    bestF = _objective_and_grad(P, t, alpha)[0]
    for _ in range(4):
        res = minimize(_objective_and_grad, best, jac=True, method="L-BFGS-B",
                       args=(t, alpha),
                       options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-9, "maxls": 80})
        if res.x is not None and np.all(np.isfinite(res.x)):
            if res.fun < bestF:
                best, bestF = res.x, res.fun
    return best


def construct():
    global _T0, _DEADLINE
    deadline = _DEADLINE
    best = _hex_init()
    best_score = _score(best)

    def remaining():
        return deadline - time.time()

    starts = [
        ("hex", _hex_init(0.0)),
        ("hexj", _hex_init(0.004)),
        ("ring", _ring_init()),
    ]

    # main optimization with soft-min annealing
    for name, P0 in starts:
        if remaining() < 25.0:
            break
        t = 0.05
        alpha = 200.0
        P = P0.copy()
        for it in range(12):
            if remaining() < 25.0:
                break
            F, g = _objective_and_grad(P, t, alpha)
            res = minimize(_objective_and_grad, P, jac=True, method="L-BFGS-B",
                           args=(t, alpha),
                           options={"maxiter": 120, "ftol": 1e-12, "gtol": 1e-7, "maxls": 60})
            if res.x is not None and np.all(np.isfinite(res.x)):
                P = res.x
            # anneal: reduce softness, increase weight
            t = max(0.012, t * 0.80)
            alpha = min(4000.0, alpha * 1.5)

        # polish
        Pp = _polish(P, 0.012, 4000.0)
        if np.all(np.isfinite(Pp)):
            s = _score(Pp)
            if s > best_score:
                best, best_score = Pp, s

    # final high-precision polish on the best found
    if remaining() > 8.0:
        Pp = _polish(best, 0.010, 8000.0)
        if np.all(np.isfinite(Pp)) and _score(Pp) > best_score:
            best = Pp

    # ensure validity: clip any point that drifted outside (shouldn't happen)
    r = np.linalg.norm(best, axis=1)
    if np.any(r > 1.0):
        scale = np.where(r > 1.0, (0.999999) / r, 1.0)
        best = best * scale[:, None]

    return best
