# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 630.0
_EPS = 1e-6
_TOL = 1e-9
_RNG = np.random.default_rng(20240517)


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    t = a + np.deg2rad(60.0 * np.arange(6))
    return np.stack([cx + s * np.cos(t), cy + s * np.sin(t)], axis=1)


def _pair_sep(A, B):
    # A, B: (6,2) vertex arrays. Returns max over 12 edge normals of
    # (maxA - minB) and (maxB - minA). Positive => separated.
    A2 = (A + np.roll(A, -1, axis=0)) / 2.0
    B2 = (B + np.roll(B, -1, axis=0)) / 2.0
    nA = np.empty((6, 2))
    nA[:, 0] = A2[:, 1] - A[:, 1]
    nA[:, 1] = A[:, 0] - A2[:, 0]
    nA /= np.linalg.norm(nA, axis=1, keepdims=True)
    nB = np.empty((6, 2))
    nB[:, 0] = B2[:, 1] - B[:, 1]
    nB[:, 1] = B[:, 0] - B2[:, 0]
    nB /= np.linalg.norm(nB, axis=1, keepdims=True)
    aA = A @ nA.T
    aB = B @ nA.T
    bA = A @ nB.T
    bB = B @ nB.T
    return np.maximum(aA.max(axis=0) - aB.min(axis=0),
                      bB.max(axis=0) - bA.min(axis=0))


def _separation_deficit(x):
    # x: (11,3). Returns (11,11) matrix of deficit (0 = touching/ok).
    V = np.empty((11, 6, 2))
    for i in range(11):
        V[i] = _hex_vertices(x[i, 0], x[i, 1], 1.0, x[i, 2])
    D = np.zeros((11, 11))
    for i in range(11):
        for j in range(i + 1, 11):
            sep = _pair_sep(V[i], V[j])
            d = _EPS - sep.max()
            D[i, j] = D[j, i] = d
    return D


def _min_side_for_containment(x, cx, cy, ang):
    V = np.empty((11, 6, 2))
    for i in range(11):
        V[i] = _hex_vertices(x[i, 0], x[i, 1], 1.0, x[i, 2])
    a = np.deg2rad(ang)
    t = a + np.deg2rad(60.0 * np.arange(6))
    O = np.stack([cx + np.cos(t), cy + np.sin(t)], axis=1)
    O2 = (O + np.roll(O, -1, axis=0)) / 2.0
    n = np.empty((6, 2))
    n[:, 0] = O2[:, 1] - O[:, 1]
    n[:, 1] = O[:, 0] - O2[:, 0]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    # For each inner vertex p and each outer edge normal n_k, required S:
    # p.n <= O2_k.n + S n_k.n  => S >= (p.n - O2_k.n)/n_k.n  (n_k.n>0)
    # n_k.n == cos(a_k - ang) > 0 for all k.
    req = np.zeros((11, 6, 6))
    for i in range(11):
        pn = V[i] @ n.T                       # (6,6)
        o2n = O2 @ n                          # (6,)
        diff = pn - o2n[None, :]
        req[i] = diff / n[None, :]            # (6,6)
    return req.max(), O, n, req


def _objective(x, cx, cy, ang):
    # Returns (S, violation). violation>0 means invalid.
    D = _separation_deficit(x)
    sep_bad = D.max()
    S, O, n, req = _min_side_for_containment(x, cx, cy, ang)
    # containment margin: all req <= S - tol
    cont_bad = (req.max() - S) + _TOL
    viol = max(sep_bad, cont_bad)
    return S, viol


def _honeycomb_seed():
    d = np.sqrt(3.0)
    pts = [(0, 0)]
    for i in range(6):
        a = np.deg2rad(60.0 * i)
        pts.append((d * np.cos(a), d * np.sin(a)))
    for i in range(6):
        a = np.deg2rad(60.0 * i + 30.0)
        pts.append((2.0 * d * np.cos(a), 2.0 * d * np.sin(a)))
    x = np.zeros((11, 3))
    for k, (px, py) in enumerate(pts):
        x[k, 0] = px
        x[k, 1] = py
    return x


def _optimize_from(x0, cx, cy, ang, max_iter):
    def f(z):
        xx = z.reshape(11, 3)
        S, viol = _objective(xx, cx, cy, ang)
        return S + 1e4 * viol
    res = minimize(f, x0.ravel(), method='L-BFGS-B',
                   options={'maxiter': max_iter, 'maxfun': max_iter * 3,
                            'ftol': 1e-12, 'gtol': 1e-8})
    xx = res.x.reshape(11, 3)
    S, viol = _objective(xx, cx, cy, ang)
    return xx, S, viol, res


def hexagon_packing_11():
    t0 = time.time()
    x0 = _honeycomb_seed()
    cx, cy, ang = 0.0, 0.0, 0.0

    best_x, best_S, best_viol = x0.copy(), np.inf, np.inf
    # First: refine the seed
    xx, S, viol, _ = _optimize_from(x0, cx, cy, ang, max_iter=200)
    if viol <= 1e-7 and S < best_S:
        best_x, best_S, best_viol = xx, S, viol

    # Random restarts: perturb positions/angles around the best
    n_restarts = 12
    for r in range(n_restarts):
        if time.time() - t0 > _DEADLINE - 20.0:
            break
        scale = 0.6 + 0.15 * r
        noise = _RNG.normal(0.0, scale, size=(11, 3))
        noise[:, 2] *= 30.0  # angle in degrees, more spread
        x_try = best_x + noise
        try:
            xx, S, viol, _ = _optimize_from(x_try, cx, cy, ang, max_iter=120)
        except Exception:
            continue
        if viol <= 1e-7 and S < best_S:
            best_x, best_S, best_viol = xx, S, viol

    # Final: re-center the inner cluster at the outer center for the output
    # (outer is at origin). Keep best_x as is; outer centered at (0,0,0).
    outer = np.array([0.0, 0.0, 0.0])
    # Use the minimal side that contains best_x at the current outer.
    S, O, n, req = _min_side_for_containment(best_x, 0.0, 0.0, 0.0)
    # Ensure a tiny safety margin so containment is strict
    S = max(S + 1e-5, np.sqrt(11.0) + 1e-5)

    return best_x, outer, float(S)


# EVOLVE-BLOCK-END
