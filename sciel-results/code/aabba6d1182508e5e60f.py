"""Symmetry-parameterized polytope + gradient refinement for log-energy on S^2."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100


def _energy_and_grad(P):
    """Logarithmic energy sum_{i<j} log(1/|x_i-x_j|) and its gradient (Cartesian)."""
    D = P[:, None, :] - P[None, :, :]          # (N,N,3)
    dist = np.linalg.norm(D, axis=2)           # (N,N)
    np.fill_diagonal(dist, 1.0)
    energy = float(np.sum(np.log(1.0 / dist)))
    # grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    G = D * (1.0 / (dist ** 2))[:, :, None]
    np.fill_diagonal(G, 0.0)
    grad = G.sum(axis=1)
    return energy, grad


def _project(P):
    """Project points back onto the unit sphere (safe against zero norms)."""
    n = np.linalg.norm(P, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return P / n


def _golden_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _icosahedron():
    t = (1.0 + np.sqrt(5.0)) / 2.0
    pts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _pentagon(z, r, phase):
    ang = phase + 2.0 * np.pi * np.arange(5) / 5.0
    return np.column_stack([r * np.cos(ang), r * np.sin(ang), np.full(5, z)])


def _sym_start(params):
    """100 points = icosahedron(12) + 18 pentagonal rings (90) = 100."""
    s = float(params[0])
    base = _icosahedron()
    base = base / np.linalg.norm(base, axis=1, keepdims=True) * s
    pts = [base]
    nring = 18
    for k in range(nring):
        z = 1.0 - (2.0 * (k + 0.5) / nring)
        r = np.sqrt(max(0.0, 1.0 - z * z)) * s
        pts.append(_pentagon(z, r, float(params[1]) * k))
    P = np.vstack(pts)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    return P


def _refine(P0, deadline):
    """L-BFGS-B with analytic gradient, projected onto the sphere."""
    best = _project(P0.copy())
    be, _ = _energy_and_grad(best)
    for _ in range(4):
        if time.time() > deadline:
            break
        P0 = _project(best.copy())
        try:
            res = minimize(
                _energy_and_grad, P0, jac=True,
                method="L-BFGS-B",
                options={"maxiter": 300, "gtol": 1e-8, "maxfun": 3000},
            )
        except Exception:
            break
        if res.x is None:
            break
        cand = _project(np.asarray(res.x, dtype=float))
        if not np.all(np.isfinite(cand)):
            break
        ce, _ = _energy_and_grad(cand)
        if np.isfinite(ce) and ce < be - 1e-12:
            best, be = cand, ce
        else:
            break
    return best, be


def construct():
    deadline = time.time() + 140.0  # safety margin under 150s wall-clock
    rng = np.random.default_rng(12345)

    starts = []
    starts.append(_sym_start(np.array([1.0, 0.0])))
    # a few low-dimensional symmetry deformations (twist + radial scale)
    for s in (0.96, 1.04):
        for tw in (0.0, 0.18, -0.18, 0.37):
            starts.append(_sym_start(np.array([s, tw])))
    # random restarts to escape shallow local minima
    starts.append(_golden_sphere(N))
    for _ in range(6):
        starts.append(_project(rng.standard_normal((N, 3))))

    best = None
    best_e = np.inf
    for P0 in starts:
        if time.time() > deadline:
            break
        P, e = _refine(P0, deadline)
        if e < best_e:
            best, best_e = P, e

    if best is None:
        best = _golden_sphere(N)
    best = _project(np.asarray(best, dtype=float))
    # final safety: replace any non-finite/zero row with a golden point
    if not np.all(np.isfinite(best)) or np.any(np.linalg.norm(best, axis=1) < 1e-12):
        best = _golden_sphere(N)
    return best
