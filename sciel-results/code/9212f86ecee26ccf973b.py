"""Logarithmic-energy minimization of 100 points on the unit sphere (fresh design)."""
import time
import math
import numpy as np

N = 100
DEADLINE = 150.0          # evaluator wall-clock limit
MARGIN = 10.0             # safety margin (seconds)
FLOOR = 1e-12             # distance floor for log/gradient (avoids NaNs)
EPS = 1e-9                # evaluator's minimum-pair-distance validity threshold

# Precompute the inverse of the tangential Newton operator H = 2*(I - nn^T)
# (constant across all iterations; depends only on N).
_H = 2.0 * (np.eye(N) - np.ones((N, N)) / N)
_Hinv = np.linalg.inv(_H)


def _energy(pts):
    """Logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(N, k=1)
    return float(np.sum(np.log(1.0 / np.maximum(d[iu], FLOOR))))


def _energy_grad(pts):
    """Return (energy, gradient). Gradient is the Riesz-energy force."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    d = np.maximum(d, FLOOR)
    iu = np.triu_indices(N, k=1)
    D = d[iu]
    e = float(np.sum(np.log(1.0 / D)))
    w = 1.0 / D[:, None]
    g = np.zeros((N, 3))
    np.add.at(g, iu[0], -w * (pts[iu[0]] - pts[iu[1]]))
    np.add.at(g, iu[1], +w * (pts[iu[0]] - pts[iu[1]]))
    return e, g


def _project(pts):
    """Normalize each point to the unit sphere."""
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return pts / n


def _local_opt(pts, start_time, max_iter=800):
    """Gradient flow + Newton correction; returns (best_pts, best_e)."""
    pts = _project(pts.copy())
    e, g = _energy_grad(pts)
    best_pts, best_e = pts.copy(), e
    lr = 1e-3
    for _ in range(max_iter):
        if time.time() - start_time > DEADLINE - MARGIN:
            break
        if np.max(np.abs(g)) < 1e-9:
            break
        # Newton correction: solve H * delta = g in the tangent space
        gp = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        delta = (_Hinv @ gp.reshape(-1)).reshape(N, 3)
        # Damped step (Newton when |delta| small, gradient flow otherwise)
        step = np.linalg.norm(delta)
        if step > 1e-2:
            step = delta * min(1.0, 1e-2 / step)
        cand = _project(pts - lr * gp + 1e-4 * step)
        ce, cg = _energy_grad(cand)
        if ce < e - 1e-12:
            pts, e, g = cand, ce, cg
            if e < best_e:
                best_pts, best_e = pts.copy(), e
            if lr < 1e-2:
                lr *= 1.2
        else:
            lr *= 0.5
    return best_pts, best_e


def construct():
    start_time = time.time()
    rng = np.random.default_rng(0)

    # 1) Fibonacci (seed) initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_pts, best_e = _local_opt(fib, start_time)

    # 2) Diverse random / jittered starts
    for _ in range(20):
        if time.time() - start_time > DEADLINE - MARGIN:
            break
        start = _project(rng.normal(size=(N, 3)))
        p, e = _local_opt(start, start_time)
        if e < best_e:
            best_pts, best_e = p, e

    # 3) Simulated annealing: perturb best config, re-optimize
    T = 1e-2
    for _ in range(400):
        if time.time() - start_time > DEADLINE - MARGIN:
            break
        perturbed = _project(best_pts + T * rng.normal(size=(N, 3)))
        p, e = _local_opt(perturbed, start_time, max_iter=300)
        if e < best_e:
            best_pts, best_e = p, e
            T = min(1e-2, T * 1.0)   # keep small; improvement found
        else:
            T *= 0.99                # cool slowly
            if T < 1e-5:
                T = 1e-5

    # Final polish on the best configuration
    if time.time() - start_time < DEADLINE - MARGIN:
        p, e = _local_opt(best_pts, start_time, max_iter=2000)
        if e < best_e:
            best_pts, best_e = p, e

    # Safety: enforce validity (no pair closer than EPS)
    best_pts = _project(best_pts)
    d = np.linalg.norm(best_pts[:, None, :] - best_pts[None, :, :], axis=-1)
    iu = np.triu_indices(N, k=1)
    if np.min(d[iu]) < EPS:
        # Nudge the closest pairs apart
        for _ in range(200):
            d = np.linalg.norm(best_pts[:, None, :] - best_pts[None, :, :], axis=-1)
            if np.min(d[iu]) >= EPS:
                break
            order = np.argsort(d[iu])[:5]
            a = iu[0][order]
            b = iu[1][order]
            for k in range(len(a)):
                dirv = best_pts[a[k]] - best_pts[b[k]]
                n = np.linalg.norm(dirv)
                if n > 0:
                    best_pts[a[k]] += 1e-6 * dirv / n
                    best_pts[b[k]] -= 1e-6 * dirv / n
        best_pts = _project(best_pts)

    return best_pts
