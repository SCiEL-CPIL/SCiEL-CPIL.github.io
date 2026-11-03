"""Optimized logarithmic-energy placement of N points on the unit sphere."""
import time
import math
import numpy as np

N = 150
DEADLINE = 150.0
SAFETY = 6.0
EPS = 1e-9


def _energy(points):
    pts = points
    d2 = (pts @ pts.T).diagonal()[:, None] + (pts @ pts.T).diagonal()[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(d2, 0.0)
    np.maximum(d2, 1e-300, out=d2)
    dist = np.sqrt(d2)
    iu = np.triu_indices(N, k=1)
    return math.fsum(np.log(1.0 / dist[iu]).tolist())


def _energy_grad(points):
    pts = points
    d2 = (pts @ pts.T).diagonal()[:, None] + (pts @ pts.T).diagonal()[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(d2, 0.0)
    np.maximum(d2, 1e-300, out=d2)
    dist = np.sqrt(d2)
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    rowsum = inv_d2.sum(axis=1)
    colpts = (inv_d2[:, :, None] * pts[None, :, :]).sum(axis=1)
    grad = rowsum[:, None] * pts - colpts
    iu = np.triu_indices(N, k=1)
    energy = math.fsum(np.log(1.0 / dist[iu]).tolist())
    return energy, grad


def _project(pts, grad):
    return grad - pts * (pts @ grad).T


def _renormalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n = np.maximum(n, EPS)
    return pts / n


def _fibonacci_start(rng):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    return Q @ pts.T


def _design_start(rng):
    pts = rng.normal(size=(N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    for _ in range(3):
        for i in range(N):
            d = np.linalg.norm(pts[i][None, :] - pts, axis=1)
            d[i] = -1
            j = int(np.argmin(d))
            direction = pts[i] - pts[j]
            pts[i] = pts[i] + 0.02 * direction
            pts[i] /= np.linalg.norm(pts[i])
    return pts


def _grad_step(pts, step, momentum, vel):
    energy, grad = _energy_grad(pts)
    g = _project(pts, grad)
    gn = np.linalg.norm(g)
    if gn < 1e-14:
        return energy, pts, 0.0, vel
    g = g / gn
    if momentum > 0.0:
        vel = momentum * vel + (1.0 - momentum) * g
        g = vel
        g /= (np.linalg.norm(g) + 1e-14)
    new_pts = _renormalize(pts + step * g)
    return energy, new_pts, gn, vel


def _metropolis_step(pts, energy, T, rng, walk_sigma=0.05):
    d = rng.normal(size=(N, 3))
    d = d - pts * (pts @ d).T
    d /= (np.linalg.norm(d) + 1e-14)
    new_pts = _renormalize(pts + walk_sigma * d)
    new_energy, _ = _energy_grad(new_pts)
    delta = new_energy - energy
    if delta < 0 or rng.random() < math.exp(-delta / T):
        return new_pts, new_energy
    return pts, energy


def construct():
    t0 = time.time()
    deadline = t0 + (DEADLINE - SAFETY)
    rng = np.random.default_rng(12345)

    n_rep = 6
    starts = [_fibonacci_start(rng), _design_start(rng)]
    replicas = []
    for k in range(n_rep):
        if k < len(starts):
            pts = starts[k]
        else:
            pts = _fibonacci_start(rng)
        e, _ = _energy_grad(pts)
        if k % 3 == 2:
            T, mode = 0.5, "hot"
        else:
            T, mode = 0.0, "cold"
        replicas.append([pts, e, T, mode, np.zeros((N, 3))])

    best = min(r[1] for r in replicas)
    best_pts = replicas[int(np.argmin([r[1] for r in replicas]))][0].copy()
    steps = [0.05] * n_rep

    while time.time() < deadline:
        for r in range(n_rep):
            if time.time() > deadline:
                break
            pts, e, T, mode, vel = replicas[r]
            if mode == "cold":
                e, new_pts, gn, vel = _grad_step(pts, steps[r], 0.9, vel)
                replicas[r][0] = new_pts
                replicas[r][4] = vel
                if gn > 1.0:
                    steps[r] = max(0.01, steps[r] * 0.7)
                else:
                    steps[r] = min(0.2, steps[r] * 1.05)
                if rng.random() < 0.02:
                    replicas[r][2] = 0.5
                    replicas[r][3] = "hot"
            else:
                new_pts, e = _metropolis_step(pts, e, T, rng)
                replicas[r][0] = new_pts
                replicas[r][1] = e
                replicas[r][2] = max(0.0, T * 0.999)
                if replicas[r][2] < 0.05 and rng.random() < 0.5:
                    replicas[r][3] = "cold"
                    replicas[r][2] = 0.0
                    replicas[r][4] = np.zeros((N, 3))

        if rng.random() < 0.3:
            i = int(rng.integers(0, n_rep))
            j = int(rng.integers(0, n_rep))
            if i != j:
                ei, ej = replicas[i][1], replicas[j][1]
                Ti, Tj = replicas[i][2], replicas[j][2]
                accept = False
                if Ti > 0 and Tj > 0:
                    delta = (ei / Tj + ej / Ti) - (ei / Ti + ej / Tj)
                    if rng.random() < math.exp(-delta):
                        accept = True
                elif (Ti > Tj and ei > ej) or (Tj > Ti and ej > ei):
                    accept = True
                if accept:
                    replicas[i][0], replicas[j][0] = replicas[j][0], replicas[i][0]
                    replicas[i][1], replicas[j][1] = replicas[j][1], replicas[i][1]
                    replicas[i][2], replicas[j][2] = replicas[j][2], replicas[i][2]
                    replicas[i][3], replicas[j][3] = replicas[j][3], replicas[i][3]
                    replicas[i][4], replicas[j][4] = replicas[j][4], replicas[i][4]

        for r in range(n_rep):
            if replicas[r][1] < best:
                best = replicas[r][1]
                best_pts = replicas[r][0].copy()

    pts = best_pts
    vel = np.zeros((N, 3))
    for _ in range(200):
        if time.time() > deadline:
            break
        e, pts, gn, vel = _grad_step(pts, 0.02, 0.9, vel)
        if gn < 1e-10:
            break

    return _renormalize(pts)
