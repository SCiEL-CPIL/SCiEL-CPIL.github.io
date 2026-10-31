import math
import time
import random
import numpy as np

N = 11
TIME_LIMIT = 150.0
SAFE_MARGIN = 8.0

# ---------- Geometry ----------
def hex_normals(theta):
    a = theta * math.pi / 180.0
    n = np.empty((6, 2))
    for k in range(6):
        ang = a + k * math.pi / 3.0
        n[k] = (math.cos(ang), math.sin(ang))
    return n

def hex_vertices(cx, cy, s, theta):
    a = theta * math.pi / 180.0
    v = np.empty((6, 2))
    for k in range(6):
        ang = a + k * math.pi / 3.0
        v[k] = (cx + s * math.cos(ang), cy + s * math.sin(ang))
    return v

def hex_support(pos, s, theta, normals):
    # returns (max over vertices of dot(vertex, n_k)) for each of 6 normals
    verts = pos[:, None, :] + s * (np.cos(theta[:, None] * 0 + np.arange(6) * math.pi / 3.0 + theta[:, None]))
    # theta[:,None] rotation: vertex angle = theta + k*60
    verts = pos[:, None, :] + s * np.stack([
        np.cos(theta[:, None] + np.arange(6) * math.pi / 3.0),
        np.sin(theta[:, None] + np.arange(6) * math.pi / 3.0)
    ], axis=-1)
    # dot with each normal
    # verts: (N,6,2), normals: (6,2) -> (N,6)
    return np.einsum('nki,ki->nk', verts, normals)

def hex_max_support(pos, s, theta, normals):
    return hex_support(pos, s, theta, normals).max(axis=1).max()

def containment(pos, s, theta, normals, cx, cy):
    # max over vertices of dot(vertex - c, n_k)
    verts = pos[:, None, :] + s * np.stack([
        np.cos(theta[:, None] + np.arange(6) * math.pi / 3.0),
        np.sin(theta[:, None] + np.arange(6) * math.pi / 3.0)
    ], axis=-1)
    rel = verts - np.array([cx, cy])
    return np.einsum('nki,ki->nk', rel, normals).max()

def score_layout(pos, s, theta, cx, cy):
    # outer side length needed to contain all, with outer rotation 0
    normals = hex_normals(0.0)
    m = containment(pos, s, theta, normals, cx, cy)
    return m / (math.sqrt(3.0) / 2.0)

# ---------- Force-directed packing ----------
def pack_one(deadline, rng, n, s=1.0, iters=4000, lr=0.02, rep_k=5.0, bnd_k=20.0,
             min_sep=2.0, jitter=0.05):
    # initial positions on a hex-ish grid + jitter
    r = math.sqrt(3.0) / 2.0
    grid = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            grid.append((i * 2.0 + (j % 2) * 1.0, j * 2 * r))
    rng.shuffle(grid)
    pos = np.array(grid[:n], dtype=float)
    pos += rng.uniform(-jitter, jitter, size=(n, 2))
    theta = np.zeros(n)

    normals = hex_normals(0.0)
    best_side = score_layout(pos, s, theta, 0.0, 0.0)
    best = (pos.copy(), theta.copy(), best_side)
    cx = cy = 0.0

    for it in range(iters):
        if time.time() > deadline:
            break
        # compute forces
        force = np.zeros((n, 2))
        # pairwise repulsion (overlap resolution)
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = math.hypot(d[0], d[1])
                if dist < min_sep:
                    if dist < 1e-9:
                        d = (rng.uniform(-1, 1), rng.uniform(-1, 1))
                        dist = math.hypot(d[0], d[1])
                    push = (min_sep - dist) / dist
                    f = rep_k * push * d
                    force[i] += f
                    force[j] -= f
        # boundary soft constraint: pull vertices that exceed container
        # container side chosen as current best_side (adaptive)
        cont = best_side
        for i in range(n):
            verts = pos[i] + s * np.stack([
                np.cos(theta[i] + np.arange(6) * math.pi / 3.0),
                np.sin(theta[i] + np.arange(6) * math.pi / 3.0)
            ])
            for k in range(6):
                nk = normals[k]
                for v in verts:
                    val = (v[0] - cx) * nk[0] + (v[1] - cy) * nk[1]
                    lim = cont * math.sqrt(3.0) / 2.0
                    if val > lim:
                        over = val - lim
                        force[i] -= bnd_k * over * nk
        # integrate
        pos += lr * force
        # re-center
        pos -= pos.mean(axis=0)
        cx, cy = 0.0, 0.0
        # anneal learning rate
        lr = 0.02 * (0.999 ** it)
        # evaluate
        side = score_layout(pos, s, theta, 0.0, 0.0)
        if side < best_side:
            best_side = side
            best = (pos.copy(), theta.copy(), best_side)
    return best

def refine(pos, s, theta, cx, cy, deadline, rng, iters=600, lr=0.01):
    # local polish: minimize side with small random perturbations (simulated annealing)
    normals = hex_normals(0.0)
    cur = score_layout(pos, s, theta, cx, cy)
    best = cur
    bp = pos.copy()
    bth = theta.copy()
    temp = 0.5
    for it in range(iters):
        if time.time() > deadline:
            break
        i = rng.randrange(len(pos))
        d = rng.normal(0, lr, size=2)
        pos[i] += d
        new = score_layout(pos, s, theta, cx, cy)
        if new < cur or rng.random() < math.exp(-(new - cur) / max(temp, 1e-9)):
            cur = new
            if new < best:
                best = new
                bp = pos.copy()
                bth = theta.copy()
        else:
            pos[i] -= d
        temp *= 0.999
    return bp, bth, best

def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT - SAFE_MARGIN
    rng = random.Random(12345)
    best_side = float('inf')
    best_pos = None
    best_theta = None

    # run several packings with different seeds / params
    seeds = [1, 7, 13, 29, 41, 53, 67, 83, 97, 113]
    iters_list = [5000, 5000, 4000, 4000, 3000, 3000, 2500, 2500, 2000, 2000]
    for idx, seed in enumerate(seeds):
        if time.time() > deadline - 1.0:
            break
        rr = random.Random(seed)
        pos, theta, side = pack_one(deadline, rr, N, iters=iters_list[idx])
        if side < best_side:
            best_side = side
            best_pos = pos
            best_theta = theta

    # polish the best
    if best_pos is not None and time.time() < deadline:
        bp, bth, bside = refine(best_pos, 1.0, best_theta, 0.0, 0.0, deadline, rng, iters=2000)
        if bside < best_side:
            best_side = bside
            best_pos = bp
            best_theta = bth

    # final safety: ensure containment by slightly expanding if needed
    if best_pos is None:
        # fallback
        best_pos = np.array([[2.0 * k, 0.0] for k in range(-5, 6)], dtype=float)
        best_theta = np.zeros(N)
        best_side = 40.0

    inner = [[float(x), float(y), float(a)] for (x, y), a in zip(best_pos, best_theta)]
    return inner, (0.0, 0.0), float(best_side), 0.0
