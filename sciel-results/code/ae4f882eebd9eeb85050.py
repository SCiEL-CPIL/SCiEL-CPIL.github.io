"""Rewritten solver: hexagonal init + max-min force relaxation with time budget."""
import time
import numpy as np

N = 200
DEADLINE = 150.0
SAFE = 12.0  # safety margin (s) before the evaluator kills us


def _score(pts):
    # pts: (N,2). Return (r, min_pair, min_bound) in double precision.
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    bound = 0.5 - np.abs(pts).max()
    r = min(min_pair * 0.5, bound)
    return r, min_pair, bound


def _hex_init():
    # Hexagonal lattice with horizontal rows, densified by shifting alternating rows.
    # Start with a generous grid then trim to N, keeping the densest central block.
    # Row spacing for a hex lattice of unit spacing s: dx = s, dy = s*sqrt(3)/2.
    # We want ~200 points; choose a square-ish block.
    # Use a hex lattice with spacing 1, build a big block, then keep N points nearest centre.
    s = 1.0
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    # number of rows/cols to get > N
    ncol = 22
    nrow = 11
    pts = []
    for j in range(nrow):
        y = (j - (nrow - 1) / 2.0) * dy
        off = (dx / 2.0) * (j % 2)
        for i in range(ncol):
            x = (i - (ncol - 1) / 2.0) * dx + off
            pts.append([x, y])
    pts = np.array(pts)
    # keep the N points closest to origin (central, densest)
    r2 = (pts ** 2).sum(1)
    idx = np.argsort(r2)[:N]
    pts = pts[idx]
    return pts


def _knn_indices(pts, k):
    # return list of (i, j) for j in k nearest of i (undirected, i<j)
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, 1e18)
    nn = np.argsort(dist, axis=1)[:, :k]
    pairs = set()
    for i in range(N):
        for j in nn[i]:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    return list(pairs)


def _relax(pts0, t_start, k=6, steps=40000, lr0=1e-3, decay=0.99995, beta=1.0):
    pts = pts0.copy()
    v = np.zeros_like(pts)
    m = np.zeros_like(pts)
    t = 0
    lr = lr0
    best_r, best_pts = -1.0, pts.copy()
    # precompute pairs; recompute occasionally as points move
    pairs = _knn_indices(pts, k)
    recompute_every = 200
    # force scale: we want to push min distance up. Use repulsion ~ 1/d^2 on near pairs.
    for step in range(steps):
        if time.time() - t_start > DEADLINE - SAFE:
            break
        # recompute k-nearest pairs periodically
        if step % recompute_every == 0:
            pairs = _knn_indices(pts, k)
        # forces
        F = np.zeros_like(pts)
        # repulsion between near neighbours: push apart, stronger when close
        for (i, j) in pairs:
            dij = pts[i] - pts[j]
            dist = np.sqrt((dij ** 2).sum()) + 1e-12
            # force magnitude: want to increase dist; use 1/dist^2
            f = (1.0 / (dist * dist)) * 1.0
            unit = dij / dist
            F[i] += unit * f
            F[j] -= unit * f
        # boundary: push inward strongly when near edge
        # clearance to each wall
        c = 0.5 - np.abs(pts)  # min clearance to boundary per point (vector of 2 per point)
        # use a soft penalty: for each coordinate, if close to wall, push in
        # per-axis
        for ax in range(2):
            # wall at +0.5 and -0.5
            over = 0.5 - pts[:, ax]  # distance to + wall (positive inside)
            # push toward centre when over is small
            # force = -sign * max(0, 1/over - 1) style
            # use: if over < 0.1, push in with strength ~ (0.1-over)
            close = over < 0.12
            if np.any(close):
                strength = (0.12 - over[close]) / 0.12
                direction = -np.sign(pts[close, ax])  # push toward 0
                F[close, ax] += direction * strength * 2.0
            over2 = pts[:, ax] + 0.5  # distance to - wall
            close2 = over2 < 0.12
            if np.any(close2):
                strength = (0.12 - over2[close2]) / 0.12
                direction = -np.sign(pts[close2, ax])
                F[close2, ax] += direction * strength * 2.0
        # tiny global centering to avoid drift
        F -= 1e-4 * pts
        # Adam update
        v = 0.9 * v + 0.1 * F
        m = 0.99 * m + 0.01 * (F * F)
        m = np.maximum(m, 1e-8)
        pts += lr * v / np.sqrt(m)
        lr *= decay
        # keep strictly inside (clip)
        pts = np.clip(pts, -0.5 + 1e-9, 0.5 - 1e-9)
        # periodic score check
        if step % 400 == 0:
            r, _, _ = _score(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    return best_pts, best_r


def construct():
    t_start = time.time()
    best_pts = None
    best_r = -1.0

    # multiple restarts with slight jitter around hex init
    rng = np.random.default_rng(12345)
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() - t_start > DEADLINE - SAFE:
            break
        if restart == 0:
            pts0 = _hex_init()
        else:
            pts0 = _hex_init()
            # jitter
            pts0 = pts0 + rng.normal(0, 0.02, size=pts0.shape)
            pts0 = np.clip(pts0, -0.5 + 1e-9, 0.5 - 1e-9)
        # adapt steps to remaining time
        remaining = (DEADLINE - SAFE) - (time.time() - t_start)
        # estimate: we'll cap steps but the inner loop also checks deadline
        pts, r = _relax(pts0, t_start, k=6, steps=30000, lr0=1.2e-3, decay=0.99997)
        if r > best_r:
            best_r = r
            best_pts = pts

    # final precise polish on the best
    if best_pts is not None and (DEADLINE - SAFE) - (time.time() - t_start) > 5.0:
        # tighter, slower polish
        best_pts, r2 = _relax(best_pts, t_start, k=8, steps=40000, lr0=3e-4, decay=0.99999)
        if r2 > best_r:
            best_r = r2
            best_pts = best_pts.copy()

    # safety: ensure finite and inside
    if best_pts is None:
        best_pts = _hex_init()
    best_pts = np.nan_to_num(best_pts, nan=0.0, posinf=0.49, neginf=-0.49)
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    return best_pts
