"""Packing 200 equal circles in a unit square via hex init + radius-driven local search."""
import time
import numpy as np

N = 200
HALF = 0.5
DEADLINE = 135.0  # safety margin before 150s kill


def _score(pts):
    """Compute the achievable common radius for centres pts (N,2)."""
    if not np.all(np.isfinite(pts)):
        return 0.0
    # wall clearance
    wall = HALF - np.max(np.abs(pts), axis=1)
    r_wall = np.min(wall)
    # pairwise
    # use scipy cKDTree for speed
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    pairs = tree.query_pairs(2.0)  # generous; we'll filter
    if len(pairs) == 0:
        return float(r_wall)
    pairs = np.array(pairs)
    d = pts[pairs[:, 0]] - pts[pairs[:, 1]]
    dists = np.linalg.norm(d, axis=1)
    # keep only the true nearest neighbours (distance < 2*r_wall*1.5 + eps) to be safe
    r_pair = np.min(dists) / 2.0
    return float(min(r_wall, r_pair))


def _hex_init(n):
    """Hexagonal lattice init, trimmed to n points, centered."""
    # hex grid: rows offset
    # spacing 1; we'll scale later
    pts = []
    row = 0
    while len(pts) < n + 40:
        y = row * np.sqrt(3.0) / 2.0
        offset = (row % 2) * 0.5
        col = -20
        while len(pts) < n + 40:
            x = col + offset
            pts.append((x, y))
            col += 1
        row += 1
    pts = np.array(pts[:n + 40])
    # center
    pts -= pts.mean(axis=0)
    # scale to roughly fit in [-0.5,0.5] with margin
    extent = np.max(np.abs(pts))
    scale = 0.45 / max(extent, 1e-9)
    pts *= scale
    # trim to n by keeping the most central
    r2 = np.sum(pts ** 2, axis=1)
    idx = np.argsort(r2)[:n]
    pts = pts[idx]
    return pts.copy()


def _local_search(pts, time_budget):
    """Radius-driven local search: find bottleneck, push apart, anneal."""
    from scipy.spatial import cKDTree
    n = len(pts)
    pts = pts.copy()
    best_pts = pts.copy()
    best_r = _score(pts)

    t0 = time.time()
    # temperature schedule
    T = 0.02
    T_min = 1e-6
    cool = 0.97

    while time.time() - t0 < time_budget:
        # compute current bottleneck
        wall = HALF - np.max(np.abs(pts), axis=1)
        r_wall = np.min(wall)
        tree = cKDTree(pts)
        pairs = tree.query_pairs(2.0 * max(r_wall, 1e-4) + 0.1)
        if len(pairs) == 0:
            r_pair = np.inf
            pair = None
        else:
            pairs = np.array(pairs)
            d = pts[pairs[:, 0]] - pts[pairs[:, 1]]
            dists = np.linalg.norm(d, axis=1)
            imin = np.argmin(dists)
            r_pair = dists[imin] / 2.0
            pair = (pairs[imin, 0], pairs[imin, 1])

        r = min(r_wall, r_pair)

        # update best
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

        # identify the constraint that is binding
        # generate a perturbation that increases the bottleneck
        # pick a random point, push it away from its nearest neighbour or wall
        i = np.random.randint(n)
        # find nearest neighbour of i
        d, j = tree.query(pts[i])
        # direction away from j
        if d < 1e-12:
            dirv = np.array([np.random.randn(), np.random.randn()])
        else:
            dirv = pts[i] - pts[j]
            norm = np.linalg.norm(dirv)
            if norm < 1e-12:
                dirv = np.array([np.random.randn(), np.random.randn()])
            else:
                dirv = dirv / norm
        # also consider wall push
        wall_dir = np.zeros(2)
        for k in range(2):
            if pts[i, k] > 0:
                wall_dir[k] = -1.0
            elif pts[i, k] < 0:
                wall_dir[k] = 1.0
        # mix
        w = np.random.rand()
        if w < 0.5:
            move_dir = dirv
        else:
            move_dir = wall_dir
        if np.linalg.norm(move_dir) < 1e-12:
            move_dir = np.array([np.random.randn(), np.random.randn()])
            move_dir /= np.linalg.norm(move_dir)

        step = T * np.sqrt(1.0 / max(r, 1e-6))
        new_pts = pts.copy()
        new_pts[i] += move_dir * step * (0.5 + np.random.rand())
        # also nudge the partner
        if pair is not None and i == pair[0]:
            new_pts[pair[1]] -= move_dir * step * 0.5 * np.random.rand()
        elif pair is not None and i == pair[1]:
            new_pts[pair[0]] += move_dir * step * 0.5 * np.random.rand()

        # keep inside square (clamp with margin)
        new_pts = np.clip(new_pts, -HALF + 1e-9, HALF - 1e-9)

        new_r = _score(new_pts)
        # accept if better or with probability exp((new_r - r)/T)
        if new_r >= r:
            pts = new_pts
        else:
            if np.random.rand() < np.exp((new_r - r) / max(T, 1e-12)):
                pts = new_pts

        T *= cool
        if T < T_min:
            T = 0.005  # reheat occasionally
            if np.random.rand() < 0.1:
                T = 0.01

    return best_pts, best_r


def _polish(pts, time_budget):
    """High-precision active-set polish on binding constraints."""
    from scipy.spatial import cKDTree
    n = len(pts)
    pts = pts.copy()
    best_r = _score(pts)
    t0 = time.time()
    for it in range(2000):
        if time.time() - t0 > time_budget:
            break
        wall = HALF - np.max(np.abs(pts), axis=1)
        r_wall = np.min(wall)
        tree = cKDTree(pts)
        pairs = tree.query_pairs(2.0 * max(r_wall, 1e-6) + 0.05)
        if len(pairs) == 0:
            r_pair = np.inf
        else:
            pairs = np.array(pairs)
            d = pts[pairs[:, 0]] - pts[pairs[:, 1]]
            dists = np.linalg.norm(d, axis=1)
            r_pair = np.min(dists) / 2.0
        r = min(r_wall, r_pair)
        if r > best_r:
            best_r = r

        # if r is very small, break
        if r < 1e-10:
            break

        # find all near-binding constraints (within 1e-4 of r)
        tol = 1e-5
        moved = False
        # wall bindings
        wall = HALF - np.max(np.abs(pts), axis=1)
        wall_bind = np.where(wall < r + tol)[0]
        for i in wall_bind:
            # push i inward along the wall normal
            for k in range(2):
                if pts[i, k] > 0 and pts[i, k] > HALF - r - tol:
                    pts[i, k] = HALF - r - tol * 0.5
                    moved = True
                elif pts[i, k] < 0 and pts[i, k] < -HALF + r + tol:
                    pts[i, k] = -HALF + r + tol * 0.5
                    moved = True
        # pair bindings
        if len(pairs) > 0:
            pairs = np.array(pairs)
            d = pts[pairs[:, 0]] - pts[pairs[:, 1]]
            dists = np.linalg.norm(d, axis=1)
            bind = np.where(dists < 2.0 * r + tol)[0]
            for b in bind:
                i, j = pairs[b]
                dd = pts[i] - pts[j]
                dist = np.linalg.norm(dd)
                if dist < 1e-12:
                    continue
                dd = dd / dist
                # push i and j apart by half the deficit
                deficit = (2.0 * r - dist) / 2.0
                if deficit > 0:
                    pts[i] += dd * deficit
                    pts[j] -= dd * deficit
                    moved = True
        if not moved:
            break
        # clamp
        pts = np.clip(pts, -HALF + 1e-12, HALF - 1e-12)

    return pts, best_r


def construct():
    t_start = time.time()
    # hex init
    pts = _hex_init(N)
    # ensure valid
    pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)

    # local search with most of the budget
    ls_budget = 90.0
    pts, r1 = _local_search(pts, ls_budget)

    # polish
    polish_budget = min(20.0, DEADLINE - (time.time() - t_start) - 5.0)
    if polish_budget > 1.0:
        pts, r2 = _polish(pts, polish_budget)
    else:
        r2 = r1

    # final validity check
    if not np.all(np.isfinite(pts)):
        pts = _hex_init(N)
    pts = np.clip(pts, -HALF + 1e-12, HALF - 1e-12)

    return pts
