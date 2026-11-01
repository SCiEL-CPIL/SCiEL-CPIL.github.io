"""
Improved circle packer with hex-lattice seed, fast local evaluation,
vectorized annealing, and a final gradient-free local polish.
"""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300
SEED = 42
T_START = time.time()
DEADLINE = 130.0  # 20s margin before 150s limit


def _score_fast(points, tree=None):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    if tree is None:
        tree = cKDTree(points)
    d, _ = tree.query(points, k=2, workers=-1)
    r_pair = 0.5 * d[:, 1].min()
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _score(points):
    """Full score with tree rebuild."""
    return _score_fast(points)


def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _anneal(points, t_budget, rng):
    """
    Annealing with KD-tree incremental updates and adaptive step size.
    """
    start = time.time()
    best = points.copy()
    best_r = _score_fast(best)
    cur = points.copy()
    cur_r = best_r
    tree = cKDTree(cur)

    # initial temperature
    d0 = tree.query(cur, k=2, workers=-1)[0][:, 1].min()
    T = 0.02 * d0
    T_min = 1e-6 * d0
    step = 0.04 * d0
    min_step = 1e-7 * d0

    while time.time() - start < t_budget and T > T_min:
        idx = rng.integers(N)
        ang = rng.random() * 2.0 * np.pi
        mag = rng.random() * step
        dx = mag * np.cos(ang)
        dy = mag * np.sin(ang)

        old = cur[idx].copy()
        new_pt = old + np.array([dx, dy])

        # quick container check
        if np.hypot(new_pt[0], new_pt[1]) > 1.0 - 1e-10:
            T *= 0.9995
            continue

        # update tree
        cur[idx] = new_pt
        tree.update(cur)

        # fast score
        d_arr, _ = tree.query(cur, k=2, workers=-1)
        r_pair = 0.5 * d_arr[:, 1].min()
        r_bound = (1.0 - np.linalg.norm(cur, axis=1)).min()
        new_r = min(r_pair, r_bound)

        if new_r >= cur_r or rng.random() < np.exp((new_r - cur_r) / max(T, 1e-12)):
            cur_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = cur.copy()
            step *= 1.005
        else:
            cur[idx] = old
            tree.update(cur)
            step *= 0.995

        step = np.clip(step, min_step, 0.15 * d0)
        T *= 0.9995

    return best, best_r


def _polish(points, t_budget, rng):
    """
    High-precision local polish: repeatedly nudge the most-constrained
    points towards less-constrained neighbors, with tiny random kicks.
    """
    start = time.time()
    cur = points.copy()
    best = cur.copy()
    best_r = _score_fast(best)
    tree = cKDTree(cur)

    while time.time() - start < t_budget:
        # score and find the bottleneck
        d_arr, idx_arr = tree.query(cur, k=2, workers=-1)
        r_pair = 0.5 * d_arr[:, 1].min()
        r_bound = (1.0 - np.linalg.norm(cur, axis=1)).min()
        cur_r = min(r_pair, r_bound)

        if cur_r > best_r:
            best_r = cur_r
            best = cur.copy()

        # find the most constrained point
        if r_pair < r_bound:
            # pair constraint dominates: find the closest pair
            i = np.argmin(d_arr[:, 1])
            j = idx_arr[i, 1]
            # push both slightly apart
            diff = cur[i] - cur[j]
            n = np.linalg.norm(diff)
            if n < 1e-12:
                diff = rng.standard_normal(2)
                n = np.linalg.norm(diff)
            diff /= n
            push = 0.002 * (d_arr[i, 1] / 2.0 - r_pair)
            push = max(push, 1e-8)
            new_i = cur[i] + push * diff
            new_j = cur[j] - push * diff
            # check container
            if np.hypot(new_i[0], new_i[1]) < 1.0 - 1e-10 and \
               np.hypot(new_j[0], new_j[1]) < 1.0 - 1e-10:
                cur[i] = new_i
                cur[j] = new_j
                tree.update(cur)
        else:
            # container constraint dominates
            i = np.argmin(r_bound)
            norm_i = np.linalg.norm(cur[i])
            if norm_i > 0:
                direction = -cur[i] / norm_i
                move = 0.002 * (r_bound - r_bound + 1e-6)
                move = max(move, 1e-8)
                new_i = cur[i] + move * direction
                if np.hypot(new_i[0], new_i[1]) < 1.0 - 1e-10:
                    cur[i] = new_i
                    tree.update(cur)

        # occasional small random kick on a random point
        if rng.random() < 0.1:
            k = rng.integers(N)
            kick = 0.001 * rng.standard_normal(2)
            new_k = cur[k] + kick
            if np.hypot(new_k[0], new_k[1]) < 1.0 - 1e-10:
                cur[k] = new_k
                tree.update(cur)

    # final score
    final_r = _score_fast(cur)
    if final_r > best_r:
        best_r = final_r
        best = cur.copy()
    return best, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # initial hex-lattice seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    best = pts.copy()
    best_r = _score(best)

    # Phase 1: Multi-start annealing
    while time.time() - t0 < DEADLINE - 25.0:
        jitter = 0.008 * rng.standard_normal((N, 2))
        cand = best + jitter
        rad = np.linalg.norm(cand, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-7)

        t_left = DEADLINE - 25.0 - (time.time() - t0)
        if t_left < 5.0:
            break
        t_alloc = min(t_left - 2.0, 12.0)
        cand, r = _anneal(cand, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand

    # Phase 2: High-precision polish
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 5.0:
        best, best_r = _polish(best, t_left - 2.0, rng)

    # final safety
    final_r = _score(best)
    if final_r <= 0.0:
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
