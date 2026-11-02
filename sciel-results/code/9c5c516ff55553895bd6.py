"""Fresh design: repulsive-force optimizer + pairwise feasibility refinement."""
import time
import numpy as np

N = 200
DEADLINE = 130.0  # safety margin below the 150s kill (evaluator uses ~120s budget)


def _hex_init(n, rng):
    """Hexagonal (triangular) lattice points, cropped to n, centred in the square."""
    # generate a generous hex grid, keep points inside the square, take first n
    pts = []
    spacing = 0.05
    dx = spacing
    dy = spacing * np.sqrt(3.0) / 2.0
    offset = dx / 2.0
    # iterate over rows
    row = 0
    while len(pts) < n:
        y = -0.5 + 0.02 + row * dy
        if y > 0.5 - 0.02:
            break
        x = -0.5 + 0.02
        col = 0
        while x < 0.5 - 0.02:
            xx = x + (offset if row % 2 else 0.0)
            if -0.5 + 1e-6 <= xx <= 0.5 - 1e-6 and -0.5 + 1e-6 <= y <= 0.5 - 1e-6:
                pts.append((xx, y))
            x += dx
            col += 1
        row += 1
    pts = np.array(pts[:n], dtype=float)
    # centre the cloud at origin
    pts -= pts.mean(axis=0)
    # scale so it fits inside the square with margin
    maxabs = np.max(np.abs(pts))
    if maxabs > 0:
        pts *= (0.48 / maxabs)
    return pts


def _score(pts):
    """Feasible radius for a configuration (0 if infeasible)."""
    n = len(pts)
    # boundary
    b = 0.5 - np.max(np.abs(pts), axis=1)
    r = float(np.min(b))
    if r <= 0:
        return 0.0
    # pairwise
    for i in range(n):
        d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
        m = float(np.min(d))
        if m / 2.0 < r:
            r = m / 2.0
    return r


def _repulse_step(pts, step, k_rep, k_wall):
    """One repulsive-force step. Returns new points."""
    n = len(pts)
    force = np.zeros_like(pts)
    # pairwise repulsion
    for i in range(n):
        diff = pts[i] - pts[i + 1:]
        d2 = np.einsum('ij,ij->i', diff, diff)
        d = np.sqrt(np.maximum(d2, 1e-12))
        # repulsion magnitude ~ 1/d^2, direction away
        w = k_rep / (d2 + 1e-9)
        force[i] += np.sum(diff * w[:, None], axis=0)
        # apply to others (Newton's third law)
        for j in range(i + 1, n):
            force[j] -= diff[j - i - 1] * w[j - i - 1]
    # wall repulsion: push inward when near boundary
    for axis in range(2):
        x = pts[:, axis]
        # distance to right wall
        d_right = 0.5 - x
        d_left = x + 0.5
        # force = -k * (1/d^2) toward centre when close
        f = np.zeros(n)
        # right wall
        mask = d_right < 0.15
        f[mask] -= k_wall / (d_right[mask] ** 2 + 1e-9)
        # left wall
        mask = d_left < 0.15
        f[mask] += k_wall / (d_left[mask] ** 2 + 1e-9)
        force[:, axis] += f
    pts = pts + step * force
    # clamp inside square
    pts = np.clip(pts, -0.5 + 1e-9, 0.5 - 1e-9)
    return pts


def _refine(pts, iters):
    """Pairwise direct repair: push violating pairs apart, keep inside."""
    n = len(pts)
    for _ in range(iters):
        r = _score(pts)
        if r <= 0:
            return pts
        # find all violating pairs and fix
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[j] - pts[i]
                d = np.linalg.norm(diff)
                if d < 2.0 * r - 1e-12:
                    # push apart to exactly 2r
                    if d < 1e-12:
                        # coincident: nudge randomly
                        u = np.array([1.0, 0.0])
                    else:
                        u = diff / d
                    target = 2.0 * r
                    # move each by (target - d)/2 along u
                    delta = (target - d) / 2.0
                    pts[i] -= delta * u
                    pts[j] += delta * u
                    changed = True
        # clamp inside
        pts = np.clip(pts, -0.5 + 1e-9, 0.5 - 1e-9)
        if not changed:
            break
    return pts


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    best = _hex_init(N, rng)
    best_score = _score(best)

    # ---- Phase 1: repulsive-force soft optimization ----
    step = 0.02
    k_rep = 0.0005
    k_wall = 0.0005
    total_soft = 400
    for it in range(total_soft):
        if time.time() - t0 > DEADLINE - 25:
            break
        best = _repulse_step(best, step, k_rep, k_wall)
        # anneal: shrink step over time
        step = 0.02 * (1.0 - 0.8 * (it / total_soft))
        s = _score(best)
        if s > best_score:
            best_score = s

    # ---- Phase 2: direct pairwise refinement (monotonic improvement) ----
    # do several passes; each pass is O(n^2)
    pass_count = 0
    while time.time() - t0 < DEADLINE - 10:
        before = _score(best)
        best = _refine(best, iters=2)
        after = _score(best)
        pass_count += 1
        if after <= before + 1e-15:
            # try a small random perturbation to escape local minima
            if time.time() - t0 < DEADLINE - 15:
                best = best + 0.001 * rng.standard_normal(best.shape)
                best = np.clip(best, -0.5 + 1e-9, 0.5 - 1e-9)
                best = _refine(best, iters=2)
                s = _score(best)
                if s > best_score:
                    best_score = s
                else:
                    # revert perturbation
                    best = _refine(best, iters=2)
            else:
                break
        else:
            best_score = max(best_score, after)

    # final clamp and sanity
    best = np.clip(best, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best)):
        # fallback to a safe grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    return best
