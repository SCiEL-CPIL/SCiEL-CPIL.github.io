"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26
_DEADLINE = 150.0
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _valid(p):
    """Check disjoint interiors and inside unit square (with tiny margin)."""
    n = len(p)
    for i in range(n):
        x, y, r = p[i]
        if r < -1e-9:
            return False
        if x - r < -1e-9 or y - r < -1e-9 or x + r > 1.0 + 1e-9 or y + r > 1.0 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = p[i, 0] - p[j, 0]
            dy = p[i, 1] - p[j, 1]
            d = np.hypot(dx, dy)
            if d < p[i, 2] + p[j, 2] - 1e-8:
                return False
    return True


def _score(p):
    return float(np.sum(p[:, 2]))


def _initial_grid(n):
    side = int(np.ceil(np.sqrt(n)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:n], dtype=float)


def _repel(p, dt):
    """Physics: repulsive forces from overlaps + walls, move centers."""
    n = len(p)
    F = np.zeros((n, 2))
    for i in range(n):
        for j in range(i + 1, n):
            dx = p[i, 0] - p[j, 0]
            dy = p[i, 1] - p[j, 1]
            d = np.hypot(dx, dy)
            overlap = p[i, 2] + p[j, 2] - d
            if overlap > 0:
                if d < 1e-12:
                    dx, dy = 1e-6, 1e-6
                    d = np.hypot(dx, dy)
                fx = overlap * dx / d
                fy = overlap * dy / d
                F[i, 0] += fx
                F[i, 1] += fy
                F[j, 0] -= fx
                F[j, 1] -= fy
    # wall repulsion
    for i in range(n):
        r = p[i, 2]
        if p[i, 0] - r < 0:
            F[i, 0] += -p[i, 0] + r
        if p[i, 0] + r > 1:
            F[i, 0] += p[i, 0] + r - 1
        if p[i, 1] - r < 0:
            F[i, 1] += -p[i, 1] + r
        if p[i, 1] + r > 1:
            F[i, 1] += p[i, 1] + r - 1
    p[:, 0] += dt * F[:, 0]
    p[:, 1] += dt * F[:, 1]
    return p


def _clamp_inside(p):
    for i in range(len(p)):
        r = p[i, 2]
        p[i, 0] = min(max(p[i, 0], r), 1.0 - r)
        p[i, 1] = min(max(p[i, 1], r), 1.0 - r)
    return p


def _relax(p, max_iter=4000, dt=0.05, decay=0.999):
    """Iterate repulsion until no overlaps remain."""
    p = p.copy()
    d = dt
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 5:
            break
        p = _repel(p, d)
        p = _clamp_inside(p)
        d *= decay
        # check convergence
        max_overlap = 0.0
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                ov = p[i, 2] + p[j, 2] - np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1])
                if ov > max_overlap:
                    max_overlap = ov
        if max_overlap < 1e-9:
            break
    return p


def _grow_and_relax(p, n_steps=200):
    """Grow all radii uniformly while keeping feasibility via relaxation."""
    p = p.copy()
    best = p.copy()
    for s in range(n_steps):
        if _elapsed() > _DEADLINE - 5:
            break
        # try a small uniform growth
        p[:, 2] *= 1.001
        # relax to fix overlaps
        p = _relax(p, max_iter=300, dt=0.05)
        p = _clamp_inside(p)
        if _valid(p):
            best = p.copy()
        else:
            # revert growth
            p = best.copy()
            break
    return best


def _local_refine(p, iters=2000):
    """Local optimization: move centers to allow slight radius growth, keep feasible."""
    p = p.copy()
    best = p.copy()
    best_score = _score(p)
    lr = 0.01
    for it in range(iters):
        if _elapsed() > _DEADLINE - 5:
            break
        # gradient of overlap -> push apart
        F = np.zeros((len(p), 2))
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                d = np.hypot(dx, dy)
                if d < 1e-12:
                    continue
                # encourage separation: repel proportional to closeness
                gap = d - p[i, 2] - p[j, 2]
                if gap < 0.02:
                    w = max(0.0, 0.02 - gap)
                    F[i, 0] += w * dx / d
                    F[i, 1] += w * dy / d
                    F[j, 0] -= w * dx / d
                    F[j, 1] -= w * dy / d
        # walls
        for i in range(len(p)):
            r = p[i, 2]
            if p[i, 0] - r < 0.02:
                F[i, 0] += 0.5 * (0.02 - (p[i, 0] - r))
            if p[i, 0] + r > 1 - 0.02:
                F[i, 0] -= 0.5 * ((p[i, 0] + r) - (1 - 0.02))
            if p[i, 1] - r < 0.02:
                F[i, 1] += 0.5 * (0.02 - (p[i, 1] - r))
            if p[i, 1] + r > 1 - 0.02:
                F[i, 1] -= 0.5 * ((p[i, 1] + r) - (1 - 0.02))
        p[:, 0] += lr * F[:, 0]
        p[:, 1] += lr * F[:, 1]
        p = _clamp_inside(p)
        if _valid(p):
            sc = _score(p)
            if sc > best_score:
                best_score = sc
                best = p.copy()
        else:
            p = best.copy()
    return best


def construct():
    best = _initial_grid(N)
    best_score = _score(best)
    rng = np.random.default_rng(0)

    # multiple random restarts
    n_restarts = 6
    for restart in range(n_restarts):
        if _elapsed() > _DEADLINE - 10:
            break
        if restart == 0:
            p = _initial_grid(N)
        else:
            # jittered grid
            p = _initial_grid(N)
            p[:, 0] += rng.uniform(-0.05, 0.05, N)
            p[:, 1] += rng.uniform(-0.05, 0.05, N)
            p = _clamp_inside(p)
            p = _relax(p, max_iter=2000, dt=0.05)
            if not _valid(p):
                continue

        # phase 1: physics relaxation
        p = _relax(p, max_iter=3000, dt=0.05)
        p = _clamp_inside(p)
        if not _valid(p):
            continue

        # phase 2: radius growth
        p = _grow_and_relax(p, n_steps=150)
        if not _valid(p):
            continue

        # phase 3: local refine
        p = _local_refine(p, iters=1500)
        if not _valid(p):
            continue

        sc = _score(p)
        if sc > best_score:
            best_score = sc
            best = p.copy()

    return best
