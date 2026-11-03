"""Circle packing in the unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
import time

N = 26
_DEADLINE = time.time() + 140.0  # safety margin under the 150 s kill limit


def _is_valid(pos, r, eps=1e-9):
    x, y = pos
    if x < -eps or y < -eps or x + r > 1.0 + eps or y + r > 1.0 + eps:
        return False
    for k in range(N):
        if k == 0:
            continue
        dx = x - pos[k, 0]
        dy = y - pos[k, 1]
        if dx * dx + dy * dy < (r + r[k] - eps) ** 2:
            return False
    return True


def _objective(pos, r):
    # objective = sum(r) - penalty(overlap + boundary)
    n = len(pos)
    pen = 0.0
    for i in range(n):
        ri = r[i]
        pi = pos[i]
        # boundary
        b = min(pi[0] - ri, pi[1] - ri, 1.0 - pi[0] - ri, 1.0 - pi[1] - ri)
        if b < 0.0:
            pen += b * b
        for j in range(i + 1, n):
            d = np.linalg.norm(pi - pos[j])
            gap = d - (ri + r[j])
            if gap < 0.0:
                pen += gap * gap
    return float(np.sum(r)) - pen


def _grad_step(pos, r, lr, mu):
    n = len(pos)
    gpos = np.zeros_like(pos)
    gr = np.zeros(n)
    for i in range(n):
        pi = pos[i]
        ri = r[i]
        # boundary gradients (push inward)
        if pi[0] - ri < 0.0:
            gpos[i, 0] += (pi[0] - ri) * 2.0
            gr[i] -= (pi[0] - ri) * 2.0
        if pi[1] - ri < 0.0:
            gpos[i, 1] += (pi[1] - ri) * 2.0
            gr[i] -= (pi[1] - ri) * 2.0
        if 1.0 - pi[0] - ri < 0.0:
            gpos[i, 0] += -(1.0 - pi[0] - ri) * 2.0
            gr[i] -= (1.0 - pi[0] - ri) * 2.0
        if 1.0 - pi[1] - ri < 0.0:
            gpos[i, 1] += -(1.0 - pi[1] - ri) * 2.0
            gr[i] -= (1.0 - pi[1] - ri) * 2.0
        for j in range(i + 1, n):
            d = pi - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.array([1e-6, 1e-6])
                dist = np.linalg.norm(d)
            gap = dist - (ri + r[j])
            if gap < 0.0:
                g = -gap  # positive magnitude
                unit = d / dist
                # d(d2)/dpos_i = 2*(pi-pj); d(gap^2)/dpos_i = 2*gap*2*(pi-pj)
                gpos[i] += 4.0 * gap * (pi - pos[j])
                gpos[j] += 4.0 * gap * (pos[j] - pi)
                gr[i] += -2.0 * gap
                gr[j] += -2.0 * gap
    pos = pos - lr * gpos
    r = r - lr * gr
    # project: r >= 0
    r = np.maximum(r, 1e-4)
    return pos, r


def _optimize(pos, r, mu, steps, lr0, lr1):
    n = len(pos)
    for s in range(steps):
        lr = lr0 * (lr1 / lr0) ** (s / max(1, steps - 1))
        # adaptive: shrink if objective worsening handled implicitly
        pos, r = _grad_step(pos, r, lr, mu)
    return pos, r


def _local_search(pos, r, budget):
    # greedy radius growth + gradient polish
    for _ in range(200):
        if time.time() > budget:
            break
        improved = False
        for i in range(N):
            if time.time() > budget:
                break
            base = r[i]
            # grow radius i as far as possible
            for _ in range(40):
                trial = r.copy()
                trial[i] *= 1.02
                if _is_valid(pos, trial):
                    r = trial
                    improved = True
                else:
                    break
        # gradient polish
        pos, r = _grad_step(pos, r, 0.01, 1.0)
        if not improved:
            break
    return pos, r


def _grid_init(seed):
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side))
    cells = cells[:N]
    pos = np.array(cells, dtype=float)
    base_r = 0.5 / side
    # small jitter
    pos += rng.uniform(-0.01, 0.01, size=pos.shape)
    pos = np.clip(pos, 0.0, 1.0)
    r = np.full(N, base_r)
    return pos, r


def _random_init(seed):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 1.0, size=(N, 2))
    r = rng.uniform(0.05, 0.15, size=N)
    return pos, r


def _mix_init(seed):
    # a few big circles + many small ones, placed to avoid obvious overlap
    rng = np.random.default_rng(seed)
    big = 6
    pos = np.zeros((N, 2))
    r = np.zeros(N)
    # big circles in a loose grid of corners/center
    spots = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.5), (0.5, 0.05)]
    for k in range(big):
        pos[k] = spots[k]
        r[k] = 0.16
    # small circles scattered
    for k in range(big, N):
        pos[k] = rng.uniform(0.08, 0.92, size=2)
        r[k] = 0.05
    return pos, r


def _evaluate(pos, r):
    if not _is_valid(pos, r):
        return -1.0
    return float(np.sum(r))


def construct():
    start = time.time()
    budget = _DEADLINE - 3.0  # leave margin
    best_val = -1.0
    best = None

    inits = [
        _grid_init(0),
        _mix_init(1),
        _random_init(2),
        _mix_init(7),
        _random_init(13),
    ]

    for seed_idx, (pos, r) in enumerate(inits):
        if time.time() > budget:
            break
        # local search: grow + polish
        pos, r = _local_search(pos, r, budget)
        val = _evaluate(pos, r)
        if val > best_val:
            best_val = val
            best = (pos.copy(), r.copy())
        # gradient optimization from here
        pos2, r2 = _optimize(pos.copy(), r.copy(), 1.0, 300, 0.02, 0.001)
        val2 = _evaluate(pos2, r2)
        if val2 > best_val:
            best_val = val2
            best = (pos2.copy(), r2.copy())

    if best is None:
        # fallback to a valid grid
        pos, r = _grid_init(0)
        best_val = _evaluate(pos, r)
        best = (pos, r)

    pos, r = best
    # final validity repair: shrink any offending radius
    for _ in range(100):
        if _is_valid(pos, r):
            break
        # find worst offender and shrink
        for i in range(N):
            if time.time() > _DEADLINE - 1.0:
                break
            if _is_valid(pos, r):
                break
            r[i] *= 0.98
    # ensure absolute validity
    if not _is_valid(pos, r):
        # last resort: tiny uniform
        side = int(np.ceil(np.sqrt(N)))
        pos = np.array([((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)][:N])
        r = np.full(N, 0.5 / side - 1e-3)

    out = np.column_stack([pos[:, 0], pos[:, 1], r])
    return out
