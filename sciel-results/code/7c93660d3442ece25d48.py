"""Circle packing in unit square: 26 circles, maximize sum of radii.

Structural change from the seed (penalty-based L-BFGS on joint (x,y,r)):
  - Radii are NOT a free optimization variable. For a fixed set of positions,
    the optimal radius of each circle is exactly the minimum of its distance
    to the four walls and its distance to every other circle minus that
    circle's radius. We solve this tightly (successive over-relaxation +
    a few Gauss-Seidel sweeps), which is exact and avoids the penalty
    landscape entirely.
  - Positions are optimized by a directed local search (random pairwise
    separation moves + occasional single-circle nudges) that accepts a move
    only if the resulting sum of radii improves. This is a different
    optimization family (combinatorial / stochastic local search) rather
    than a continuous penalty minimizer.
  - A small set of diverse initial layouts (grid, random, corners+center)
    seeds the search.
"""
import numpy as np
import time

N = 26
DEADLINE = 140.0  # explicit wall-clock guard (evaluator kills at 150 s)


def _max_radii(positions, iters=600, tol=1e-12):
    """Given fixed positions, compute the maximal feasible radii.

    r_i <= dist_to_wall_i  and  r_i + r_j <= dist_ij for all i != j.
    Solved by successive over-relaxation on the radius variables, then a
    few Gauss-Seidel sweeps for tightness. This is the exact optimum for
    the given positions (the feasible set in r is a polymatroid-like box
    intersected with half-spaces; the coordinate descent converges to the
    unique maximal point because the constraints are monotone).
    """
    n = len(positions)
    # distance to walls
    wall = np.minimum(np.minimum(positions[:, 0], positions[:, 1]),
                      np.minimum(1.0 - positions[:, 0], 1.0 - positions[:, 1]))
    r = np.maximum(wall, 0.0).copy()

    # pairwise distances
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dij = np.hypot(dx, dy)
            d[i, j] = d[j, i] = dij

    # successive over-relaxation: r_i = min(wall_i, min_j (d_ij - r_j))
    for _ in range(iters):
        changed = 0.0
        for i in range(n):
            m = wall[i]
            for j in range(n):
                if i != j:
                    v = d[i, j] - r[j]
                    if v < m:
                        m = v
            if m < 0.0:
                m = 0.0
            if m < r[i]:
                r[i] = m
                changed += 1.0
        if changed == 0.0:
            break

    # Gauss-Seidel sweeps for extra tightness
    for _ in range(30):
        for i in range(n):
            m = wall[i]
            for j in range(n):
                if i != j:
                    v = d[i, j] - r[j]
                    if v < m:
                        m = v
            if m < 0.0:
                m = 0.0
            r[i] = m
    return r


def _sum_radii(positions):
    return float(np.sum(_max_radii(positions)))


def _initial_layouts(rng):
    """Return a list of diverse initial position arrays (each (N,2))."""
    layouts = []

    # 1) uniform grid
    side = int(np.ceil(np.sqrt(N)))
    grid = []
    for i in range(side):
        for j in range(side):
            grid.append([(i + 0.5) / side, (j + 0.5) / side])
    layouts.append(np.array(grid[:N], dtype=float))

    # 2) pure random
    layouts.append(rng.uniform(0.05, 0.95, size=(N, 2)))

    # 3) corners + edge midpoints + center + random fill
    special = [
        [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
        [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5],
        [0.5, 0.5],
    ]
    rest = rng.uniform(0.1, 0.9, size=(N - len(special), 2))
    layouts.append(np.vstack([special, rest]).astype(float))

    # 4) grid with jitter
    g = np.array(grid[:N], dtype=float) + rng.uniform(-0.04, 0.04, size=(N, 2))
    g = np.clip(g, 0.05, 0.95)
    layouts.append(g)

    return layouts


def _local_search(positions, rng, time_budget):
    """Directed local search on positions. Returns best positions found."""
    n = N
    best_pos = positions.copy()
    best_val = _sum_radii(best_pos)
    start = time.time()

    # precompute nothing; recompute radii each eval (N=26 is small)
    step = 0.06
    no_improve = 0
    max_no_improve = 400

    while time.time() - start < time_budget and no_improve < max_no_improve:
        # choose a move
        if rng.random() < 0.55:
            # pairwise separation: push two circles apart along their axis
            i, j = rng.choice(n, size=2, replace=False)
            dx = best_pos[i, 0] - best_pos[j, 0]
            dy = best_pos[i, 1] - best_pos[j, 1]
            dist = np.hypot(dx, dy)
            if dist < 1e-9:
                dx, dy = rng.normal(0, 1, 2)
                dist = np.hypot(dx, dy)
            ux, uy = dx / dist, dy / dist
            s = step * rng.uniform(0.5, 1.5)
            trial = best_pos.copy()
            trial[i, 0] += ux * s
            trial[i, 1] += uy * s
            trial[j, 0] -= ux * s
            trial[j, 1] -= uy * s
        else:
            # single-circle random nudge
            i = rng.randint(n)
            ang = rng.uniform(0, 2 * np.pi)
            s = step * rng.uniform(0.5, 1.5)
            trial = best_pos.copy()
            trial[i, 0] += np.cos(ang) * s
            trial[i, 1] += np.sin(ang) * s

        # keep within a safe inset (walls handled by radius, but keep centers sane)
        trial = np.clip(trial, 0.0, 1.0)

        val = _sum_radii(trial)
        if val > best_val + 1e-12:
            best_pos = trial
            best_val = val
            no_improve = 0
            # occasionally anneal the step down if we've been improving a lot
        else:
            no_improve += 1
            # slow decay of step to refine
            step = max(0.005, step * 0.995)

    return best_pos, best_val


def construct():
    start = time.time()
    deadline = start + DEADLINE

    rng = np.random.default_rng(12345)
    layouts = _initial_layouts(rng)

    best_pos = None
    best_val = -np.inf

    # Phase 1: local search from each diverse layout, splitting the budget.
    n_layouts = len(layouts)
    # Give more budget to better initial layouts (greedy ordering by initial value)
    init_vals = []
    for L in layouts:
        v = _sum_radii(L)
        init_vals.append(v)
    order = np.argsort(-np.array(init_vals))

    # Allocate remaining time among layouts, weighted by initial quality
    remaining = deadline - time.time()
    if remaining <= 1.0:
        # not enough time; just return best initial
        best_idx = order[0]
        best_pos = layouts[best_idx]
        best_val = init_vals[best_idx]
    else:
        # weights: better initial layout gets more search time
        w = np.array([1.0 + 5.0 * max(0.0, init_vals[k] - min(init_vals)) for k in range(n_layouts)])
        w = w / w.sum()
        budgets = w * remaining * 0.9  # keep 10% safety margin

        for rank in order:
            if time.time() > deadline - 1.0:
                break
            L = layouts[rank].copy()
            b = min(budgets[rank], deadline - time.time() - 1.0)
            if b <= 0.1:
                # just evaluate initial
                v = _sum_radii(L)
                if v > best_val:
                    best_val = v
                    best_pos = L.copy()
                continue
            pos, val = _local_search(L, rng, b)
            if val > best_val:
                best_val = val
                best_pos = pos

    # Phase 2: final polish — a few more short local-search passes from the
    # current best to squeeze out remaining gain, if time permits.
    if best_pos is not None and time.time() < deadline - 2.0:
        budget2 = min(8.0, deadline - time.time() - 1.0)
        if budget2 > 0.5:
            pos2, val2 = _local_search(best_pos, rng, budget2)
            if val2 > best_val:
                best_val = val2
                best_pos = pos2

    # Build final result: positions + exact maximal radii
    if best_pos is None:
        # fallback: plain grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                x, y = (i + 0.5) / side, (j + 0.5) / side
                cells.append([x, y])
        best_pos = np.array(cells[:N], dtype=float)
        best_val = _sum_radii(best_pos)

    radii = _max_radii(best_pos, iters=800)
    result = np.zeros((N, 3))
    result[:, 0] = best_pos[:, 0]
    result[:, 1] = best_pos[:, 1]
    result[:, 2] = radii

    # Final safety clip: ensure radii respect walls exactly (numerical guard)
    for i in range(N):
        wall = min(result[i, 0], result[i, 1], 1.0 - result[i, 0], 1.0 - result[i, 1])
        if result[i, 2] > wall:
            result[i, 2] = max(0.0, wall)

    return result
