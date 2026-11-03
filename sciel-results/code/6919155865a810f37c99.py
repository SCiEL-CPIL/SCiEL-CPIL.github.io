"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 118.0  # safety margin below the 120 s wall-clock limit


def _score(pts):
    """Radius for a centre array: min(pair_dist/2, wall_dist)."""
    d = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    min_wall = np.min(0.5 - np.abs(pts))
    return min(min_pair * 0.5, min_wall)


def _anneal(pts, T0, T1, steps, rng):
    """Simulated annealing with random restarts on the minimum radius."""
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    for i in range(steps):
        T = T0 * (T1 / T0) ** (i / steps)
        # Random kick: move a random subset by a small random amount.
        kick = 0.02 * (T / T0 + 0.05)
        idx = rng.choice(N, size=max(1, N // 5), replace=False)
        cand = cur.copy()
        cand[idx] += rng.normal(0.0, kick, size=(len(idx), 3))
        cand_r = _score(cand)
        if cand_r > cur_r or rng.random() < np.exp((cand_r - cur_r) / max(T, 1e-9)):
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
        if i % 200 == 0 and time.time() > _DEADLINE:
            break
    return best, best_r


def _gradient_ascent(pts, iters, step0, step1, rng):
    """Gradient ascent on the minimum radius (subgradient of the min)."""
    cur = pts.copy()
    cur_r = _score(cur)
    best = cur.copy()
    best_r = cur_r
    for i in range(iters):
        step = step0 * (step1 / step0) ** (i / iters)
        d = cur[:, None, :] - cur[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        # Pair subgradient: push apart the closest pair(s).
        g = np.zeros_like(cur)
        near = dist < (dist.min() + 1e-6)
        near.fill_diagonal(False)
        for a in range(N):
            for b in range(a + 1, N):
                if near[a, b]:
                    diff = cur[a] - cur[b]
                    n = np.linalg.norm(diff)
                    if n > 1e-12:
                        g[a] += diff / n
                        g[b] -= diff / n
        # Wall subgradient: push inward the closest wall(s).
        wall = 0.5 - np.abs(cur)
        wmin = wall.min()
        for k in range(3):
            for j in range(N):
                if wall[j, k] < wmin + 1e-6:
                    g[j, k] += np.sign(cur[j, k])  # move away from that wall
        g = g + rng.normal(0.0, 0.005, size=cur.shape)
        cand = cur + step * g
        cand_r = _score(cand)
        if cand_r >= cur_r - 1e-9:
            cur = cand
            cur_r = cand_r
            if cur_r > best_r:
                best_r = cur_r
                best = cur.copy()
        if i % 50 == 0 and time.time() > _DEADLINE:
            break
    return best, best_r


def construct():
    global _T0
    rng = np.random.default_rng(12345)

    # Start from a 4x4x4 grid (the seed's arrangement) plus jittered starts.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    base = grid[:N].copy()
    base_r = _score(base)
    best = base.copy()
    best_r = base_r

    starts = [base]
    # A few random valid starts (jittered grid) to escape the local optimum.
    for _ in range(4):
        s = base + rng.normal(0.0, 0.03, size=base.shape)
        # Keep inside the cube.
        s = np.clip(s, -0.5 + 1e-6, 0.5 - 1e-6)
        starts.append(s)

    for s in starts:
        if time.time() > _DEADLINE:
            break
        a, ar = _anneal(s, T0=0.05, T1=0.001, steps=4000, rng=rng)
        if ar > best_r:
            best_r, best = ar, a
        if time.time() > _DEADLINE:
            break
        g, gr = _gradient_ascent(a, iters=2000, step0=0.01, step1=0.0005, rng=rng)
        if gr > best_r:
            best_r, best = gr, g

    return best
# EVOLVE-BLOCK-END
