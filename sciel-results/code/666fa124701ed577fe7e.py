"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.

MATERIALLY DIFFERENT STRATEGY:
- Represent the whole configuration (positions + radii) as one continuous vector.
- Use Simulated Annealing (SA) with a hard-constraint penalty (quadratic + linear)
  to move centers AND radii together, so the optimizer can trade off radius for
  better placement instead of the current "freeze positions, greedy-grow radii".
- Periodically call a Delaunay-contact-graph-aware SLSQP micro-polish that only
  exposes the *active* constraints (Delaunay edges + wall contacts) to the solver,
  which is much faster than all-pair constraints and lets SA escape shallow traps.
- Multi-restart from diverse structured/random initializations; keep best valid.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-9
DEADLINE = 140.0  # safety margin before 150s kill


def _violation(p, r):
    """Return (max_violation, total_penalty) for a configuration."""
    # wall violations
    wall = np.maximum(
        np.maximum(p[:, 0] - (1.0 - r), p[:, 1] - (1.0 - r)),
        np.maximum(r - p[:, 0], r - p[:, 1])
    )
    # pairwise overlap violations
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    overlap = np.maximum(r[:, None] + r[None, :] - d, 0.0)
    np.fill_diagonal(overlap, 0.0)
    max_v = max(wall.max(), overlap.max())
    pen = 10.0 * (wall.sum() ** 2 + overlap.sum() ** 2) + 50.0 * (wall.sum() + overlap.sum())
    return max_v, pen


def _valid_check(p, r):
    """Return (arr, score) if all N circles are valid, else (None, -1)."""
    for i in range(N):
        if r[i] <= 0:
            return None, -1.0
        if p[i, 0] < r[i] + EPS or p[i, 0] > 1.0 - r[i] - EPS:
            return None, -1.0
        if p[i, 1] < r[i] + EPS or p[i, 1] > 1.0 - r[i] - EPS:
            return None, -1.0
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    if np.any(d < (r[:, None] + r[None, :]) - 1e-9):
        return None, -1.0
    arr = np.column_stack([p, r])
    return arr, float(r.sum())


def _delaunay_edges(p):
    """Return set of (i,j) pairs that are Delaunay neighbors (potential contacts)."""
    try:
        tri = Delaunay(p)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = simplex[a], simplex[b]
                    if i < j:
                        edges.add((i, j))
        return edges
    except Exception:
        # fallback: all pairs (expensive but correct)
        return {(i, j) for i in range(N) for j in range(i + 1, N)}


def _slsqp_polish(p, r, deadline):
    """Delaunay-contact-graph-aware SLSQP polish. Only active constraints exposed."""
    n = N
    x0 = np.concatenate([p.reshape(-1), r])

    def f(v):
        pp = v[:2 * n].reshape(n, 2)
        rr = v[2 * n:]
        return -rr.sum()

    # Build constraints from Delaunay edges + walls
    edges = _delaunay_edges(p)
    cons = []
    for (i, j) in edges:
        cons.append({
            'type': 'ineq',
            'fun': (lambda v, i=i, j=j:
                    np.hypot(v[2 * i] - v[2 * j], v[2 * i + 1] - v[2 * j + 1])
                    - v[2 * n + i] - v[2 * n + j])
        })
    for i in range(n):
        for ax in range(2):
            cons.append({'type': 'ineq',
                         'fun': (lambda v, i=i, ax=ax: v[2 * i + ax] - v[2 * n + i])})
            cons.append({'type': 'ineq',
                         'fun': (lambda v, i=i, ax=ax: 1.0 - v[2 * i + ax] - v[2 * n + i])})

    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    try:
        res = minimize(f, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 30, 'ftol': 1e-10})
        if res.fun < f(x0):
            pp = res.x[:2 * n].reshape(n, 2)
            rr = res.x[2 * n:]
            return pp, rr
    except Exception:
        pass
    return p, r


def _sa_step(p, r, T, rng, step_scale):
    """One SA move: perturb positions and radii, accept/reject by Metropolis."""
    n = N
    # perturb positions
    dp = rng.normal(0, step_scale, size=(n, 2))
    # perturb radii (small relative perturbation)
    dr = rng.normal(0, 0.002, size=n)
    p_new = p + dp
    r_new = np.clip(r + dr, 1e-6, 0.5)

    # clamp positions to [0,1]
    p_new = np.clip(p_new, 0.0, 1.0)

    _, pen_old = _violation(p, r)
    _, pen_new = _violation(p_new, r_new)
    # energy = -sum(r) + penalty
    e_old = -r.sum() + pen_old
    e_new = -r_new.sum() + pen_new
    delta_e = e_new - e_old
    if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-12)):
        return p_new, r_new
    return p, r


def _grow_exact(p, r):
    """Set each radius to its exact maximal value given positions (one pass)."""
    n = N
    r_new = np.empty(n)
    for i in range(n):
        dmin = min(p[i, 0], 1.0 - p[i, 0], p[i, 1], 1.0 - p[i, 1])
        for j in range(n):
            if i != j:
                dmin = min(dmin, np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1]) - r[j])
        r_new[i] = max(0.0, dmin)
    return r_new


def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    best = None
    best_score = -1.0

    def consider(p, r):
        nonlocal best, best_score
        arr, s = _valid_check(p, r)
        if arr is not None and s > best_score:
            best_score = s
            best = arr

    # --- Initializations ---
    def grid(side):
        cells = [((i + 0.5) / side, (j + 0.5) / side)
                 for i in range(side) for j in range(side)][:N]
        return np.array(cells, dtype=float)

    def hex_layout():
        pts = []
        rows = 5
        for row in range(rows):
            y = (row + 0.5) / rows
            offset = 0.5 / rows if row % 2 else 0.0
            for col in range(rows):
                x = (col + 0.5) / rows + offset
                if 0 <= x <= 1:
                    pts.append((x, y))
            if len(pts) >= N:
                break
        return np.array(pts[:N], dtype=float)

    def rings():
        pts = [(0.5, 0.5)]
        for rad, cnt in [(0.2, 6), (0.35, 12), (0.44, 10)]:
            for k in range(cnt):
                a = 2 * np.pi * k / cnt
                pts.append((0.5 + rad * np.cos(a), 0.5 + rad * np.sin(a)))
            if len(pts) >= N:
                break
        return np.array(pts[:N], dtype=float)

    starts = []
    for side in range(3, 8):
        starts.append(grid(side))
    starts.append(hex_layout())
    starts.append(rings())
    for _ in range(8):
        starts.append(rng.random((N, 2)) * 0.84 + 0.08)

    # --- SA + periodic SLSQP polish loop ---
    for si, pos0 in enumerate(starts):
        if time.time() > DEADLINE - 15:
            break
        p = np.array(pos0, dtype=float).copy()
        # initial radii: small, will grow
        r = np.full(N, 0.5 / np.sqrt(N) * 0.8)

        # Initial exact growth to get a decent starting point
        for _ in range(50):
            r = _grow_exact(p, r)

        # SA parameters
        T = 0.05
        T_min = 0.0001
        cooling = 0.995
        step_scale = 0.02
        polish_every = 40

        for it in range(600):
            if time.time() > DEADLINE - 15:
                break
            p, r = _sa_step(p, r, T, rng, step_scale)
            T *= cooling
            if it % polish_every == 0:
                p, r = _slsqp_polish(p, r, DEADLINE)
                # Re-exact radii after polish
                r = _grow_exact(p, r)
            # Track best
            consider(p, r)

        # Final polish on this start
        if time.time() < DEADLINE - 10:
            p, r = _slsqp_polish(p, r, DEADLINE)
            r = _grow_exact(p, r)
            consider(p, r)

    # --- Final global polish on best ---
    if best is not None:
        pp, rr = best[:, :2], best[:, 2]
        for _ in range(3):
            if time.time() > DEADLINE - 5:
                break
            pp, rr = _slsqp_polish(pp, rr, DEADLINE)
            rr = _grow_exact(pp, rr)
            consider(pp, rr)

    if best is None:
        # Fallback: grid of 26 circles (5x5 + 1)
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)][:N]
        return np.array(cells, dtype=float)
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
