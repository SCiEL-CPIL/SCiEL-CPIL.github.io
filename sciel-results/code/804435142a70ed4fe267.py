"""Circle packing in unit square: 26 circles, maximize sum of radii.
Combined strategy:
  - Physics-based force-directed relaxation with growing radii (from inspiration)
  - Delaunay contact-graph constrained SLSQP (from current)
  - Full-pairwise penalty polish with escalating weights (from inspiration)
  - Diverse initializations + local search perturbations
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
DEADLINE = 145.0  # safety margin below 150s kill


def _valid_check(arr, tol=1e-7):
    """Check all circles are in-square and non-overlapping."""
    n = len(arr)
    for i in range(n):
        x, y, r = arr[i]
        if r < -tol:
            return False
        if x < r + tol or x > 1 - r - tol or y < r + tol or y > 1 - r - tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(arr[i, 0] - arr[j, 0], arr[i, 1] - arr[j, 1])
            if d < arr[i, 2] + arr[j, 2] - tol:
                return False
    return True


def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _delaunay_edges(pos):
    """Return set of (i,j) pairs that are Delaunay neighbors."""
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = simplex[a], simplex[b]
                    edges.add((min(i, j), max(i, j)))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _physics_relax(xyz, time_limit=4.0, grow_rate=0.0015):
    """
    Force-directed physics relaxation with growing radii.
    Primary optimization engine: repulsion handles overlaps naturally.
    """
    t0 = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()

    best_score = float(np.sum(r))
    best_xyz = xyz.copy()

    dt = 0.004
    damp = 0.92
    vx = np.zeros(N)
    vy = np.zeros(N)

    for it in range(8000):
        if time.time() - t0 > time_limit:
            break

        # Grow radii
        r *= (1.0 + grow_rate)
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)

        # Forces
        fx = np.zeros(N)
        fy = np.zeros(N)

        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) + 1e-12
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    force = (min_dist - dist) * 80.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist

        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 200.0
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 200.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 200.0
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 200.0

        # Velocity update with damping
        vx = damp * vx + dt * fx
        vy = damp * vy + dt * fy
        x += vx
        y += vy

        # Clip
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1.0 - r[i])
            y[i] = np.clip(y[i], r[i], 1.0 - r[i])

        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r

        if _valid_check(xyz):
            s = float(np.sum(r))
            if s > best_score:
                best_score = s
                best_xyz = xyz.copy()

    return best_xyz


def _slsqp_step(x0, maxiter=150):
    """One SLSQP optimization step with Delaunay-based constraints."""
    pos0 = x0[:2*N].reshape(N, 2)
    edges = _delaunay_edges(pos0)

    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N + i] - x[2*i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N + i] - x[2*i+1]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N + i]})
    for (i, j) in edges:
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j:
                     np.hypot(x[2*i] - x[2*j], x[2*i+1] - x[2*j+1]) - x[2*N+i] - x[2*N+j]})

    def obj(x):
        return -np.sum(x[2*N:])

    def jac(x):
        g = np.zeros_like(x)
        g[2*N:] = -1.0
        return g

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))

    try:
        res = minimize(obj, x0, jac=jac, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-10})
        return res.x
    except Exception:
        return x0


def _penalty_polish(xyz, time_budget=6.0):
    """Full-pairwise penalty polish with escalating weights."""
    t0 = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_val = -float(np.sum(xyz[:, 2]))

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))

    for w in [500.0, 5000.0, 50000.0]:
        if time.time() - t0 > time_budget:
            break

        def obj(p, w=w):
            x = p[0::3]
            y = p[1::3]
            r = p[2::3]
            pen = 0.0
            pen += np.sum(np.maximum(0, -(x - r))**2)
            pen += np.sum(np.maximum(0, x + r - 1)**2)
            pen += np.sum(np.maximum(0, -(y - r))**2)
            pen += np.sum(np.maximum(0, y + r - 1)**2)
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    ov = r[i] + r[j] - d
                    if ov > 0:
                        pen += ov * ov
            return -np.sum(r) + w * pen

        try:
            res = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            if _valid_check(res.x.reshape(N, 3)):
                s = float(np.sum(res.x[2::3]))
                if s > -best_val:
                    best_val = -s
                    best_params = res.x.copy()
        except Exception:
            pass

    return best_params.reshape(N, 3)


def _enforce_validity(params):
    """Shrink radii and clip to make a configuration valid."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    for _ in range(200):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i, j = iu[0][k], iu[1][k]
            ov = overlap[k]
            if p[i, 2] <= p[j, 2]:
                sh = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= sh
                rem = ov - sh
                if rem > 0:
                    p[j, 2] -= min(rem, p[j, 2] - 1e-6)
            else:
                sh = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= sh
                rem = ov - sh
                if rem > 0:
                    p[i, 2] -= min(rem, p[i, 2] - 1e-6)
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    return p


def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    best = None
    best_score = -1.0

    def consider(pos, rad):
        nonlocal best, best_score
        arr = np.column_stack([pos, rad])
        if not _valid_check(arr):
            arr = _enforce_validity(arr)
            if not _valid_check(arr):
                return
        s = float(arr[:, 2].sum())
        if s > best_score:
            best_score = s
            best = arr.copy()

    # ---- Phase A: Diverse initializations ----
    starts = []

    # Grid initializations
    for side in range(4, 9):
        if time.time() > DEADLINE - 40:
            break
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
        cells = cells[:N]
        pos = np.array(cells, dtype=float)
        rad = _max_radius(pos, np.full(N, 0.5 / side))
        starts.append((pos.copy(), rad.copy()))

        # Perturbed
        for k in range(2):
            if time.time() > DEADLINE - 40:
                break
            p0 = pos + 0.04 * (rng.random((N, 2)) * 2 - 1)
            p0 = np.clip(p0, 0.05, 0.95)
            r0 = _max_radius(p0, np.full(N, 0.5 / side))
            starts.append((p0, r0))

    # Hexagonal
    if time.time() <= DEADLINE - 40:
        hex_pos = []
        r_hex = 0.12
        dx = 2 * r_hex
        dy = r_hex * np.sqrt(3)
        for row in range(8):
            for col in range(8):
                x = col * dx + (row % 2) * dx / 2 + 0.15
                y = row * dy + 0.15
                if 0.05 < x < 0.95 and 0.05 < y < 0.95:
                    hex_pos.append((x, y))
        hex_pos = hex_pos[:N]
        if len(hex_pos) >= N:
            hp = np.array(hex_pos, dtype=float)
            hr = _max_radius(hp, np.full(N, 0.1))
            starts.append((hp, hr))

    # Random
    for k in range(6):
        if time.time() > DEADLINE - 40:
            break
        p0 = rng.random((N, 2)) * 0.8 + 0.1
        r0 = _max_radius(p0, np.full(N, 0.05))
        starts.append((p0, r0))

    # ---- Phase B: Physics relaxation from each start ----
    for idx, (pos0, rad0) in enumerate(starts):
        if time.time() > DEADLINE - 25:
            break
        budget = min(5.0, max(2.0, (DEADLINE - 25 - time.time()) / max(1, len(starts) - idx)))
        xyz = np.column_stack([pos0, rad0])
        xyz_relaxed = _physics_relax(xyz, time_limit=budget)
        consider(xyz_relaxed[:, :2], xyz_relaxed[:, 2])

    # ---- Phase C: SLSQP + polish on best ----
    if best is not None:
        pos_b = best[:, :2].copy()
        rad_b = best[:, 2].copy()

        # SLSQP step
        x = np.concatenate([pos_b.reshape(-1), rad_b])
        x_new = _slsqp_step(x, maxiter=200)
        pos_new = x_new[:2*N].reshape(N, 2)
        rad_new = np.maximum(x_new[2*N:], 0.0)
        consider(pos_new, rad_new)

        # Penalty polish
        if best is not None and time.time() < DEADLINE - 10:
            xyz_pol = _penalty_polish(best, time_budget=8.0)
            consider(xyz_pol[:, :2], xyz_pol[:, 2])

    # ---- Phase D: Local search perturbation ----
    if best is not None and time.time() < DEADLINE - 15:
        for k in range(8):
            if time.time() > DEADLINE - 8:
                break
            pos_p = best[:, :2].copy()
            rad_p = best[:, 2].copy()
            n_perturb = rng.integers(3, 8)
            idx_perturb = rng.choice(N, size=n_perturb, replace=False)
            pos_p[idx_perturb] += 0.015 * (rng.random((n_perturb, 2)) * 2 - 1)
            pos_p = np.clip(pos_p, 0.02, 0.98)
            rad_p = _max_radius(pos_p, rad_p)
            xyz_p = np.column_stack([pos_p, rad_p])
            xyz_rel = _physics_relax(xyz_p, time_limit=3.0)
            consider(xyz_rel[:, :2], xyz_rel[:, 2])

    # ---- Final: one more penalty polish on absolute best ----
    if best is not None and time.time() < DEADLINE - 5:
        xyz_final = _penalty_polish(best, time_budget=5.0)
        consider(xyz_final[:, :2], xyz_final[:, 2])

    if best is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
