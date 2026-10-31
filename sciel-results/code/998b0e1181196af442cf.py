"""Circle packing in unit square - 26 circles maximizing sum of radii.
Rewritten with a stronger, faster, and more robust multi-phase optimizer:
  * Fast vectorized penalty objective (no Python loops).
  * Multi-start local search (L-BFGS-B) from diverse, symmetry-aware seeds.
  * Compact structural DE (10-dim) for global exploration.
  * Direct 78-dim local search from the best structural seed.
  * A fast, vectorized force-directed / radius-growth phase for topological
    rearrangement and packing tightening.
  * A final high-precision polish (L-BFGS-B with very tight tolerances) and an
    SLSQP constraint-based cross-check, all guarded by an explicit deadline.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    # Boundary check
    if np.any(pos - radii < -tol) or np.any(pos + radii > 1.0 + tol):
        return False
    # Overlap check (vectorized)
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    if np.any(dist < min_dist - tol):
        return False
    return True


def _objective(x, deadline):
    """Objective: minimize -sum(radii) + smooth penalty. Fully vectorized."""
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    eps = 1e-7

    def _soft(v):
        return (v + np.sqrt(v * v + eps)) * 0.5

    # Boundary violations
    left = _soft(radii - pos[:, 0])
    right = _soft(pos[:, 0] - (1.0 - radii))
    bot = _soft(radii - pos[:, 1])
    top = _soft(pos[:, 1] - (1.0 - radii))
    bpen = np.sum(left * left) + np.sum(right * right) + np.sum(bot * bot) + np.sum(top * top)

    # Overlap violations
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = min_dist - dist
    np.fill_diagonal(viol, 0.0)
    viol_s = _soft(viol)
    np.fill_diagonal(viol_s, 0.0)
    open_pen = 0.5 * np.sum(viol_s * viol_s)

    return -np.sum(radii) + 100.0 * (bpen + open_pen)


def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity (fast, vectorized)."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(300):
        if time.time() > deadline:
            break
        # Boundary clamp
        max_r = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                           np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        too_big = radii > max_r
        if np.any(too_big):
            radii[too_big] = max_r[too_big] * 0.999

        # Overlap resolution (vectorized)
        d = pos[:, 0][:, None] - pos[:, 0][None, :]
        e = pos[:, 1][:, None] - pos[:, 1][None, :]
        dist = np.sqrt(d * d + e * e)
        min_dist = radii[:, None] + radii[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0.0)

        overlap = min_dist - dist
        np.fill_diagonal(overlap, 0.0)
        # Only upper triangle matters; but we can check all
        has_overlap = np.any(overlap > 1e-12)
        if not has_overlap:
            break

        # Shrink the smaller circle in each overlapping pair
        # For each pair (i,j) with overlap, shrink min(r_i, r_j)
        iu = np.triu_indices(N, k=1)
        for i, j in zip(iu[0], iu[1]):
            if overlap[i, j] > 1e-12:
                if radii[i] <= radii[j]:
                    needed = overlap[i, j] / (2.0 * radii[i] + 1e-12)
                    radii[i] *= max(0.5, 1.0 - needed * 0.999)
                else:
                    needed = overlap[i, j] / (2.0 * radii[j] + 1e-12)
                    radii[j] *= max(0.5, 1.0 - needed * 0.999)

    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))

    return pos, radii


# ---------------- Compact structural parameterization ----------------

def _expand_struct(p):
    """Map a 10-dim structural vector to (pos, radii) of 26 circles."""
    corner_r = np.clip(p[0], 0.05, 0.30)
    center_r = np.clip(p[1], 0.05, 0.35)
    cx = np.clip(p[2], 0.2, 0.8)
    cy = np.clip(p[3], 0.2, 0.8)
    r1_off = np.clip(p[4], 0.15, 0.6)
    r1_r = np.clip(p[5], 0.03, 0.20)
    r1_r2 = np.clip(p[6], 0.03, 0.20)
    r2_off = np.clip(p[7], 0.25, 0.7)
    r2_r = np.clip(p[8], 0.02, 0.16)
    r2_r2 = np.clip(p[9], 0.02, 0.16)

    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    # 4 corners
    pos[0] = [corner_r, corner_r]
    pos[1] = [1.0 - corner_r, corner_r]
    pos[2] = [corner_r, 1.0 - corner_r]
    pos[3] = [1.0 - corner_r, 1.0 - corner_r]
    radii[0:4] = corner_r

    # center
    pos[4] = [cx, cy]
    radii[4] = center_r

    # ring 1: 8 circles
    for i in range(8):
        ang = 2 * np.pi * i / 8.0
        pos[5 + i] = [cx + r1_off * np.cos(ang), cy + r1_off * np.sin(ang)]
        radii[5 + i] = r1_r if i % 2 == 0 else r1_r2

    # ring 2: 13 circles
    for i in range(13):
        ang = 2 * np.pi * i / 13.0 + 0.1
        pos[13 + i] = [cx + r2_off * np.cos(ang), cy + r2_off * np.sin(ang)]
        radii[13 + i] = r2_r if i % 2 == 0 else r2_r2

    return pos, radii


def _struct_objective(p, deadline):
    pos, radii = _expand_struct(p)
    x = np.concatenate([pos.flatten(), radii])
    return _objective(x, deadline)


# ---------------- Seeds ----------------

def _make_grid(n, seed=42):
    side = 5
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < n:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.005
    return pos, radii


def _make_hexagonal(n, seed=42):
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    r = 0.1
    rows = 5
    for row in range(rows):
        y = 0.1 + row * r * np.sqrt(3)
        if row % 2 == 0:
            xs = np.arange(5) * 2 * r + r
        else:
            xs = np.arange(5) * 2 * r + 2 * r
        for col, x in enumerate(xs):
            idx = row * 5 + col
            if idx < n:
                pos[idx] = [x, y]
                radii[idx] = r
    return pos, radii


def _make_center_ring(n, seed=42):
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.15
    r_inner = 0.12
    dist_inner = 0.3
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pos[i + 1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
        radii[i + 1] = r_inner
    r_outer = 0.08
    dist_outer = 0.42
    for i in range(17):
        angle = 2 * np.pi * i / 17
        pos[i + 9] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
        radii[i + 9] = r_outer
    return pos, radii


def _make_random(n, seed=42):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = rng.uniform(0.03, 0.1, size=n)
    return pos, radii


def _polish(pos, radii, deadline, maxiter=3000, ftol=1e-18, gtol=1e-16):
    """High-precision L-BFGS-B polish."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, radii[i] * 0.3), min(1.0, 1.0 - radii[i] * 0.3)))
        bounds.append((max(0.0, radii[i] * 0.3), min(1.0, 1.0 - radii[i] * 0.3)))
    for i in range(N):
        bounds.append((1e-4, 0.35))

    try:
        result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = result.x[2 * N:]
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass
    return pos, radii


def _slsqp_polish(pos, radii, deadline):
    """Constraint-based SLSQP polish."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])

    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-4, 0.35)] * N

    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i] - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2 * i]) - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i + 1] - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2 * i + 1]) - x[2 * N + i]})
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i, j=j:
                         np.sqrt((x[2 * i] - x[2 * j]) ** 2 +
                                 (x[2 * i + 1] - x[2 * j + 1]) ** 2)
                         - x[2 * N + i] - x[2 * N + j]})

    def neg_sum_rad(x):
        if time.time() > deadline:
            return 1e10
        return -np.sum(x[2 * N:])

    try:
        result = minimize(neg_sum_rad, x0, method='SLSQP', bounds=bounds,
                          constraints=cons, options={'maxiter': 500, 'ftol': 1e-12})
        if result.x is not None:
            pos_opt = result.x[:2 * N].reshape(N, 2)
            radii_opt = result.x[2 * N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    return pos, radii


def _force_directed_anneal(pos, radii, deadline, max_iter=1500, rng_seed=123):
    """Force-directed annealing with radius growth."""
    pos = pos.copy()
    radii = radii.copy()
    rng = np.random.RandomState(rng_seed)

    T = 0.05
    cooling = 0.999

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    for it in range(max_iter):
        if time.time() > deadline:
            break
        if T < 1e-6:
            break

        forces = np.zeros_like(pos)

        # Boundary forces
        for i in range(N):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - pos[i, 0]) * 10.0
            if pos[i, 0] > 1.0 - radii[i]:
                forces[i, 0] -= (pos[i, 0] - (1.0 - radii[i])) * 10.0
            if pos[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - pos[i, 1]) * 10.0
            if pos[i, 1] > 1.0 - radii[i]:
                forces[i, 1] -= (pos[i, 1] - (1.0 - radii[i])) * 10.0

        # Repulsive / attractive forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    f = overlap * 50.0 / dist
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy
                elif dist > min_dist and dist < min_dist * 1.5:
                    f = (min_dist * 1.5 - dist) * 0.1 / dist
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy

        for i in range(N):
            norm_f = np.linalg.norm(forces[i]) + 1e-12
            scale = 0.01 * T * (1.0 / (0.5 + radii[i]))
            delta = scale * (forces[i] / norm_f + rng.randn(2) * 0.5)
            pos[i] += delta
            pos[i, 0] = np.clip(pos[i, 0], 0.001, 0.999)
            pos[i, 1] = np.clip(pos[i, 1], 0.001, 0.999)

        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
            for i in range(N):
                max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                grow_factor = 1.0 + 0.002 * (1.0 - radii[i])
                radii[i] = min(radii[i] * grow_factor, max_r * 0.999)

        T *= cooling

    return best_pos, best_radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

    best_pos = None
    best_radii = None
    best_score = -np.inf

    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        if time.time() > deadline:
            return
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    struct_bounds = [
        (0.06, 0.28),
        (0.06, 0.32),
        (0.3, 0.7),
        (0.3, 0.7),
        (0.18, 0.45),
        (0.04, 0.16),
        (0.04, 0.16),
        (0.30, 0.55),
        (0.03, 0.12),
        (0.03, 0.12),
    ]

    # ---------- Track A: DE over structural space ----------
    if time.time() < deadline - 60:
        try:
            de_res = differential_evolution(_struct_objective, struct_bounds,
                                            args=(deadline,),
                                            maxiter=150, popsize=25, seed=7,
                                            tol=1e-9, polish=False,
                                            mutation=(0.4, 1.0), recombination=0.7,
                                            updating='deferred')
            if de_res.x is not None:
                pos_s, radii_s = _expand_struct(de_res.x)
                update_best(pos_s, radii_s)
                if time.time() < deadline - 35:
                    pos_p, radii_p = _polish(pos_s, radii_s, deadline)
                    if _is_valid(pos_p, radii_p):
                        s = np.sum(radii_p)
                        if s > best_score:
                            best_score = s
                            best_pos = pos_p.copy()
                            best_radii = radii_p.copy()
        except Exception:
            pass

    # Second DE run
    if time.time() < deadline - 40:
        try:
            de_res2 = differential_evolution(_struct_objective, struct_bounds,
                                             args=(deadline,),
                                             maxiter=100, popsize=20, seed=123,
                                             tol=1e-8, polish=False,
                                             mutation=(0.6, 1.2), recombination=0.8,
                                             updating='immediate')
            if de_res2.x is not None:
                pos_s2, radii_s2 = _expand_struct(de_res2.x)
                update_best(pos_s2, radii_s2)
                if time.time() < deadline - 25:
                    pos_p2, radii_p2 = _polish(pos_s2, radii_s2, deadline)
                    if _is_valid(pos_p2, radii_p2):
                        s2 = np.sum(radii_p2)
                        if s2 > best_score:
                            best_score = s2
                            best_pos = pos_p2.copy()
                            best_radii = radii_p2.copy()
        except Exception:
            pass

    # L-BFGS-B on structural seeds
    struct_seeds = [
        [0.13, 0.16, 0.5, 0.5, 0.30, 0.11, 0.11, 0.42, 0.07, 0.07],
        [0.12, 0.18, 0.5, 0.5, 0.33, 0.10, 0.10, 0.45, 0.06, 0.06],
        [0.15, 0.14, 0.5, 0.5, 0.28, 0.12, 0.12, 0.40, 0.08, 0.08],
        [0.14, 0.15, 0.5, 0.5, 0.31, 0.105, 0.105, 0.43, 0.065, 0.065],
        [0.11, 0.20, 0.5, 0.5, 0.35, 0.09, 0.09, 0.48, 0.05, 0.05],
        [0.16, 0.12, 0.5, 0.5, 0.26, 0.13, 0.13, 0.38, 0.09, 0.09],
    ]
    for s0 in struct_seeds:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_struct_objective, np.array(s0), args=(deadline,),
                           method='L-BFGS-B', bounds=struct_bounds,
                           options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12})
            pos_s, radii_s = _expand_struct(res.x)
            update_best(pos_s, radii_s)
        except Exception:
            pass

    # ---------- Track B: Direct 78-dim local search from good seeds ----------
    if time.time() < deadline - 25:
        configs = []
        pos_s0, radii_s0 = _expand_struct(np.array(struct_seeds[0]))
        configs.append((pos_s0, radii_s0))
        configs.append(_make_center_ring(N))
        configs.append(_make_hexagonal(N))

        for pos, radii in configs:
            if time.time() > deadline - 15:
                break
            x0 = np.concatenate([pos.flatten(), radii])
            bounds = []
            for i in range(N):
                bounds.append((0.01, 0.99))
                bounds.append((0.01, 0.99))
            for i in range(N):
                bounds.append((0.01, 0.32))
            try:
                result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                                  bounds=bounds, options={'maxiter': 500, 'ftol': 1e-11})
                if result.success or result.fun < _objective(x0, deadline):
                    pos_opt = result.x[:2 * N].reshape(N, 2)
                    radii_opt = result.x[2 * N:]
                    update_best(pos_opt, radii_opt)
            except Exception:
                pass

    # ---------- Track C: Force-directed annealing ----------
    if best_pos is not None and time.time() < deadline - 15:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, deadline,
                                                       max_iter=1200, rng_seed=456)
            if _is_valid(pos_fd, radii_fd):
                s = np.sum(radii_fd)
                if s > best_score:
                    best_score = s
                    best_pos = pos_fd.copy()
                    best_radii = radii_fd.copy()
        except Exception:
            pass

    # ---------- Final high-precision polish ----------
    if best_pos is not None and time.time() < deadline - 6:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline, maxiter=4000, ftol=1e-20, gtol=1e-18)
        if _is_valid(pos_p, radii_p):
            s = np.sum(radii_p)
            if s > best_score:
                best_score = s
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()

    # ---------- SLSQP cross-check ----------
    if best_pos is not None and time.time() < deadline - 2:
        pos_q, radii_q = _slsqp_polish(best_pos, best_radii, deadline)
        if _is_valid(pos_q, radii_q):
            s = np.sum(radii_q)
            if s > best_score:
                best_score = s
                best_pos = pos_q.copy()
                best_radii = radii_q.copy()

    # Fallback
    if best_pos is None:
        pos, radii = _make_grid(N)
        best_pos, best_radii = _post_process(pos, radii, deadline)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
