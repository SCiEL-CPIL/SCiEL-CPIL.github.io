"""Circle packing in unit square - 26 circles maximizing sum of radii.
Combines compact structural parameterization (corner + center + rings)
with high-precision L-BFGS-B polish and differential evolution.
Uses a smoother combined penalty (quadratic + log-sum-exp soft-max)."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x, deadline):
    """Objective: minimize negative sum of radii + smooth combined penalty."""
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    # Boundary violations (vectorized)
    left = radii - pos[:, 0]
    right = pos[:, 0] - (1.0 - radii)
    bot = radii - pos[:, 1]
    top = pos[:, 1] - (1.0 - radii)

    # Overlap violations (vectorized via distance matrix)
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = min_dist - dist
    np.fill_diagonal(viol, 0.0)

    # Stack all violation magnitudes
    all_v = np.concatenate([left, right, bot, top, viol.ravel()])

    # Quadratic penalty on positive violations
    pos_viol = np.maximum(all_v, 0.0)
    quad_penalty = np.sum(pos_viol ** 2)

    # Log-sum-exp soft-max on the worst violation
    beta = 60.0
    m = np.max(all_v)
    if m > 0:
        shifted = beta * (all_v - m)
        sm = (np.log(np.sum(np.exp(shifted))) + beta * m) / beta
    else:
        sm = 0.0

    # Combined penalty
    penalty = 50.0 * quad_penalty + 20.0 * sm

    return -np.sum(radii) + penalty


# ---------------- Compact structural parameterization ----------------
# p = [corner_r, center_r, cx, cy,
#      ring1_off, ring1_r, ring1_r2,
#      ring2_off, ring2_r, ring2_r2]
# ring1: 8 circles, ring2: 13 circles, plus 4 corners + 1 center = 26.

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

    # ring 1: 8 circles (indices 5..12)
    for i in range(8):
        ang = 2 * np.pi * i / 8.0
        pos[5 + i] = [cx + r1_off * np.cos(ang), cy + r1_off * np.sin(ang)]
        radii[5 + i] = r1_r if i % 2 == 0 else r1_r2

    # ring 2: 13 circles (indices 13..25)
    for i in range(13):
        ang = 2 * np.pi * i / 13.0 + 0.1
        pos[13 + i] = [cx + r2_off * np.cos(ang), cy + r2_off * np.sin(ang)]
        radii[13 + i] = r2_r if i % 2 == 0 else r2_r2

    return pos, radii


def _struct_objective(p, deadline):
    pos, radii = _expand_struct(p)
    x = np.concatenate([pos.flatten(), radii])
    return _objective(x, deadline)


# ---------------- Structural seeds ----------------

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


# ---------------- Post-processing & polish ----------------

def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(500):
        if time.time() > deadline:
            break
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.999
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    if radii[i] <= radii[j]:
                        needed = (min_dist - dist) / (2.0 * radii[i] + 1e-12)
                        radii[i] *= max(0.5, 1.0 - needed * 0.999)
                    else:
                        needed = (min_dist - dist) / (2.0 * radii[j] + 1e-12)
                        radii[j] *= max(0.5, 1.0 - needed * 0.999)
                    valid = False
        if valid:
            break

    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))

    return pos, radii


def _polish(pos, radii, deadline):
    """High-precision L-BFGS-B polish with tight bounds and tolerances."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
        bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
    for i in range(N):
        bounds.append((1e-4, 0.35))

    try:
        result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-14})
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = result.x[2 * N:]
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass

    pos_pp, radii_pp = _post_process(pos, radii, deadline)
    return pos_pp, radii_pp


def _struct_polish(p, deadline):
    """Polish the structural parameters using L-BFGS-B."""
    p = p.copy()
    bounds = [
        (0.05, 0.30),
        (0.05, 0.35),
        (0.2, 0.8),
        (0.2, 0.8),
        (0.15, 0.6),
        (0.03, 0.20),
        (0.03, 0.20),
        (0.25, 0.7),
        (0.02, 0.16),
        (0.02, 0.16),
    ]
    try:
        result = minimize(lambda q: _struct_objective(q, deadline), p, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
        p_opt = result.x
        pos, radii = _expand_struct(p_opt)
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass

    pos, radii = _expand_struct(p)
    pos_pp, radii_pp = _post_process(pos, radii, deadline)
    return pos_pp, radii_pp


def _run_differential_evolution(deadline):
    """Run differential evolution on the structural parameters."""
    bounds = [
        (0.05, 0.30),
        (0.05, 0.35),
        (0.2, 0.8),
        (0.2, 0.8),
        (0.15, 0.6),
        (0.03, 0.20),
        (0.03, 0.20),
        (0.25, 0.7),
        (0.02, 0.16),
        (0.02, 0.16),
    ]

    try:
        result = differential_evolution(
            lambda p: _struct_objective(p, deadline),
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False,
            update_interval=10,
            workers=-1
        )
        if time.time() < deadline:
            p_opt = result.x
            pos, radii = _expand_struct(p_opt)
            pos_pp, radii_pp = _post_process(pos, radii, deadline)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130

    best_score = 0.0
    best_pos = None
    best_radii = None

    # Phase 1: Try multiple initializations with polish
    inits = [
        _make_grid(N),
        _make_hexagonal(N),
        _make_center_ring(N),
        _make_random(N, 42),
        _make_random(N, 123),
        _make_random(N, 456),
    ]

    for pos, radii in inits:
        if time.time() > deadline - 20:
            break
        try:
            pos_pp, radii_pp = _polish(pos, radii, deadline)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp
                    best_radii = radii_pp
        except Exception:
            continue

    # Phase 2: Try differential evolution on structural parameters
    if time.time() < deadline - 30:
        try:
            de_result = _run_differential_evolution(deadline)
            if de_result is not None:
                pos_de, radii_de = de_result
                if _is_valid(pos_de, radii_de):
                    score_de = np.sum(radii_de)
                    if score_de > best_score:
                        best_score = score_de
                        best_pos = pos_de
                        best_radii = radii_de
        except Exception:
            pass

    # Phase 3: Polish the best result found
    if best_pos is not None and time.time() < deadline - 5:
        try:
            pos_final, radii_final = _polish(best_pos, best_radii, deadline)
            if _is_valid(pos_final, radii_final):
                score_final = np.sum(radii_final)
                if score_final > best_score:
                    best_score = score_final
                    best_pos = pos_final
                    best_radii = radii_final
        except Exception:
            pass

    # Fallback: return grid if nothing worked
    if best_pos is None:
        pos, radii = _make_grid(N)
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        return np.column_stack([pos_pp, radii_pp])

    # Ensure final validity
    pos_pp, radii_pp = _post_process(best_pos, best_radii, deadline)
    if not _is_valid(pos_pp, radii_pp):
        pos, radii = _make_grid(N)
        pos_pp, radii_pp = _post_process(pos, radii, deadline)

    return np.column_stack([pos_pp, radii_pp])

# EVOLVE-BLOCK-END
