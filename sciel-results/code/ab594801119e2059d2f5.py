# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _sat_gap(vi, vj):
    """Return minimum SAT gap between two hexagons. Positive = separated, negative = overlapping."""
    all_normals = []
    for v in [vi, vj]:
        for e in range(6):
            p1 = v[e]
            p2 = v[(e + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-15:
                n = n / norm
                all_normals.append(n)

    min_gap = np.inf
    for n in all_normals:
        proj_i = vi @ n
        proj_j = vj @ n
        gap = max(proj_j.min() - proj_i.max(), proj_i.min() - proj_j.max())
        if gap > 0:
            min_gap = min(min_gap, gap)
        else:
            return gap
    return min_gap


def _check_validity(inner, outer, s):
    """
    Check all validity conditions. Returns (is_valid, min_violation).
    inner: (12,3) array (x,y,angle_deg)
    outer: (3,) array (x,y,angle_deg)
    s: float side length
    """
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer)) and np.isfinite(s)):
        return False, -1.0

    if s < np.sqrt(12.0) - 1e-9:
        return False, -1.0

    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-6:
                return False, -1.0

    for i in range(12):
        vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(i + 1, 12):
            vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            gap = _sat_gap(vi, vj)
            if gap < 1e-6:
                return False, -1.0

    for k in range(12):
        vi = _hex_vertices(inner[k, 0], inner[k, 1], 1.0, inner[k, 2])
        for v in vi:
            for e in range(6):
                p1 = ov[e]
                p2 = ov[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False, -1.0

    return True, 0.0


def _compute_outer_s(inner, cx, cy, ang):
    """Compute outer hexagon side length given inner positions and outer center/angle."""
    all_verts = []
    for k in range(12):
        v = _hex_vertices(inner[k, 0], inner[k, 1], 1.0, inner[k, 2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)

    a = np.deg2rad(ang)
    normals = np.array([[np.cos(a + np.deg2rad(30 + 60 * i)),
                         np.sin(a + np.deg2rad(30 + 60 * i))] for i in range(6)])

    dots = all_verts @ normals.T
    M = dots.max(axis=0)
    vals = M - np.array([cx, cy]) @ normals.T
    return vals.max() / (np.sqrt(3) / 2)


def _overlap_penalty(inner, threshold=0.005):
    """Compute penalty for overlapping inner hexagons. Returns 0 if all separated by threshold."""
    penalty = 0.0
    for i in range(12):
        vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(i + 1, 12):
            vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            gap = _sat_gap(vi, vj)
            if gap < threshold:
                penalty += (threshold - gap) ** 2
    return penalty


def _seed_config_hex_cluster():
    """Hexagonal cluster: center + 6 ring1 + 5 ring2"""
    sqrt3 = np.sqrt(3)
    return np.array([
        [0, 0, 0],
        [2, 0, 0],
        [-2, 0, 0],
        [1, sqrt3, 0],
        [-1, sqrt3, 0],
        [1, -sqrt3, 0],
        [-1, -sqrt3, 0],
        [3, sqrt3, 0],
        [-3, sqrt3, 0],
        [3, -sqrt3, 0],
        [-3, -sqrt3, 0],
        [0, 2 * sqrt3, 0],
    ])


def _seed_config_spread():
    """More spread out arrangement"""
    sqrt3 = np.sqrt(3)
    return np.array([
        [0, 0, 0],
        [2.2, 0, 0],
        [-2.2, 0, 0],
        [1.1, 1.9, 0],
        [-1.1, 1.9, 0],
        [1.1, -1.9, 0],
        [-1.1, -1.9, 0],
        [3.3, 1.9, 0],
        [-3.3, 1.9, 0],
        [3.3, -1.9, 0],
        [-3.3, -1.9, 0],
        [0, 3.8, 0],
    ])


def _seed_config_rotated():
    """Cluster with slight rotation of outer hexagons"""
    sqrt3 = np.sqrt(3)
    base = _seed_config_hex_cluster().copy()
    base[6:, 2] = 10.0
    return base


def _seed_config_diagonal():
    """Diagonal band arrangement"""
    sqrt3 = np.sqrt(3)
    return np.array([
        [0, 0, 0],
        [2, 0, 0],
        [4, 0, 0],
        [1, sqrt3, 0],
        [3, sqrt3, 0],
        [5, sqrt3, 0],
        [-1, sqrt3, 0],
        [2, 2*sqrt3, 0],
        [4, 2*sqrt3, 0],
        [-1, -sqrt3, 0],
        [1, -sqrt3, 0],
        [3, -sqrt3, 0],
    ])


def _seed_config_ring():
    """Ring of 12 hexagons around center, with center filled"""
    sqrt3 = np.sqrt(3)
    # 12 hexagons in a ring at radius ~3, plus we need 12 total
    # Let's do 6 inner ring + 6 outer ring
    inner_r = 2.0
    outer_r = 4.0
    config = []
    # 6 at inner radius
    for i in range(6):
        ang = 60 * i
        x = inner_r * math.cos(math.radians(ang))
        y = inner_r * math.sin(math.radians(ang))
        config.append([x, y, 0])
    # 6 at outer radius, offset by 30 degrees
    for i in range(6):
        ang = 60 * i + 30
        x = outer_r * math.cos(math.radians(ang))
        y = outer_r * math.sin(math.radians(ang))
        config.append([x, y, 0])
    return np.array(config)


def _sa_optimize(inner, deadline, rng, use_global_translation=True):
    """
    Simulated annealing to optimize inner hexagon positions (and their angles)
    while minimizing the required outer hexagon side length.
    """
    sqrt3 = np.sqrt(3)
    best_inner = inner.copy()
    best_s = np.inf
    best_ang = 0.0

    # Initial: find best outer angle for current inner
    for ang in np.arange(0, 60, 5.0):
        s_val = _compute_outer_s(inner, 0, 0, ang)
        if s_val < best_s:
            best_s = s_val
            best_ang = ang

    for ang in np.arange(0, 60, 0.5):
        s_val = _compute_outer_s(inner, 0, 0, ang)
        if s_val < best_s:
            best_s = s_val
            best_ang = ang

    # SA state
    current_inner = inner.copy()
    current_ang = best_ang
    current_s = best_s
    current_penalty = _overlap_penalty(current_inner)
    current_obj = current_s + current_penalty * 10.0

    T = 0.5  # Initial temperature - lower for more focused search
    T_min = 0.0005
    step_size = 0.2
    n_iter = 0
    max_iter = 500000
    accept_count = 0
    best_obj = current_obj
    no_improve_count = 0

    while n_iter < max_iter and time.time() < deadline - 5:
        n_iter += 1

        # Generate perturbation
        candidate = current_inner.copy()
        # Move 1-3 random hexagons
        n_move = rng.integers(1, 4)
        indices = rng.choice(12, size=n_move, replace=False)
        for idx in indices:
            candidate[idx, 0] += rng.normal(0, step_size)
            candidate[idx, 1] += rng.normal(0, step_size)
            candidate[idx, 2] += rng.normal(0, step_size * 8)
            candidate[idx, 2] = candidate[idx, 2] % 60.0

        # Small chance to perturb outer angle
        if rng.random() < 0.15:
            cand_ang = (current_ang + rng.normal(0, step_size * 3)) % 60.0
        else:
            cand_ang = current_ang

        # Evaluate
        cand_s = _compute_outer_s(candidate, 0, 0, cand_ang)
        cand_penalty = _overlap_penalty(candidate)
        cand_obj = cand_s + cand_penalty * 10.0

        delta = cand_obj - current_obj

        # Accept?
        if delta < 0 or rng.random() < math.exp(-delta / max(T, T_min)):
            current_inner = candidate
            current_ang = cand_ang
            current_s = cand_s
            current_penalty = cand_penalty
            current_obj = cand_obj
            accept_count += 1

            if current_penalty < 0.001 and current_s < best_s:
                best_s = current_s
                best_inner = current_inner.copy()
                no_improve_count = 0
            elif current_obj < best_obj:
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # Adaptive temperature and step size
        if n_iter % 200 == 0:
            accept_rate = accept_count / 200.0
            if accept_rate > 0.5:
                T *= 0.92
                step_size *= 0.95
            elif accept_rate < 0.1:
                T = min(T * 1.3, 0.5)
                step_size = min(step_size * 1.15, 0.8)
            accept_count = 0

        # Periodically re-optimize outer angle
        if n_iter % 1000 == 0:
            for ang in np.arange(max(0, current_ang - 3), min(60, current_ang + 3.1), 0.5):
                s_val = _compute_outer_s(current_inner, 0, 0, ang)
                if s_val < current_s:
                    current_s = s_val
                    current_ang = ang

        # Restart if stuck for too long
        if no_improve_count > 2000 and T > T_min * 2:
            T = max(T * 0.5, T_min * 5)
            step_size = max(step_size * 0.7, 0.05)
            no_improve_count = 0

    # Final: re-optimize outer angle finely
    for ang in np.arange(max(0, best_ang - 2), min(60, best_ang + 2.1), 0.1):
        s_val = _compute_outer_s(best_inner, 0, 0, ang)
        if s_val < best_s:
            best_s = s_val
            best_ang = ang

    return best_inner, np.array([0.0, 0.0, best_ang]), best_s


def _refine_with_local_search(inner, outer, s, deadline, rng):
    """
    Local refinement: try small perturbations of individual hexagons
    to see if we can reduce s further.
    """
    best_inner = inner.copy()
    best_outer = outer.copy()
    best_s = s
    best_ang = outer[2]

    for _ in range(200):
        if time.time() > deadline - 5:
            break

        improved = False
        for idx in range(12):
            for dx in [-0.1, -0.05, 0, 0.05, 0.1]:
                for dy in [-0.1, -0.05, 0, 0.05, 0.1]:
                    if dx == 0 and dy == 0:
                        continue
                    candidate = best_inner.copy()
                    candidate[idx, 0] += dx
                    candidate[idx, 1] += dy

                    cand_s = _compute_outer_s(candidate, 0, 0, best_ang)
                    cand_penalty = _overlap_penalty(candidate)
                    if cand_penalty < 0.001 and cand_s < best_s - 1e-6:
                        best_inner = candidate
                        best_s = cand_s
                        improved = True

            # Try angle perturbations
            for dangle in [-5, -2, -1, 1, 2, 5]:
                candidate = best_inner.copy()
                candidate[idx, 2] = (best_inner[idx, 2] + dangle) % 60.0

                cand_s = _compute_outer_s(candidate, 0, 0, best_ang)
                cand_penalty = _overlap_penalty(candidate)
                if cand_penalty < 0.001 and cand_s < best_s - 1e-6:
                    best_inner = candidate
                    best_s = cand_s
                    improved = True

        # Try outer angle refinement
        for ang in np.arange(max(0, best_ang - 1), min(60, best_ang + 1.1), 0.25):
            s_val = _compute_outer_s(best_inner, 0, 0, ang)
            if s_val < best_s - 1e-6:
                best_s = s_val
                best_ang = ang
                improved = True

        if not improved:
            break

    best_outer = np.array([0.0, 0.0, best_ang])
    return best_inner, best_outer, best_s


def _multi_start_sa(seed, deadline, rng, n_starts=3):
    """
    Run SA multiple times from slightly perturbed versions of the seed.
    """
    best_inner = None
    best_outer = None
    best_s = np.inf

    for start in range(n_starts):
        if time.time() > deadline - 30:
            break

        # Create perturbed seed
        perturbed = seed.copy()
        for i in range(12):
            perturbed[i, 0] += rng.normal(0, 0.3)
            perturbed[i, 1] += rng.normal(0, 0.3)
            perturbed[i, 2] = (perturbed[i, 2] + rng.normal(0, 5)) % 60.0

        # Allocate time for this start
        remaining_starts = n_starts - start
        per_start_time = (deadline - 30 - time.time()) / max(remaining_starts, 1)
        sa_deadline = time.time() + min(per_start_time, deadline - 30)

        inner, outer, s = _sa_optimize(perturbed, sa_deadline, rng)

        if s < best_s:
            best_s = s
            best_inner = inner.copy()
            best_outer = outer.copy()

    return best_inner, best_outer, best_s


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing the outer hexagon side length.
    Returns (inner, outer, s).
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s kill

    rng = np.random.default_rng(42)

    # Multiple seed configurations
    seeds = [
        _seed_config_hex_cluster(),
        _seed_config_spread(),
        _seed_config_rotated(),
        _seed_config_diagonal(),
        _seed_config_ring(),
    ]

    best_inner = None
    best_outer = None
    best_s = np.inf

    # Phase 1: SA from each seed with multi-start
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 60:
            break

        # Allocate remaining time among seeds
        remaining_seeds = len(seeds) - i
        per_seed_time = (deadline - 60 - time.time()) / max(remaining_seeds, 1)

        # For early seeds, use multi-start; for later ones, single SA
        n_starts = 3 if i < 2 else 2 if i < 4 else 1
        inner, outer, s = _multi_start_sa(seed, time.time() + per_seed_time, rng, n_starts)

        if s < best_s:
            best_s = s
            best_inner = inner.copy()
            best_outer = outer.copy()

    # Phase 2: Fine-tune best result with more SA iterations
    if best_inner is not None and time.time() < deadline - 30:
        inner, outer, s = _sa_optimize(best_inner, deadline - 30, rng)
        if s < best_s:
            best_s = s
            best_inner = inner.copy()
            best_outer = outer.copy()

    # Phase 3: Local refinement
    if best_inner is not None and time.time() < deadline - 10:
        inner, outer, s = _refine_with_local_search(best_inner, best_outer, best_s, deadline - 10, rng)
        if s < best_s:
            best_s = s
            best_inner = inner.copy()
            best_outer = outer.copy()

    # Phase 4: Final outer angle optimization
    if best_inner is not None and time.time() < deadline - 5:
        for ang in np.arange(0, 60, 0.25):
            s_val = _compute_outer_s(best_inner, 0, 0, ang)
            if s_val < best_s:
                best_s = s_val
                best_outer = np.array([0.0, 0.0, ang])

        # Fine refine
        if best_outer is not None:
            for ang in np.arange(max(0, best_outer[2] - 0.25), min(60, best_outer[2] + 0.26), 0.025):
                s_val = _compute_outer_s(best_inner, 0, 0, ang)
                if s_val < best_s:
                    best_s = s_val
                    best_outer = np.array([0.0, 0.0, ang])

    # Final verification
    if best_inner is None:
        inner = _seed_config_hex_cluster()
        best_s = _compute_outer_s(inner, 0, 0, 0.0)
        best_inner = inner
        best_outer = np.array([0.0, 0.0, 0.0])

    valid, _ = _check_validity(best_inner, best_outer, best_s)
    if not valid:
        # Fallback to a known valid configuration
        inner = np.array([
            [0, 0, 0],
            [2.5, 0, 0],
            [-2.5, 0, 0],
            [1.25, 2.17, 0],
            [-1.25, 2.17, 0],
            [1.25, -2.17, 0],
            [-1.25, -2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, -2.17, 0],
            [-3.75, -2.17, 0],
            [0, -4, 0],
        ])
        outer = np.array([0, 0, 0])
        s = 8.0
        return inner, outer, s

    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
