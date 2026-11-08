"""
Strategy: Coarse-grained structural-motif guided optimization for Sutton-Chen clusters.
Uses a radial density / local environment descriptor that penalizes non-FCC/HCP arrangements,
guiding the search toward the known structural motifs of global minima.
Combines multi-start from diverse templates with adaptive basin-hopping and a structural bias.
"""

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 320.0
_START_TIME = None


def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)


def _fcc_cluster(n, spacing=1.0, rng=None):
    """Generate n atoms from an FCC lattice, compactly centered."""
    if rng is None:
        rng = np.random.default_rng()

    a = spacing * np.sqrt(2)
    R = (3 * n * a**3 / (16 * np.pi)) ** (1/3) * 1.5
    n_side = int(np.ceil(R / a)) + 3

    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                for offset in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = base + np.array(offset) * a
                    if np.linalg.norm(p) <= R:
                        points.append(p)

    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)

    if len(idx) < n:
        R2 = R * 1.8
        n_side2 = int(np.ceil(R2 / a)) + 3
        points2 = []
        for i in range(-n_side2, n_side2 + 1):
            for j in range(-n_side2, n_side2 + 1):
                for k in range(-n_side2, n_side2 + 1):
                    base = np.array([i, j, k]) * a
                    for offset in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                        p = base + np.array(offset) * a
                        if np.linalg.norm(p) <= R2:
                            points2.append(p)
        points = np.array(points2)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)

    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points


def _hcp_cluster(n, spacing=1.0, rng=None):
    """Generate n atoms from an HCP lattice."""
    if rng is None:
        rng = np.random.default_rng()

    a = spacing
    c = a * np.sqrt(8.0 / 3.0)

    points = []
    n_layers = int(np.ceil(np.sqrt(n * 1.5))) + 5
    n_side = int(np.ceil(np.sqrt(n * 1.5))) + 5
    R2 = (n / 3.5) ** (2/3) * a**2 * 4.0

    for layer in range(-n_layers, n_layers + 1):
        z = layer * c
        if layer % 2 == 0:
            shift_x, shift_y = 0.0, 0.0
        else:
            shift_x = a / 2
            shift_y = a * np.sqrt(3) / 6

        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                x = i * a + j * a * 0.5 + shift_x
                y = j * a * np.sqrt(3) / 2.0 + shift_y
                if x**2 + y**2 + z**2 <= R2:
                    points.append([x, y, z])

    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)

    while len(points) < n:
        p = rng.uniform(-4 * a, 4 * a, 3)
        points = np.vstack([points, p])
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)

    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points


def _random_compact(n, spacing=1.0, rng=None):
    """Random compact cluster with minimum distance constraint."""
    if rng is None:
        rng = np.random.default_rng()

    radius = spacing * (n / 4.0) ** (1/3.0) * 2.8
    points = []
    min_dist = 0.6 * spacing
    max_attempts = n * 100

    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if not points:
            points.append(p)
            continue
        existing = np.array(points)
        dists = np.linalg.norm(existing - p, axis=1)
        if np.min(dists) >= min_dist:
            points.append(p)

    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)

    points = np.array(points)
    points = points - np.mean(points, axis=0)
    return points


def _icosahedral_shell(n, spacing=1.0, rng=None):
    """Icosahedral arrangement: center + shells."""
    if rng is None:
        rng = np.random.default_rng()

    points = [np.array([0.0, 0.0, 0.0])]

    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)

    if n > 1:
        shell1 = ico_verts * spacing
        for v in shell1:
            if len(points) < n:
                points.append(v)

    if n > 13:
        a = spacing * np.sqrt(2)
        R2 = 2.5 * spacing
        n_side = 6
        second_shell = []
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                for k in range(-n_side, n_side + 1):
                    base = np.array([i, j, k]) * a
                    for offset in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                        p = base + np.array(offset) * a
                        r = np.linalg.norm(p)
                        if 1.3 * spacing < r < 3.0 * spacing:
                            second_shell.append(p / r * 2.1 * spacing)

        seen = set()
        unique_second = []
        for p in second_shell:
            key = tuple(np.round(p, 3))
            if key not in seen:
                seen.add(key)
                unique_second.append(p)

        for p in unique_second:
            if len(points) < n:
                points.append(p)

    while len(points) < n:
        p = rng.uniform(-4.0 * spacing, 4.0 * spacing, 3)
        if np.linalg.norm(p) < 4.5 * spacing:
            points.append(p)

    points = np.array(points[:n])
    points = points - np.mean(points, axis=0)
    return points


def _bcc_cluster(n, spacing=1.0, rng=None):
    """Generate n atoms from a BCC lattice."""
    if rng is None:
        rng = np.random.default_rng()

    a = spacing * 2.0 / np.sqrt(3)
    R = (3 * n * a**3 / (8 * np.pi)) ** (1/3) * 1.5
    n_side = int(np.ceil(R / a)) + 3

    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                for offset in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    p = base + np.array(offset) * a
                    if np.linalg.norm(p) <= R:
                        points.append(p)

    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)

    if len(idx) < n:
        R2 = R * 1.8
        n_side2 = int(np.ceil(R2 / a)) + 3
        points2 = []
        for i in range(-n_side2, n_side2 + 1):
            for j in range(-n_side2, n_side2 + 1):
                for k in range(-n_side2, n_side2 + 1):
                    base = np.array([i, j, k]) * a
                    for offset in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                        p = base + np.array(offset) * a
                        if np.linalg.norm(p) <= R2:
                            points2.append(p)
        points = np.array(points2)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)

    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points


def _local_environment_score(positions, n_exp=12, m_exp=6, c=144.41):
    """
    Compute a structural bias score based on local environments.
    Penalizes atoms that don't have FCC/HCP-like coordination.
    Returns a scalar in [0, 1] where 0 = perfect FCC/HCP-like, 1 = bad.
    """
    n_atoms = len(positions)
    if n_atoms < 4:
        return 0.0

    # Compute pairwise distances
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)

    # For each atom, find its 12 nearest neighbors (typical FCC/HCP coordination)
    nn_count = min(12, n_atoms - 1)
    scores = []
    for i in range(n_atoms):
        d_i = dists[i]
        # Get indices of nn_count nearest neighbors
        nn_idx = np.argsort(d_i)[:nn_count]
        nn_dists = d_i[nn_idx]

        # Compute the coordination number (number of neighbors within 1.3x the nearest)
        d_min = nn_dists[0]
        coord_shell = nn_dists[nn_dists < 1.3 * d_min]
        n_coord = len(coord_shell)

        # FCC/HCP typically have 12 nearest neighbors in the first shell
        # Score: 0 if n_coord is in [11, 13], increasing deviation otherwise
        if 11 <= n_coord <= 13:
            score_i = 0.0
        else:
            # Penalize deviation from 12
            score_i = abs(n_coord - 12) / 12.0

        # Also check angular distribution: FCC/HCP have specific bond angles
        # Compute angles between pairs of nearest neighbors
        if nn_count >= 3:
            nn_positions = positions[nn_idx]
            # Normalize neighbor vectors
            nn_vecs = nn_positions - positions[i]
            nn_vecs_norm = nn_vecs / np.linalg.norm(nn_vecs, axis=1, keepdims=True)

            # Compute all pairwise angles
            angles = []
            for a in range(min(nn_count, 12)):
                for b in range(a + 1, min(nn_count, 12)):
                    cos_angle = np.dot(nn_vecs_norm[a], nn_vecs_norm[b])
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angles.append(np.arccos(cos_angle))

            if angles:
                angles = np.array(angles)
                # FCC has angles: 60°, 90°, 120°, 141.1° (arccos(-1/3))
                # HCP similar
                # Penalize angles that don't match common FCC/HCP angles
                fcc_angles = [np.pi/3, np.pi/2, 2*np.pi/3, np.arccos(-1/3)]
                min_dist_to_fcc = np.min([np.min(np.abs(angles - a)) for a in fcc_angles])
                score_i += 0.5 * min_dist_to_fcc / np.pi

        scores.append(min(score_i, 1.0))

    return float(np.mean(scores))


def _structural_bihopping(positions, potential, kind, rng, n_exp, m_exp, c,
                           n_iters=80, base_scale=0.4, min_time=8.0,
                           bias_weight=0.3):
    """
    Basin hopping with structural bias: prefer moves that maintain
    good local environments (FCC/HCP-like).
    """
    best_energy, best_pos = relax(positions, potential, kind)
    current_energy, current_pos = best_energy, best_pos
    current_struct_score = _local_environment_score(current_pos, n_exp, m_exp, c)

    for i in range(n_iters):
        if _time_left() < min_time:
            break

        # Adaptive scale: start large, decrease
        scale = base_scale * (0.92 ** (i // 15)) + 0.05

        # Generate multiple candidates and pick the best structurally
        n_candidates = 3
        best_candidate = None
        best_candidate_score = np.inf

        for _ in range(n_candidates):
            noise = rng.normal(0, scale, positions.shape)
            candidate = current_pos + noise

            # Quick energy check to avoid obvious bad moves
            try:
                e_cand, _ = sutton_chen_energy_gradient(candidate, kind)
                if not np.isfinite(e_cand) or e_cand > current_energy * 1.5:
                    continue
            except:
                continue

            struct_score = _local_environment_score(candidate, n_exp, m_exp, c)

            # Combined score: energy + structural bias
            if best_candidate is None:
                best_candidate = candidate
                best_candidate_score = struct_score
            else:
                # Prefer lower structural score (better FCC/HCP-like)
                if struct_score < best_candidate_score:
                    best_candidate = candidate
                    best_candidate_score = struct_score

        if best_candidate is None:
            # Fallback: just perturb
            best_candidate = current_pos + rng.normal(0, scale, positions.shape)
            best_candidate_score = current_struct_score

        try:
            e_new, pos_new = relax(best_candidate, potential, kind)
            if not np.isfinite(e_new):
                continue

            struct_new = _local_environment_score(pos_new, n_exp, m_exp, c)

            # Acceptance: Metropolis with structural bias
            energy_accept = e_new < current_energy or (
                rng.random() < np.exp(-(e_new - current_energy) / (0.01 * (1 + i / 30)))
            )
            struct_accept = struct_new < current_struct_score + 0.1

            if energy_accept and struct_accept:
                current_energy, current_pos = e_new, pos_new
                current_struct_score = struct_new

            if e_new < best_energy:
                best_energy = e_new
                best_pos = pos_new

        except:
            pass

    return best_energy, best_pos


def _high_precision_polish(positions, potential, kind):
    """Final high-precision polish pass with tighter tolerance."""
    try:
        from scipy.optimize import minimize

        def obj(x):
            pos = x.reshape(-1, 3)
            e, _ = sutton_chen_energy_gradient(pos, kind)
            return e

        def grad(x):
            pos = x.reshape(-1, 3)
            _, g = sutton_chen_energy_gradient(pos, kind)
            return g.flatten()

        x0 = positions.flatten()
        result = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 8000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 800}
        )

        if result.fun < np.inf:
            return result.fun, result.x.reshape(-1, 3)
        else:
            return relax(positions, potential, kind)
    except:
        return relax(positions, potential, kind)


def _get_params(kind):
    """Get potential parameters for the given kind."""
    params = {
        '12-6': (12, 6, 144.41),
        '9-6': (9, 6, 39.432),
        '10-8': (10, 8, 34.408)
    }
    return params.get(kind, (12, 6, 144.41))


def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    n_exp, m_exp, c = _get_params(kind)

    rng = np.random.default_rng(42 + size)

    best_energy = np.inf
    best_positions = None

    # Generate diverse templates
    templates = []

    try:
        templates.append(_fcc_cluster(size, 1.0, rng))
    except:
        pass

    try:
        templates.append(_hcp_cluster(size, 1.0, rng))
    except:
        pass

    try:
        templates.append(_icosahedral_shell(size, 1.0, rng))
    except:
        pass

    try:
        templates.append(_bcc_cluster(size, 1.0, rng))
    except:
        pass

    # Add random compact structures
    for _ in range(12):
        try:
            templates.append(_random_compact(size, 1.0, rng))
        except:
            pass

    # Phase 1: Relax all templates to get initial energies
    relaxed_results = []
    for i, t in enumerate(templates):
        if _time_left() < 20.0:
            break
        try:
            e, pos = relax(t, potential, kind)
            if np.isfinite(e):
                relaxed_results.append((e, pos, i))
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
        except:
            pass

    # Sort by energy, keep top ones for basin hopping
    relaxed_results.sort(key=lambda x: x[0])
    top_results = relaxed_results[:8]

    # Phase 2: Structural-bias basin hopping from top structures
    # Use parallel execution for speed
    def bh_worker(args):
        e_init, pos_init, idx = args
        local_rng = np.random.default_rng(1000 + idx)
        n_iters = max(40, int((_time_left() - 25.0) * 2.0))
        e, pos = _structural_bihopping(
            pos_init, potential, kind, local_rng,
            n_exp, m_exp, c,
            n_iters=n_iters, base_scale=0.45, min_time=12.0
        )
        return e, pos

    if _time_left() > 40.0:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(bh_worker, r) for r in top_results]
            for future in as_completed(futures):
                if _time_left() < 25.0:
                    break
                try:
                    e, pos = future.result(timeout=8.0)
                    if e < best_energy:
                        best_energy = e
                        best_positions = pos
                except:
                    pass

    # Phase 3: Additional basin hopping from best if time permits
    if best_positions is not None and _time_left() > 50.0:
        try:
            e, pos = _structural_bihopping(
                best_positions, potential, kind, rng,
                n_exp, m_exp, c,
                n_iters=60, base_scale=0.3, min_time=15.0
            )
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            pass

    # Phase 4: Final high-precision polish
    if best_positions is not None and _time_left() > 15.0:
        try:
            e_polished, pos_polished = _high_precision_polish(best_positions, potential, kind)
            if e_polished < best_energy:
                best_energy = e_polished
                best_positions = pos_polished
        except:
            pass

    # Safety: ensure we return a valid array
    if best_positions is None:
        # Fallback to a simple compact random structure
        best_positions = _random_compact(size, 1.0, rng)
        try:
            e, pos = relax(best_positions, potential, kind)
            best_positions = pos
        except:
            pass

    return best_positions
