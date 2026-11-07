"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosahedron_vertices():
    """Return normalized icosahedron vertices (12 points)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def _icosahedron_edge_midpoints():
    """Return 30 edge midpoint directions (normalized)."""
    verts = _icosahedron_vertices()
    dists = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists.append((d, i, j))
    dists.sort()
    edge_pairs = [(dists[k][1], dists[k][2]) for k in range(30)]
    edge_dirs = []
    for i, j in edge_pairs:
        mid = (verts[i] + verts[j]) / 2.0
        edge_dirs.append(mid / np.linalg.norm(mid))
    return np.array(edge_dirs)


def _icosahedron_face_centers():
    """Return 20 face center directions (normalized)."""
    verts = _icosahedron_vertices()
    faces = [
        [0, 1, 4], [0, 4, 8], [0, 8, 10], [0, 10, 6], [0, 6, 1],
        [1, 6, 5], [1, 5, 7], [1, 7, 2], [1, 2, 0],
        [2, 7, 5], [2, 5, 9], [2, 9, 3], [2, 3, 1],
        [3, 9, 4], [3, 4, 8], [3, 8, 11], [3, 11, 7], [3, 7, 2],
        [4, 9, 11], [4, 11, 8],
    ]
    centers = []
    for f in faces:
        c = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        centers.append(c / np.linalg.norm(c))
    return np.array(centers)


def _build_icosahedral(n, radii_scale=1.0, include_center=True):
    """Build icosahedral cluster with center atom + shells."""
    positions = []
    core_radius = 0.72 * radii_scale
    core = _icosahedron_vertices() * core_radius
    positions.extend(core)
    n_remaining = n - 12

    if n_remaining <= 0:
        return np.array(positions[:n])

    if n >= 13 and include_center:
        positions.append([0.0, 0.0, 0.0])
        n_remaining -= 1

    if n_remaining <= 0:
        return np.array(positions[:n])

    edge_dirs = _icosahedron_edge_midpoints()
    shell2_radius = 1.05 * radii_scale
    n_add2 = min(n_remaining, 30)
    positions.extend(edge_dirs[:n_add2] * shell2_radius)
    n_remaining -= n_add2

    if n_remaining <= 0:
        return np.array(positions[:n])

    face_dirs = _icosahedron_face_centers()
    shell3_radius = 1.35 * radii_scale
    n_add3 = min(n_remaining, 20)
    positions.extend(face_dirs[:n_add3] * shell3_radius)
    n_remaining -= n_add3

    if n_remaining <= 0:
        return np.array(positions[:n])

    indices = np.arange(n_remaining)
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_remaining)
    theta = np.pi * (1 + np.sqrt(5)) * indices
    shell4_radius = 1.55 * radii_scale
    shell4 = np.column_stack([
        shell4_radius * np.sin(phi) * np.cos(theta),
        shell4_radius * np.sin(phi) * np.sin(theta),
        shell4_radius * np.cos(phi)
    ])
    positions.extend(shell4)
    return np.array(positions[:n])


def _build_growth(n, potential, kind, seed=0, deadline=None):
    """
    Hierarchical growth: start from small solved structures (n=13 icosahedron)
    and add atoms one at a time at optimal positions on the surface.
    """
    if deadline is not None and time.time() > deadline:
        return None

    rng = np.random.default_rng(seed)

    # Start with icosahedron (13 atoms: 12 shell + 1 center)
    if n <= 13:
        pos = _build_icosahedral(n)
        return pos

    # Start with 13-atom icosahedron
    pos = _build_icosahedral(13)
    current_n = 13

    while current_n < n:
        if deadline is not None and time.time() > deadline:
            break

        # Find optimal position for new atom by trying many candidate positions
        best_e = np.inf
        best_p = None

        # Candidate positions: on a sphere of various radii
        for radius in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
            # Try 200 random directions on the sphere
            for _ in range(200):
                # Random point on sphere
                u, v = rng.random(2)
                theta = 2 * np.pi * u
                phi = np.arccos(2 * v - 1)
                new_atom = np.array([
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.sin(phi) * np.sin(theta),
                    radius * np.cos(phi)
                ])

                # Check minimum distance constraint
                dists = np.linalg.norm(pos - new_atom, axis=1)
                if np.min(dists) < 0.35:
                    continue

                trial_pos = np.vstack([pos, new_atom])

                # Quick energy evaluation (no relaxation for speed)
                try:
                    e, _ = sutton_chen_energy_gradient(trial_pos, kind)
                    if e < best_e:
                        best_e = e
                        best_p = new_atom
                except Exception:
                    continue

        if best_p is None:
            # Fallback: add at a random position
            best_p = rng.uniform(-1.5, 1.5, 3)
            dists = np.linalg.norm(pos - best_p, axis=1)
            if np.min(dists) < 0.35:
                best_p = pos[0] + np.array([0.5, 0.5, 0.5])

        pos = np.vstack([pos, best_p])
        current_n += 1

        # Periodically relax to escape local minima during growth
        if current_n % 5 == 0 and deadline is not None and time.time() < deadline - 10:
            try:
                e, pos_relaxed = relax(pos, potential, kind)
                if _validate(pos_relaxed):
                    pos = pos_relaxed
            except Exception:
                pass

    return pos[:n]


def _build_fcc(n):
    """Build FCC lattice fragment."""
    a = 0.72
    basis = [
        [0, 0, 0],
        [0, a / 2, a / 2],
        [a / 2, 0, a / 2],
        [a / 2, a / 2, 0]
    ]
    side = int(np.ceil((n / 4) ** (1 / 3))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i * a + b[0], j * a + b[1], k * a + b[2]])
                    candidates.append(p)
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)[:n]
    positions = candidates[indices]
    positions -= np.mean(positions, axis=0)
    return positions


def _build_hcp(n):
    """Build HCP lattice fragment."""
    a = 0.72
    c = a * np.sqrt(8.0 / 3.0)
    layer_spacing = c / 2.0
    positions = []
    n_layers = int(np.ceil(np.sqrt(n / 3.0))) + 2
    for layer in range(n_layers):
        z = layer * layer_spacing
        offset = np.array([0, 0, 0]) if layer % 2 == 0 else np.array([a / 2, a / (2 * np.sqrt(3)), 0])
        max_r = int(np.ceil(np.sqrt(n / (3 * np.pi / (a ** 2 * 0.866)))))
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                x = a * (i + j / 2) + offset[0]
                y = a * np.sqrt(3) / 2 * j + offset[1]
                p = np.array([x, y, z])
                if len(positions) < n:
                    positions.append(p)
                else:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    positions = np.array(positions[:n])
    positions -= np.mean(positions, axis=0)
    return positions


def _build_random(n, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = 0.72 * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 5000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _validate(positions, min_dist=0.35):
    """Check if positions are valid (finite, no overlaps)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    for i in range(n):
        dists = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(dists < min_dist):
            return False
    return True


def _constrained_optimize(positions, potential, kind, deadline=None, barrier_coeff=100.0, min_dist=0.35):
    """
    Direct L-BFGS-B optimization with a barrier term to prevent overlaps.
    The objective is: E_sc + barrier_coeff * sum_{i<j} max(0, min_dist - r_ij)^2
    """
    n = len(positions)

    def objective_and_grad(x):
        pos = x.reshape(n, 3)
        # Compute SC energy and gradient
        e_sc, g_sc = sutton_chen_energy_gradient(pos, kind)

        # Compute barrier term and its gradient
        barrier = 0.0
        g_barrier = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                r = np.linalg.norm(d)
                if r < min_dist and r > 1e-10:
                    excess = min_dist - r
                    barrier += excess ** 2
                    # d(barrier)/d(pos[i]) = 2 * excess * d(excess)/d(pos[i])
                    # d(excess)/d(pos[i]) = -d(r)/d(pos[i]) = -d/r
                    grad_r = d / r
                    g_barrier[i] += 2 * excess * (-grad_r)
                    g_barrier[j] += 2 * excess * (grad_r)

        total_energy = e_sc + barrier_coeff * barrier
        total_grad = g_sc + barrier_coeff * g_barrier
        return total_energy, total_grad

    x0 = positions.flatten()
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if deadline is not None and time.time() > deadline:
            return None, None
        pos_opt = result.x.reshape(n, 3)
        e_final, g_final = sutton_chen_energy_gradient(pos_opt, kind)
        return e_final, pos_opt
    except Exception:
        return None, None


def _basin_hopping(positions, potential, kind, deadline, n_iterations=10, perturbation=0.3, rng=None):
    """
    Basin hopping: relax, perturb, relax, keep best.
    Uses the provided relax function which runs L-BFGS-B.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    try:
        energy, pos = relax(positions, potential, kind)
    except Exception:
        return None, None

    if not _validate(pos):
        return None, None

    for i in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break

        # Perturb
        perturbed = pos + rng.uniform(-perturbation, perturbation, pos.shape)

        if not _validate(perturbed):
            continue

        try:
            e2, pos2 = relax(perturbed, potential, kind)
        except Exception:
            continue

        if _validate(pos2) and e2 < energy:
            energy = e2
            pos = pos2

    return energy, pos


def construct(instance):
    """Structure prediction with basin hopping and constrained optimization for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind")

    start_time = time.time()
    deadline = start_time + 290  # Leave 40s margin for safety (330s total limit)

    best_energy = np.inf
    best_positions = None

    # Strategy 1: Hierarchical growth from small solved structures
    for seed in range(3):
        if time.time() > deadline:
            break
        positions = _build_growth(size, potential, kind, seed=seed, deadline=deadline)
        if positions is not None and _validate(positions):
            e, pos = _basin_hopping(positions, potential, kind, deadline, n_iterations=5, perturbation=0.25)
            if e is not None and _validate(pos) and e < best_energy:
                best_energy = e
                best_positions = pos

    # Strategy 2: Icosahedral core with center atom + basin hopping
    for rs in [0.9, 1.0, 1.1]:
        if time.time() > deadline:
            break
        positions = _build_icosahedral(size, radii_scale=rs, include_center=True)
        if not _validate(positions):
            continue
        e, pos = _basin_hopping(positions, potential, kind, deadline, n_iterations=5, perturbation=0.25)
        if e is not None and _validate(pos) and e < best_energy:
            best_energy = e
            best_positions = pos

    # Strategy 3: FCC fragment + basin hopping
    if time.time() < deadline:
        positions = _build_fcc(size)
        if _validate(positions):
            e, pos = _basin_hopping(positions, potential, kind, deadline, n_iterations=5, perturbation=0.25)
            if e is not None and _validate(pos) and e < best_energy:
                best_energy = e
                best_positions = pos

    # Strategy 4: HCP fragment + basin hopping
    if time.time() < deadline:
        positions = _build_hcp(size)
        if _validate(positions):
            e, pos = _basin_hopping(positions, potential, kind, deadline, n_iterations=5, perturbation=0.25)
            if e is not None and _validate(pos) and e < best_energy:
                best_energy = e
                best_positions = pos

    # Strategy 5: Random + basin hopping
    for seed in range(3):
        if time.time() > deadline:
            break
        positions = _build_random(size, seed=seed)
        if not _validate(positions):
            continue
        e, pos = _basin_hopping(positions, potential, kind, deadline, n_iterations=3, perturbation=0.2)
        if e is not None and _validate(pos) and e < best_energy:
            best_energy = e
            best_positions = pos

    # If no valid result, fall back to simple relaxation
    if best_positions is None:
        positions = _build_icosahedral(size, include_center=True)
        if not _validate(positions):
            positions = _build_random(size, seed=42)
        try:
            energy, best_positions = relax(positions, potential, kind)
        except Exception:
            best_positions = positions

    # Final high-precision polish: run constrained optimization with tighter tolerances
    if best_positions is not None and time.time() < deadline - 5:
        try:
            e_polish, pos_polish = _constrained_optimize(
                best_positions, potential, kind, deadline,
                barrier_coeff=100.0, min_dist=0.35
            )
            if e_polish is not None and _validate(pos_polish) and e_polish < best_energy:
                best_energy = e_polish
                best_positions = pos_polish
        except Exception:
            pass

    return best_positions
# EVOLVE-BLOCK-END
