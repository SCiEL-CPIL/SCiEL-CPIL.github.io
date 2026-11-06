"""
Force-Directed Repulsion Packing for 50 circles in a unit square.

Strategy:
1. Use Coulomb-like repulsion forces between all circles + wall-penalty forces
   to drive the system toward a dense packing. This is a 100-DOF optimization
   via gradient-based dynamics, fundamentally different from lattice parametrization.
2. Run multiple restarts from diverse initial configurations (hex grid, random,
   grid with jitter) with adaptive cooling.
3. After the force-based relaxation, perform a fast local refinement where each
   point only checks its k nearest neighbors (not all N² pairs), enabling many
   more iterations.
"""

import time
import numpy as np
from scipy.spatial import cKDTree

N = 50
DEADLINE_FRACTION = 0.85
K_NEIGHBORS = 12  # neighbors to check in local refinement


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0

    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)

    if min_boundary <= 0:
        return 0.0

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0

    return min(min_boundary, min_pair)


def _fast_radius(centers):
    """Fast radius computation without validity checks."""
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_boundary, min_pair)


def _neighbor_radius(centers, tree=None):
    """Radius computation using only k nearest neighbors for each point."""
    if tree is None:
        tree = cKDTree(centers)
    # Query k+1 nearest (including self)
    dists, _ = tree.query(centers, k=K_NEIGHBORS + 1)
    # dists[:, 0] is 0 (self), so take min of dists[:, 1:]
    min_pair = np.min(dists[:, 1:]) / 2.0
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    return min(min_boundary, min_pair)


def _init_hex_grid(n_points=N):
    """Initialize points in a hexagonal grid pattern."""
    points = []
    # Generate hex grid points
    max_r = 10
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            x = (i + j / 2.0) * np.sqrt(3)
            y = j * 1.5
            points.append([x, y])
    points = np.array(points)
    dists_from_origin = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    sorted_indices = np.argsort(dists_from_origin)
    selected = points[sorted_indices[:n_points]].copy()
    # Scale to fit in square
    max_coord = np.max(np.abs(selected))
    if max_coord > 0.4:
        selected *= 0.4 / max_coord
    return selected


def _init_random(n_points=N, seed=0):
    """Random initialization within the square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.4, 0.4, size=(n_points, 2))


def _init_grid_jitter(n_points=N, seed=0):
    """Grid initialization with random jitter."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(np.sqrt(n_points)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:n_points].astype(float)
    grid += rng.normal(0, 0.03, size=grid.shape)
    grid = np.clip(grid, -0.45, 0.45)
    return grid


def _force_relaxation(centers, n_steps=5000, deadline=None, verbose=False):
    """
    Force-directed relaxation:
    - Repulsive force between all pairs: F = k_rep / d^2 (Coulomb-like)
    - Wall penalty: push points back if too close to boundary
    - Adaptive step size with cooling
    """
    centers = centers.copy()
    k_rep = 0.01
    k_wall = 0.5
    step_size = 0.005
    min_step = 1e-6
    cooling = 0.9995
    wall_margin = 0.01

    best_r = _fast_radius(centers)
    best_centers = centers.copy()
    no_improve = 0

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        forces = np.zeros_like(centers)

        # Pairwise repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0)  # avoid div by zero

        # Force magnitude: k_rep / d^2, direction: away from neighbor
        force_mag = k_rep / (dists**2)
        np.fill_diagonal(force_mag, 0.0)

        # Sum forces (vector direction * magnitude)
        # Normalize diff to get direction
        dists_safe = np.where(dists > 1e-10, dists, 1.0)
        dir_vec = diff / dists_safe[:, :, np.newaxis]
        forces += np.sum(dir_vec * force_mag[:, :, np.newaxis], axis=1)

        # Wall penalty: push inward if close to boundary
        for i in range(N):
            for k in range(2):
                val = centers[i, k]
                if val > 0.5 - wall_margin:
                    forces[i, k] -= k_wall * (val - (0.5 - wall_margin))
                elif val < -0.5 + wall_margin:
                    forces[i, k] += k_wall * ((-0.5 + wall_margin) - val)

        # Apply forces with step size
        centers += step_size * forces

        # Clamp to square (hard constraint)
        centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)

        # Check improvement
        current_r = _neighbor_radius(centers)
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 200:
                # Cool down and reset to best
                step_size *= cooling
                if step_size < min_step:
                    step_size = min_step
                centers = best_centers.copy()
                no_improve = 0

        # Gradually cool
        step_size = max(min_step, step_size * cooling)

    return best_centers, best_r


def _local_refinement(centers, deadline=None, max_iters=20000):
    """
    Fast local refinement using nearest-neighbor checks.
    Each point is nudged to maximize its minimum distance to k nearest neighbors
    and boundary. Uses cKDTree for O(N log N) distance queries.
    """
    centers = centers.copy()
    best_r = _fast_radius(centers)
    best_centers = centers.copy()

    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    step_idx = 0
    no_improve = 0
    max_no_improve = 500
    rng = np.random.RandomState(123)

    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Pick a random point
        idx = rng.randint(N)
        orig = centers[idx].copy()
        improved = False

        for attempt in range(4):
            s = step_sizes[min(step_idx, len(step_sizes) - 1)] * (0.5 ** attempt)
            # Try 8 directions + random
            angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
            for a in angles:
                candidate = orig + s * np.array([np.cos(a), np.sin(a)])
                candidate = np.clip(candidate, -0.5 + 1e-9, 0.5 - 1e-9)
                centers[idx] = candidate

                # Quick check: only compute radius with neighbors
                tree = cKDTree(centers)
                r = _neighbor_radius(centers, tree)

                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
                    improved = True
                    break
                else:
                    centers[idx] = orig

            if improved:
                break

        if improved:
            no_improve = 0
            # Increase step slightly
            if step_idx < len(step_sizes) - 1:
                step_idx = min(step_idx + 1, len(step_sizes) - 1)
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                step_idx = min(step_idx + 1, len(step_sizes) - 1)
                no_improve = 0
                # Reset to best
                centers = best_centers.copy()

    return best_centers, best_r


def _multi_restart_local(centers, deadline):
    """Multiple short local refinement passes from slightly perturbed states."""
    best_centers = centers.copy()
    best_r = _fast_radius(best_centers)
    rng = np.random.RandomState(77)

    remaining_iters = 4
    for restart in range(remaining_iters):
        if time.time() > deadline:
            break

        # Perturb current best
        perturbed = best_centers + rng.normal(0, 0.005, size=(N, 2))
        perturbed = np.clip(perturbed, -0.5 + 1e-9, 0.5 - 1e-9)

        # Short local refinement
        c, r = _local_refinement(perturbed, deadline=deadline, max_iters=5000)
        if r > best_r:
            best_r = r
            best_centers = c

    return best_centers, best_r


def construct():
    start_time = time.time()
    total_time_budget = 150.0
    deadline = start_time + total_time_budget * DEADLINE_FRACTION

    best_centers = None
    best_radius = 0.0

    # Phase 1: Force-directed relaxation from multiple initializations
    initializations = [
        ("hex", _init_hex_grid()),
        ("grid_jitter", _init_grid_jitter(seed=1)),
        ("random1", _init_random(seed=42)),
        ("grid_jitter2", _init_grid_jitter(seed=99)),
        ("random2", _init_random(seed=200)),
    ]

    # Allocate time: ~60% for force relaxation, ~35% for local refinement, ~5% safety
    force_deadline = start_time + total_time_budget * 0.50
    refine_deadline = deadline

    for name, init_centers in initializations:
        if time.time() > force_deadline:
            break

        remaining = force_deadline - time.time()
        if remaining < 3.0:
            break

        # Run force relaxation
        centers, r = _force_relaxation(init_centers, n_steps=3000, deadline=force_deadline)

        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()

    # Phase 2: Local refinement on the best force-relaxed solution
    if best_centers is not None and time.time() < refine_deadline - 5.0:
        # First pass: main local refinement
        centers, r = _local_refinement(best_centers, deadline=refine_deadline, max_iters=30000)
        if r > best_radius:
            best_radius = r
            best_centers = centers

    # Phase 3: Multi-restart local refinement
    if time.time() < refine_deadline - 3.0:
        centers, r = _multi_restart_local(best_centers, deadline=refine_deadline)
        if r > best_radius:
            best_radius = r
            best_centers = centers

    # Fallback: if something went very wrong
    if best_centers is None or best_radius < 0.03:
        m = 8
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N].astype(float)
        best_centers = grid
        best_radius = _compute_radius(best_centers)

        if time.time() < refine_deadline - 2.0:
            centers, r = _local_refinement(best_centers, deadline=refine_deadline, max_iters=10000)
            if r > best_radius:
                best_radius = r
                best_centers = centers

    # Final validation
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0 or not np.all(np.isfinite(best_centers)):
        m = 8
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N].astype(float)
        best_centers = grid

    return best_centers
