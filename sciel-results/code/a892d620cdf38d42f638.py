"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _min_triangle_area_fast(points):
    """Fully vectorized: compute min area of any triangle formed by the n points.
    points: (n, 2) array. Returns scalar min area."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = points[j] - points[i]  # (2,)
            diff_k = points - points[i]  # (n, 2)
            cross = d[0] * diff_k[:, 1] - d[1] * diff_k[:, 0]  # (n,)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            areas = 0.5 * np.abs(cross[mask])
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _objective_cartesian(flat):
    """Objective for Nelder-Mead: minimize negative min triangle area.
    flat: (2*N,) array of cartesian coordinates.
    Returns: -min_area + penalty if any point is outside triangle."""
    points = flat.reshape(N, 2)

    x = points[:, 0]
    y = points[:, 1]

    penalty = 0.0
    eps = 1e-8
    penalty += np.sum(np.maximum(0, -y - eps)) * 10
    penalty += np.sum(np.maximum(0, y - np.sqrt(3) * x + eps)) * 10
    penalty += np.sum(np.maximum(0, y - np.sqrt(3) * (1 - x) + eps)) * 10

    min_area = _min_triangle_area_fast(points)
    return -min_area + penalty


def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]

    y = np.clip(y, 0, np.sqrt(3) / 2)

    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)

    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)

    points[:, 0] = x
    points[:, 1] = y
    return points


def _generate_grid_init(rng, grid_size=5, jitter_scale=0.05):
    """Generate a diverse initial configuration using structured grid + jitter."""
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            if i + j < grid_size:
                a = i / (grid_size - 1)
                b = j / (grid_size - 1)
                c = 1 - a - b
                if a >= 0 and b >= 0 and c >= 0:
                    x = b * 1.0 + c * 0.5
                    y = c * (np.sqrt(3) / 2)
                    positions.append([x, y])

    positions = np.array(positions)

    if len(positions) <= N:
        selected = positions.copy()
        while len(selected) < N:
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0, np.sqrt(3) / 2 * 0.8)
            if y <= np.sqrt(3) * x and y <= np.sqrt(3) * (1 - x):
                selected = np.vstack([selected, [[x, y]]])
        points = selected[:N]
    else:
        indices = rng.choice(len(positions), size=N, replace=False)
        points = positions[indices]

    points = points + rng.normal(0, jitter_scale, size=(N, 2))
    points = _clip_to_triangle(points)

    return points


def _structured_init(seed_idx):
    """Create diverse structured initial configurations based on symmetry."""
    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2  # height of triangle

    configs = []

    # Config 0: Symmetric hexagonal-like arrangement
    configs.append(np.array([
        [0.05, 0.05],
        [0.95, 0.05],
        [0.5, h - 0.05],
        [0.5, 0.05],
        [0.25, h / 2],
        [0.75, h / 2],
        [0.5, h / 3],
        [0.3, 0.3],
        [0.7, 0.3],
        [0.5, 0.45],
        [0.4, 0.35],
    ]))

    # Config 1: Three rows of 4-4-3
    configs.append(np.array([
        [0.1, 0.08], [0.35, 0.08], [0.65, 0.08], [0.9, 0.08],
        [0.2, 0.35], [0.4, 0.35], [0.6, 0.35], [0.8, 0.35],
        [0.3, 0.6], [0.5, 0.6], [0.7, 0.6],
    ]))

    # Config 2: Concentric rings
    configs.append(np.array([
        [0.5, 0.05],
        [0.2, 0.25], [0.8, 0.25],
        [0.5, 0.45],
        [0.3, 0.55], [0.7, 0.55],
        [0.1, 0.15], [0.9, 0.15],
        [0.4, 0.7], [0.6, 0.7],
        [0.5, 0.85],
    ]))

    # Config 3: Grid-based
    configs.append(np.array([
        [0.15, 0.1], [0.4, 0.1], [0.65, 0.1], [0.85, 0.1],
        [0.25, 0.35], [0.5, 0.35], [0.75, 0.35],
        [0.35, 0.6], [0.55, 0.6], [0.75, 0.6],
        [0.45, 0.8],
    ]))

    # Config 4: Random but structured
    rng = np.random.default_rng(seed_idx * 1000 + 42)
    pts = []
    # Place 3 near corners
    pts.append([0.08, 0.08])
    pts.append([0.92, 0.08])
    pts.append([0.5, h - 0.08])
    # Place 8 in interior with some structure
    for i in range(8):
        angle = i * 2 * np.pi / 8
        r = 0.2 + 0.1 * (i % 2)
        cx = 0.5 + r * np.cos(angle)
        cy = h / 3 + r * 0.8 * np.sin(angle)
        pts.append([cx, cy])
    configs.append(np.array(pts))

    return configs


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s

    rng = np.random.default_rng(42)

    best_points = None
    best_score = -float('inf')

    # Get structured initial configurations
    inits = _structured_init(0)

    # Add grid-based inits
    for jitter in [0.03, 0.05, 0.07]:
        inits.append(_generate_grid_init(rng, grid_size=4, jitter_scale=jitter))

    # Multi-start Nelder-Mead optimization
    for start_idx, points_init in enumerate(inits):
        if time.time() > deadline - 10:
            break

        # Flatten for optimization
        flat_init = points_init.flatten()

        # Run Nelder-Mead
        try:
            result = minimize(
                _objective_cartesian,
                flat_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )

            # Clip result to triangle
            points_opt = _clip_to_triangle(result.x.reshape(N, 2))

            # Evaluate
            score = _min_triangle_area_fast(points_opt)

            if score > best_score:
                best_score = score
                best_points = points_opt.copy()

        except Exception:
            pass

    # If no valid result, use a fallback
    if best_points is None:
        best_points = _clip_to_triangle(np.array([
            [0.05, 0.05], [0.95, 0.05], [0.5, np.sqrt(3) / 2 - 0.05],
            [0.5, 0.05], [0.25, np.sqrt(3) / 4], [0.75, np.sqrt(3) / 4],
            [0.5, np.sqrt(3) / 6], [0.3, 0.3], [0.7, 0.3],
            [0.5, 0.5], [0.4, 0.4]
        ]))
        best_score = _min_triangle_area_fast(best_points)

    # Final refinement: run one more Nelder-Mead from the best point
    if time.time() < deadline - 5:
        try:
            flat_best = best_points.flatten()
            result = minimize(
                _objective_cartesian,
                flat_best,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True
                }
            )
            points_final = _clip_to_triangle(result.x.reshape(N, 2))
            score_final = _min_triangle_area_fast(points_final)
            if score_final > best_score:
                best_points = points_final
                best_score = score_final
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
