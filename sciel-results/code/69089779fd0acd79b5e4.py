"""Packing 300 equal circles in a unit circle: multi-start L-BFGS-B with annealed soft-min."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the exact packing radius: min(min_pair/2, 1 - max_norm)."""
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms > 1.0 + 1e-9):
        return -1.0
    # Pairwise distances
    i_idx, j_idx = np.triu_indices(N, k=1)
    diffs = points[i_idx] - points[j_idx]
    pair_dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    min_pair = float(np.min(pair_dists))
    min_boundary = float(1.0 - np.max(norms))
    return min(min_pair / 2.0, min_boundary)


def _soft_min_value_and_grad(points, beta):
    """
    Compute the soft-min of all pairwise half-distances and boundary distances,
    plus its gradient. We maximize this soft-min, so we minimize its negative.
    """
    n = len(points)
    i_idx, j_idx = np.triu_indices(n, k=1)
    diffs = points[i_idx] - points[j_idx]
    pair_dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    norms = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - norms

    # All distances we care about
    all_dists = np.concatenate([pair_dists, boundary_dists])
    d_min = np.min(all_dists)
    shifted = all_dists - d_min
    # For numerical stability
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    if S < 1e-300:
        S = 1e-300
    log_S = np.log(S)

    # soft_min = d_min - (1/beta) * log(S)
    soft_min_val = d_min - (1.0 / beta) * log_S

    # Gradient: d(soft_min)/dx = -sum_k w_k * d(d_k)/dx  where w_k = exp(-beta*(d_k - d_min)) / S
    # We want to maximize soft_min, so gradient of -soft_min is sum_k w_k * d(d_k)/dx
    grad = np.zeros_like(points)

    # Pairwise contribution
    pair_weights = exp_terms[:len(pair_dists)] / S
    norms_pairs = pair_dists[:, np.newaxis]
    norms_pairs[norms_pairs < 1e-12] = 1e-12
    unit_vecs = diffs / norms_pairs

    # d(pair_dist)/d(point_i) = (p_i - p_j) / |p_i - p_j| = unit_vec
    # d(pair_dist)/d(point_j) = -unit_vec
    # We minimize -soft_min, so grad = sum w_k * d(d_k)/dx
    # For point i: contribution is +w * unit_vec
    # For point j: contribution is -w * unit_vec
    np.add.at(grad, i_idx, pair_weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, -pair_weights[:, np.newaxis] * unit_vecs)

    # Boundary contribution
    boundary_weights = exp_terms[len(pair_dists):] / S
    norms_safe = norms[:, np.newaxis]
    norms_safe[norms_safe < 1e-12] = 1e-12
    unit_pos = points / norms_safe
    # d(boundary_dist)/d(point_i) = -unit_pos
    # So contribution to grad of -soft_min is -w * unit_pos... wait
    # soft_min includes boundary_dists. d(boundary_dist)/dx = -x/|x|
    # grad of -soft_min = sum w_k * d(d_k)/dx
    # For boundary: d(d_k)/dx = -unit_pos, so contribution is -w * unit_pos
    grad -= boundary_weights[:, np.newaxis] * unit_pos

    return -soft_min_val, grad


def _hex_lattice_init(angle_deg=0.0):
    """Generate hexagonal lattice points within the unit disk, rotated by angle."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Estimate spacing for ~300 points in unit disk
    # Area per point ~ pi/sqrt(N) ~ 0.181, spacing ~ 0.2
    spacing = 0.2
    k = int(1.0 / spacing) + 3

    points_list = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            if np.hypot(xr, yr) <= 1.0 - spacing / 2.0:
                points_list.append((xr, yr))

    if len(points_list) < N:
        return None

    points = np.array(points_list)
    radius = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radius, kind="stable")[:N]
    return points[idx]


def _ring_init(n_rings):
    """Initialize with concentric rings."""
    points = []
    remaining = N
    # Center point
    points.append((0.0, 0.0))
    remaining -= 1

    ring_radii = [0.95 * (i + 1) / n_rings for i in range(n_rings)]
    circumferences = [2 * np.pi * r for r in ring_radii]
    total_circ = sum(circumferences)

    for i, r in enumerate(ring_radii):
        count = max(1, int(round(remaining * circumferences[i] / total_circ)))
        if i == n_rings - 1:
            count = remaining
        remaining -= count
        offset = (i * np.pi) / max(1, count)
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
        for a in angles:
            points.append((r * np.cos(a), r * np.sin(a)))

    if len(points) != N:
        # Adjust
        while len(points) > N:
            points.pop()
        while len(points) < N:
            points.append((0.95, 0.0))

    return np.array(points)


def _random_init(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = rng.uniform(0, 0.95, N)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def _optimize_from(points, deadline):
    """
    Optimize a starting configuration using multi-stage L-BFGS-B with annealed beta.
    Returns (optimized_points, score).
    """
    current_pts = points.copy()
    current_score = _compute_score(current_pts)

    # Multi-stage beta schedule
    betas = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]

    for beta in betas:
        if time.time() > deadline - 5:
            break

        def objective(params):
            if time.time() > deadline:
                return 1e10
            pts = params.reshape(-1, 2)
            if not np.all(np.isfinite(pts)):
                return 1e10
            norms = np.linalg.norm(pts, axis=1)
            if np.any(norms > 1.0):
                penalty = np.sum(np.maximum(norms - 1.0, 0.0) ** 2)
                return 1e6 + penalty
            val, _ = _soft_min_value_and_grad(pts, beta)
            return val

        def grad(params):
            if time.time() > deadline:
                return np.zeros_like(params)
            pts = params.reshape(-1, 2)
            if not np.all(np.isfinite(pts)):
                return np.zeros_like(params)
            norms = np.linalg.norm(pts, axis=1)
            if np.any(norms > 1.0):
                # Penalty gradient
                grad = np.zeros_like(pts)
                mask = norms > 1.0
                grad[mask] = 2.0 * (norms[mask] - 1.0)[:, np.newaxis] * pts[mask] / norms[mask, np.newaxis]
                return grad.flatten()
            _, g = _soft_min_value_and_grad(pts, beta)
            return g.flatten()

        try:
            res = minimize(
                objective,
                current_pts.flatten(),
                jac=grad,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )

            new_pts = res.x.reshape(-1, 2)
            # Project back inside unit circle
            norms = np.linalg.norm(new_pts, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                new_pts[mask] *= (1.0 / norms[mask])[:, np.newaxis]

            new_score = _compute_score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score

        except Exception:
            pass

    return current_pts, current_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_points = None
    best_score = -np.inf

    # Generate multiple diverse initial configurations
    configs = []

    # Hexagonal lattices with various rotations
    for angle in [0, 15, 30, 45, 60, 75]:
        pts = _hex_lattice_init(angle)
        if pts is not None:
            configs.append(pts)

    # Ring initializations
    for n_rings in [7, 10, 13, 16]:
        pts = _ring_init(n_rings)
        configs.append(pts)

    # Random initializations
    for seed in [42, 123, 456, 789]:
        pts = _random_init(seed)
        configs.append(pts)

    # Optimize each configuration
    for cfg in configs:
        if time.time() > deadline - 10:
            break

        optimized, score = _optimize_from(cfg, deadline)

        if score > best_score:
            best_score = score
            best_points = optimized.copy()

    # Final validation and cleanup
    if best_points is None:
        # Fallback: hexagonal lattice
        best_points = _hex_lattice_init(0.0)
        if best_points is None:
            best_points = _random_init(0)

    # Ensure all points are inside unit circle
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] *= (1.0 / norms[mask])[:, np.newaxis]

    return best_points

# EVOLVE-BLOCK-END
