"""Optimize 50 equal circle centres in the unit disk via max-min subgradient + adaptive radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = None
DEADLINE = 135.0  # safety margin before 150s kill


def _score(points):
    """Compute packing radius: min(min pairwise dist/2, min (1-|c_i|))."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = np.linalg.norm(points, axis=1)
    if np.any(dists >= 1.0):
        return 0.0
    boundary_margin = np.min(1.0 - dists)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    return min(min_pair_dist / 2.0, boundary_margin)


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial configuration."""
    points = np.zeros((N, 2))
    # Hexagonal lattice with spacing s
    s = 0.24
    # Generate hex lattice points
    candidates = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * s + (j % 2) * s * 0.5
            y = j * s * np.sqrt(3) / 2
            r = np.sqrt(x * x + y * y)
            if r < 0.95:
                candidates.append((r, x, y))
    candidates.sort(key=lambda t: t[0])
    for idx, (_, x, y) in enumerate(candidates[:N]):
        points[idx] = [x, y]
    return points


def _ring_init(ring_sizes, ring_radii, angle_offsets=None):
    """Generate points from concentric rings."""
    points = np.zeros((N, 2))
    idx = 0
    if angle_offsets is None:
        angle_offsets = [0.0] * len(ring_sizes)
    for ring_idx, (size, radius) in enumerate(zip(ring_sizes, ring_radii)):
        offset = angle_offsets[ring_idx] if ring_idx < len(angle_offsets) else 0.0
        for j in range(size):
            angle = 2 * np.pi * j / size + offset
            points[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            idx += 1
    return points


def _random_init():
    """Generate random initial points inside the unit disk."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _subgradient_optimize(points, r_target, max_iters=5000, lr_init=0.01):
    """
    Custom subgradient method for max-min packing.
    Iteratively pushes points apart and inward, while adapting r.
    """
    points = points.copy()
    lr = lr_init
    best_score = _score(points)
    best_points = points.copy()
    current_r = best_score

    for it in range(max_iters):
        if time.time() - START_TIME > DEADLINE:
            break

        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)

        # Compute gradient of the max-min objective
        # We want to maximize min(d_ij/2, 1-|c_i|)
        # Subgradient: for each point, sum up repulsive forces from close pairs
        # and attractive force toward center if too close to boundary

        forces = np.zeros_like(points)

        # Pairwise repulsion: push apart if d_ij < 2*r_target
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < 2 * r_target + 0.05:
                    # Strong repulsion when close
                    strength = 1.0 / (d[i, j] + 1e-8)
                    forces[i] += strength * diff[i, j]
                    forces[j] -= strength * diff[i, j]

        # Boundary: push inward if too close to boundary
        dists = np.linalg.norm(points, axis=1)
        for i in range(N):
            if dists[i] > 1.0 - r_target - 0.05:
                # Push toward center
                if dists[i] > 1e-8:
                    forces[i] -= 0.5 * points[i] / dists[i]

        # Update
        points += lr * forces

        # Project inside unit disk
        dists = np.linalg.norm(points, axis=1)
        for i in range(N):
            if dists[i] > 0.999:
                points[i] *= 0.999 / dists[i]

        # Compute score
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

        # Adapt learning rate and target radius
        lr *= 0.9995
        if it % 100 == 0:
            r_target = max(best_score, current_r * 0.95)
            current_r = best_score

    return best_points, best_score


def _lbfgs_refine(points, max_iter=2000):
    """L-BFGS-B refinement using soft-min surrogate."""
    def objective(flat):
        pts = flat.reshape(N, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists >= 1.0):
            return 1e10
        boundary_margin = 1.0 - np.min(dists)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        min_pair_dist = np.min(dist_mat)
        # Soft min: use log-sum-exp approximation
        r = min(min_pair_dist / 2.0, boundary_margin)
        if r <= 0:
            return 1e10
        return -r

    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8,
                               'maxls': 50})
    pts = result.x.reshape(N, 2)
    score = _score(pts)
    return pts, score


def _force_directed(points, max_iters=3000, dt=0.003):
    """Force-directed optimization with strong repulsion."""
    points = points.copy()
    best_score = _score(points)
    best_points = points.copy()

    for it in range(max_iters):
        if time.time() - START_TIME > DEADLINE:
            break

        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)

        forces = np.zeros_like(points)
        current_r = _score(points)
        threshold = 2.5 * current_r if current_r > 0.01 else 0.4

        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < threshold:
                    f = diff[i, j] / (d[i, j] ** 3 + 1e-8)
                    forces[i] += f
                    forces[j] -= f

        dists = np.linalg.norm(points, axis=1)
        for i in range(N):
            if dists[i] > 0.95:
                forces[i] -= 0.5 * points[i] / dists[i]

        points += dt * forces

        dists = np.linalg.norm(points, axis=1)
        for i in range(N):
            if dists[i] > 0.99:
                points[i] *= 0.99 / dists[i]

        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

        dt *= 0.999

    return best_points, best_score


def _full_optimize(points, budget_seconds=30.0):
    """Full optimization pipeline: subgradient + force-directed + L-BFGS-B."""
    t0 = time.time()
    best_points = points.copy()
    best_score = _score(points)

    # Phase 1: Subgradient method
    if time.time() - t0 < budget_seconds * 0.3:
        r_target = max(best_score, 0.12)
        pts_sg, score_sg = _subgradient_optimize(points, r_target, max_iters=2000, lr_init=0.008)
        if score_sg > best_score:
            best_score = score_sg
            best_points = pts_sg.copy()

    # Phase 2: Force-directed
    if time.time() - t0 < budget_seconds * 0.6:
        pts_f, score_f = _force_directed(best_points, max_iters=2000, dt=0.003)
        if score_f > best_score:
            best_score = score_f
            best_points = pts_f.copy()

    # Phase 3: L-BFGS-B refinement
    if time.time() - t0 < budget_seconds * 0.9:
        pts_lb, score_lb = _lbfgs_refine(best_points, max_iter=2000)
        if score_lb > best_score:
            best_score = score_lb
            best_points = pts_lb.copy()

    # Phase 4: Final fine-tune
    if time.time() - t0 < budget_seconds * 0.98:
        pts_final, score_final = _lbfgs_refine(best_points, max_iter=500)
        if score_final > best_score:
            best_score = score_final
            best_points = pts_final.copy()

    return best_points, best_score


def construct():
    global START_TIME
    START_TIME = time.time()

    best_points = None
    best_score = 0.0

    # Multiple initial configurations
    inits = [
        _hex_lattice_init(),
        _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        _ring_init([1, 6, 12, 18, 13], [0.0, 0.22, 0.45, 0.68, 0.90]),
        _ring_init([1, 6, 12, 18, 13], [0.0, 0.28, 0.55, 0.80, 0.98]),
        _random_init(),
        _random_init(),
    ]

    for init_idx, points in enumerate(inits):
        if time.time() - START_TIME > DEADLINE - 25:
            break

        # Add slight perturbation for non-first inits
        if init_idx > 0:
            points = points + np.random.uniform(-0.005, 0.005, size=(N, 2))

        # Full optimization
        budget = min(20.0, DEADLINE - (time.time() - START_TIME) - 15)
        if budget < 5:
            break

        pts_opt, score_opt = _full_optimize(points, budget_seconds=budget)
        if score_opt > best_score:
            best_score = score_opt
            best_points = pts_opt.copy()

    # Final polish on best result
    if best_points is not None and time.time() - START_TIME < DEADLINE - 5:
        pts_final, score_final = _lbfgs_refine(best_points, max_iter=1000)
        if score_final > best_score:
            best_points = pts_final
            best_score = score_final

    if best_points is None:
        best_points = _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])

    # Ensure validity
    dists = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if dists[i] >= 1.0:
            best_points[i] *= 0.999 / max(dists[i], 1e-10)

    return best_points


# EVOLVE-BLOCK-START
# The main construct function is defined above
# EVOLVE-BLOCK-END
