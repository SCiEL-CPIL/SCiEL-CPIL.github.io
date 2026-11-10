"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    def compute_score(points):
        n = len(points)
        if n < 2:
            return 0.0
        dist = cdist(points, points)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        radii = np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def gen_hex_lattice(spacing):
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.5:
                    pts.append((x, y))
        return np.array(pts)

    def gen_ring_lattice():
        """Generate points on concentric rings (hexagonal packing pattern)."""
        pts = []
        # Center point
        pts.append((0.0, 0.0))
        
        # Concentric rings
        ring_spacing = 2.0 * 0.053  # approximate
        for ring in range(1, 20):
            r = ring * ring_spacing * 0.5
            if r > 0.95:
                break
            # Number of points on this ring
            n_pts = max(6, int(2 * np.pi * r / (ring_spacing * 0.5)))
            for k in range(n_pts):
                angle = 2 * np.pi * k / n_pts
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                if np.hypot(x, y) <= 0.95:
                    pts.append((x, y))
        
        return np.array(pts[:N])

    # Generate initial hex lattice
    initial_spacing = 2.0 * np.sqrt(0.907 / N)
    spacing = initial_spacing
    while True:
        pts = gen_hex_lattice(spacing)
        if len(pts) >= N:
            break
        spacing *= 0.98

    radii = np.hypot(pts[:, 0], pts[:, 1])
    points = pts[np.argsort(radii, kind="stable")[:N]].copy()

    best_score = compute_score(points)
    best_points = points.copy()

    # Force-directed optimization with adaptive step size
    step_size = spacing * 0.1
    lr_decay = 0.9995
    max_iter = 5000

    for iteration in range(max_iter):
        if time.time() > deadline:
            break

        n = len(points)
        dist = cdist(points, points)
        np.fill_diagonal(dist, np.inf)

        # Only consider close pairs for efficiency
        close_mask = dist < 3 * spacing
        inv_dist = np.zeros_like(dist)
        inv_dist[close_mask] = 1.0 / (dist[close_mask] + 1e-10)

        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        forces = np.sum(diff * inv_dist[:, :, np.newaxis], axis=1)

        # Boundary force: push points inward if they're too close to the boundary
        radii = np.linalg.norm(points, axis=1)
        boundary_force = np.zeros_like(points)
        boundary_margin = 0.02
        over = radii > (1.0 - boundary_margin)
        if np.any(over):
            mag = (radii[over] - (1.0 - boundary_margin)) * 5.0
            boundary_force[over] = -mag[:, np.newaxis] * points[over] / radii[over, np.newaxis]

        forces += boundary_force

        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            forces /= max_force

        points += step_size * forces

        # Project inside the unit circle
        radii = np.linalg.norm(points, axis=1)
        mask_out = radii > 1.0
        if np.any(mask_out):
            points[mask_out] /= radii[mask_out, np.newaxis]

        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()

        step_size *= lr_decay
        if step_size < 1e-8:
            break

    # Second phase: more aggressive local optimization
    if time.time() < deadline - 15:
        n = len(best_points)
        dist = cdist(best_points, best_points)
        np.fill_diagonal(dist, np.inf)

        # Find the k closest pairs
        k = 100
        flat_dist = dist.flatten()
        smallest_indices = np.argpartition(flat_dist, k)[:k]

        pairs = []
        for idx in smallest_indices:
            i, j = np.unravel_index(idx, (n, n))
            if i < j:
                pairs.append((i, j))

        # Also add boundary-violating points
        radii = np.linalg.norm(best_points, axis=1)
        boundary_idx = np.where(radii > 0.97)[0]

        # Gradient descent to push apart close pairs and pull in boundary violators
        lr = 2e-4
        for it in range(3000):
            if time.time() > deadline:
                break

            grad = np.zeros_like(best_points)

            for i, j in pairs:
                d = best_points[i] - best_points[j]
                dist_ij = np.linalg.norm(d)
                if dist_ij < 1e-10:
                    continue
                # Push apart
                grad[i] += d / dist_ij
                grad[j] -= d / dist_ij

            # Pull boundary violators inward
            for idx in boundary_idx:
                r = np.linalg.norm(best_points[idx])
                if r > 1e-10:
                    grad[idx] -= best_points[idx] / r

            best_points += lr * grad

            # Project inside
            radii = np.linalg.norm(best_points, axis=1)
            mask_out = radii > 1.0
            if np.any(mask_out):
                best_points[mask_out] /= radii[mask_out, np.newaxis]

            new_score = compute_score(best_points)
            if new_score > best_score:
                best_score = new_score

            # Decay learning rate
            lr *= 0.9995

    # Third phase: fine-tuning with smaller steps
    if time.time() < deadline - 5:
        n = len(best_points)
        dist = cdist(best_points, best_points)
        np.fill_diagonal(dist, np.inf)

        # Find the k closest pairs
        k = 200
        flat_dist = dist.flatten()
        smallest_indices = np.argpartition(flat_dist, k)[:k]

        pairs = []
        for idx in smallest_indices:
            i, j = np.unravel_index(idx, (n, n))
            if i < j:
                pairs.append((i, j))

        radii = np.linalg.norm(best_points, axis=1)
        boundary_idx = np.where(radii > 0.96)[0]

        lr = 5e-5
        for it in range(2000):
            if time.time() > deadline:
                break

            grad = np.zeros_like(best_points)

            for i, j in pairs:
                d = best_points[i] - best_points[j]
                dist_ij = np.linalg.norm(d)
                if dist_ij < 1e-10:
                    continue
                grad[i] += d / dist_ij
                grad[j] -= d / dist_ij

            for idx in boundary_idx:
                r = np.linalg.norm(best_points[idx])
                if r > 1e-10:
                    grad[idx] -= best_points[idx] / r

            best_points += lr * grad

            radii = np.linalg.norm(best_points, axis=1)
            mask_out = radii > 1.0
            if np.any(mask_out):
                best_points[mask_out] /= radii[mask_out, np.newaxis]

            new_score = compute_score(best_points)
            if new_score > best_score:
                best_score = new_score

            lr *= 0.999

    return best_points
# EVOLVE-BLOCK-END
