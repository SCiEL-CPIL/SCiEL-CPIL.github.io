"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def gen_hex_lattice(spacing):
        """Generate hexagonal lattice points inside unit disk with margin."""
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                # Use a margin that accounts for the circle radius
                if r <= 1.0 - spacing * 0.51:
                    pts.append((x, y))
        return np.array(pts)

    def compute_score(points):
        """Compute the packing radius: min(min_pair_dist/2, min_boundary_dist)."""
        n = len(points)
        if n < 2:
            return 0.0
        # Compute all pairwise distances using vectorized ops
        dx = points[:, 0:1] - points[0:1, 0].T
        dy = points[:, 1:2] - points[0:1, 1].T
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        # Compute minimum distance to boundary
        radii = np.hypot(points[:, 0], points[:, 1])
        min_boundary_dist = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    # Initial spacing estimate
    # Area argument: N * pi * r^2 <= pi * 1^2, so r <= 1/sqrt(N)
    # Hex packing efficiency ~0.907, so r ~ sqrt(0.907/N)
    initial_spacing = 2.0 * np.sqrt(0.907 / N)

    # Generate initial hexagonal lattice
    spacing = initial_spacing
    while True:
        pts = gen_hex_lattice(spacing)
        if len(pts) >= N:
            break
        spacing *= 0.98
        if time.time() > deadline - 5:
            break

    # Select N points closest to center
    radii = np.hypot(pts[:, 0], pts[:, 1])
    points = pts[np.argsort(radii, kind="stable")[:N]].copy()

    # Force-directed optimization
    best_score = compute_score(points)
    best_points = points.copy()

    # Iterative optimization (Vectorized)
    step_size = spacing * 0.03
    for iteration in range(2000):
        if time.time() > deadline:
            break

        # Compute forces
        n = len(points)
        dx = points[:, 0:1] - points[0:1, 0].T
        dy = points[:, 1:2] - points[0:1, 1].T
        dist = np.sqrt(dx * dx + dy * dy)

        # Avoid division by zero
        dist_safe = np.where(dist < 1e-10, 1e-10, dist)

        # Mask diagonal to avoid self-interaction
        mask = ~np.eye(n, dtype=bool)

        # Repulsion forces: stronger for closer pairs
        # Use 1/dist^2 for strong repulsion at close range
        inv_dist_sq = np.zeros_like(dist)
        inv_dist_sq[mask] = 1.0 / (dist_safe[mask] ** 2)

        # Only apply strong repulsion for pairs that are too close
        # Threshold: if distance < 2*spacing, apply strong force
        threshold = spacing * 1.5
        strong_mask = mask & (dist < threshold)

        fx = np.sum(np.where(strong_mask, inv_dist_sq * dx / dist_safe, 0.0), axis=1)
        fy = np.sum(np.where(strong_mask, inv_dist_sq * dy / dist_safe, 0.0), axis=1)

        # Also add a weaker repulsion for all pairs to maintain spacing
        weak_inv_dist = np.zeros_like(dist)
        weak_inv_dist[mask] = 0.1 / (dist_safe[mask] ** 2)
        fx += np.sum(weak_inv_dist * dx / dist_safe, axis=1)
        fy += np.sum(weak_inv_dist * dy / dist_safe, axis=1)

        # Boundary constraint force: push inward if too close to boundary
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_margin = 0.02
        over = radii > (1.0 - boundary_margin)
        if np.any(over):
            mag = (radii[over] - (1.0 - boundary_margin)) * 5.0
            fx[over] -= mag * points[over, 0] / radii[over]
            fy[over] -= mag * points[over, 1] / radii[over]

        # Normalize forces
        force_norms = np.sqrt(fx * fx + fy * fy)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            fx /= max_force
            fy /= max_force

        # Update positions
        points[:, 0] += step_size * fx
        points[:, 1] += step_size * fy

        # Project back inside unit circle
        radii = np.hypot(points[:, 0], points[:, 1])
        mask_out = radii > 0.999
        if np.any(mask_out):
            scale = 0.999 / radii[mask_out]
            points[mask_out, 0] *= scale
            points[mask_out, 1] *= scale

        # Compute score
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()

        # Anneal step size
        step_size *= 0.998

    # Final verification and cleanup
    # Ensure all points are strictly inside the unit circle
    r_final = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = r_final > 0.9999
    if np.any(mask):
        scale = 0.9999 / r_final[mask]
        best_points[mask, 0] *= scale
        best_points[mask, 1] *= scale

    # Check for non-finite values
    if not np.all(np.isfinite(best_points)):
        # Fallback to hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[radius <= 1.0 - spacing / 2.0]
        if len(pts) >= N:
            return pts[np.argsort(radius, kind="stable")[:N]]
        else:
            return pts

    return best_points
# EVOLVE-BLOCK-END
