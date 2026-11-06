"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    def compute_radius(pts):
        """Compute max radius for given points."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        # Use KDTree for min pair distance (much faster than O(N^2))
        tree = cKDTree(pts)
        pairs = tree.query_pairs(2.0 * 0.1, output_type='ndarray')
        if len(pairs) == 0:
            min_pair = 1.0
        else:
            diff = pts[pairs[:, 0]] - pts[pairs[:, 1]]
            dists = np.sqrt(np.sum(diff**2, axis=1))
            min_pair = np.min(dists) / 2.0
        return min(min_pair, min_boundary)

    def compute_radius_full(pts):
        """Compute max radius using full O(N^2) for accuracy."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        return min(min_pair, min_boundary)

    def generate_hex_lattice(r):
        """Generate points on a hexagonal lattice with spacing 2r."""
        d = 2.0 * r
        if d <= 0:
            return np.array([])
        k = int(1.0 / d) + 2
        points = []
        sqrt3 = np.sqrt(3.0)
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * sqrt3 / 2.0
                dist = np.hypot(x, y)
                if dist + r <= 1.0:
                    points.append([x, y])
        if len(points) == 0:
            return np.array([])
        points = np.array(points)
        if len(points) >= N:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind="stable")[:N]
            return points[indices]
        return points

    def optimize(pts, max_time, init_step=0.005, softness=20.0):
        """
        Gradient ascent on soft-min surrogate using KD-tree neighbor search.
        Maximizes soft-min of (pair_dists/2) and (boundary_dists).
        """
        points = pts.copy().astype(np.float64)
        current_r = compute_radius(points)
        best_r = current_r
        best_points = points.copy()
        step = init_step
        stall_count = 0
        max_stall = 30

        while time.time() < max_time:
            # Compute neighbors within a reasonable range
            tree = cKDTree(points)
            # Search radius: we care about pairs within ~4*r of each other
            search_r = max(4.0 * current_r, 0.05)
            pairs = tree.query_pairs(search_r, output_type='ndarray')

            n = len(points)
            forces = np.zeros((n, 2))

            if len(pairs) > 0:
                i_idx = pairs[:, 0]
                j_idx = pairs[:, 1]
                diff = points[i_idx] - points[j_idx]
                dists = np.sqrt(np.sum(diff**2, axis=1))
                dists = np.maximum(dists, 1e-12)

                # Soft-min weights: exp(-(d - 2r)/softness) for d < 2r + margin
                margin = 0.02
                active = dists < (2.0 * current_r + margin)
                if np.any(active):
                    wa = np.exp(-(dists[active] - 2.0 * current_r) / softness)
                    # For each active pair, add force to both points
                    for k in range(len(i_idx)):
                        if not active[k]:
                            continue
                        w = wa[np.sum(active[:k+1]) - 1] if np.sum(active[:k+1]) > 0 else 0
                        # Recompute properly
                    # Vectorized: build weight array
                    w_all = np.zeros(len(pairs))
                    w_all[active] = np.exp(-(dists[active] - 2.0 * current_r) / softness)
                    # Force on i: w * (pi - pj) / dist
                    # Force on j: w * (pj - pi) / dist
                    unit = diff / dists[:, np.newaxis]
                    f_i = w_all[:, np.newaxis] * unit
                    # Scatter: add f_i to i_idx, subtract from j_idx
                    np.add.at(forces, i_idx, f_i)
                    np.add.at(forces, j_idx, -f_i)

            # Boundary forces
            radii = np.hypot(points[:, 0], points[:, 1])
            boundary_dists = 1.0 - radii
            boundary_active = boundary_dists < (current_r + 0.02)
            if np.any(boundary_active):
                w_b = np.exp(-(boundary_dists[boundary_active] - current_r) / softness)
                # Force toward center: -points[i] / |points[i]|
                idx_b = np.where(boundary_active)[0]
                dirs = -points[idx_b]
                norms = np.linalg.norm(dirs, axis=1)
                norms = np.maximum(norms, 1e-12)
                f_b = (w_b / norms)[:, np.newaxis] * dirs
                forces[idx_b] += f_b

            # Normalize forces
            force_norms = np.linalg.norm(forces, axis=1)
            max_f = np.max(force_norms) if len(force_norms) > 0 else 0
            if max_f > 1e-12:
                forces = forces / max_f

            # Update
            new_points = points + forces * step

            # Project inside unit circle
            new_radii = np.hypot(new_points[:, 0], new_points[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                new_points[outside] = new_points[outside] / new_radii[outside, np.newaxis] * 0.999

            new_r = compute_radius(new_points)

            if new_r > current_r:
                points = new_points
                current_r = new_r
                stall_count = 0
                if current_r > best_r:
                    best_r = current_r
                    best_points = points.copy()
                # Slowly increase step
                step = min(step * 1.02, init_step * 3.0)
            else:
                stall_count += 1
                step *= 0.85
                if step < 1e-7 or stall_count > max_stall:
                    break

        return best_points, best_r

    # 1. Find good hexagonal lattice starting point via binary search
    r_low = 0.01
    r_high = 0.08
    for _ in range(25):
        r_mid = (r_low + r_high) / 2.0
        pts = generate_hex_lattice(r_mid)
        if len(pts) >= N:
            r_low = r_mid
        else:
            r_high = r_mid

    points = generate_hex_lattice(r_low)
    if len(points) < N:
        # Fallback: random
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N)) * 0.5
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]

    best_points = points.copy()
    best_r = compute_radius(best_points)

    # 2. Multi-restart optimization
    # Phase 1: Main optimization from hex lattice
    phase1_deadline = start_time + 100.0
    pts, r = optimize(points, phase1_deadline, init_step=0.008, softness=20.0)
    if r > best_r:
        best_r = r
        best_points = pts.copy()

    # Phase 2: Fine polish
    phase2_deadline = deadline - 5.0
    pts, r = optimize(best_points, phase2_deadline, init_step=0.002, softness=50.0)
    if r > best_r:
        best_r = r
        best_points = pts.copy()

    # Phase 3: Try a few perturbed restarts if time allows
    if time.time() < deadline - 30.0:
        for trial in range(3):
            if time.time() > deadline - 25.0:
                break
            # Perturb the best solution
            perturb_scale = 0.003 * (trial + 1)
            perturbed = best_points + np.random.randn(N, 2) * perturb_scale
            # Project inside
            radii = np.hypot(perturbed[:, 0], perturbed[:, 1])
            outside = radii > 0.99
            if np.any(outside):
                perturbed[outside] = perturbed[outside] / radii[outside, np.newaxis] * 0.99

            trial_deadline = deadline - 10.0 - trial * 5.0
            pts, r = optimize(perturbed, trial_deadline, init_step=0.004, softness=30.0)
            if r > best_r:
                best_r = r
                best_points = pts.copy()

    # Final verification with full O(N^2)
    final_r = compute_radius_full(best_points)
    if final_r > best_r:
        best_r = final_r

    return best_points
# EVOLVE-BLOCK-END
