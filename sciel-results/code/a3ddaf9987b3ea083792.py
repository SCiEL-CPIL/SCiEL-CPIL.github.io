"""Combined program: affine-transformed hexagonal superlattice + local polish."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import cKDTree


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def compute_score(points):
        """Compute packing radius via cKDTree (fast)."""
        n = len(points)
        if n < 2:
            return 0.0
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        min_dist = np.min(dists[:, 1])
        radii = np.hypot(points[:, 0], points[:, 1])
        min_boundary = 1.0 - np.max(radii)
        return min(min_dist / 2.0, min_boundary)

    def compute_score_full(points):
        """Exact O(N^2) score for final verification."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        radii = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - np.max(radii)
        return min(min_dist / 2.0, min_boundary)

    def apply_affine(base, scale, rot, shear):
        """Apply global affine transform: scale -> rotate -> shear."""
        c, s = math.cos(rot), math.sin(rot)
        R = np.array([[c, s], [-s, c]])  # rotation
        S = np.array([[1.0, 0.0], [shear, 1.0]])  # shear
        M = (scale * (R @ S)).T  # applied as M @ point
        return base @ M.T

    def gen_hex_superlattice(scale, rot, shear):
        """Generate a hexagonal superlattice, apply affine transform, prune to N points inside unit circle."""
        # Generate a dense hexagonal grid in the base frame
        k = int(1.2 / (scale * 0.9)) + 4
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * scale
                y = j * scale * math.sqrt(3.0) / 2.0
                pts.append((x, y))
        base = np.array(pts)
        transformed = apply_affine(base, 1.0, rot, shear)
        # Keep only points inside the unit circle (with margin for circle radius)
        radii = np.hypot(transformed[:, 0], transformed[:, 1])
        # We need points that can host a circle; keep those with radius < 1 (will refine margin later)
        inside = radii < 1.0
        transformed = transformed[inside]
        if len(transformed) < N:
            return None
        # Sort by radius and take the N closest to center
        radii = np.hypot(transformed[:, 0], transformed[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        return transformed[idx]

    def local_optimize(points, time_budget):
        """Local optimization with adaptive step size, moving points to improve the min radius."""
        points = points.copy().astype(np.float64)
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        end_time = time.time() + time_budget

        step_size = 0.005
        min_step = 1e-7
        no_improve_count = 0
        max_no_improve = 80

        while time.time() < end_time:
            radii = np.hypot(points[:, 0], points[:, 1])
            tree = cKDTree(points)
            dists, _ = tree.query(points, k=2)
            min_dists = dists[:, 1]

            min_pair_dist = np.min(min_dists)
            min_boundary = 1.0 - np.max(radii)
            current_score = min(min_pair_dist / 2.0, min_boundary)

            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()

            improved = False
            # Try several random point moves per iteration
            for _ in range(30):
                idx = np.random.randint(N)
                if min_pair_dist / 2.0 < min_boundary:
                    # Pairwise constraint limiting: move away from nearest neighbor
                    _, nn_idx = tree.query(points[idx], k=2)
                    nn_idx = nn_idx[1]
                    diff = points[idx] - points[nn_idx]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-10:
                        direction = diff / norm
                    else:
                        angle = np.random.uniform(0, 2 * math.pi)
                        direction = np.array([math.cos(angle), math.sin(angle)])
                else:
                    # Boundary constraint limiting: move inward
                    r = np.linalg.norm(points[idx])
                    if r > 1e-10:
                        direction = -points[idx] / r
                    else:
                        angle = np.random.uniform(0, 2 * math.pi)
                        direction = np.array([math.cos(angle), math.sin(angle)])

                # Add randomness
                noise_angle = np.random.uniform(-0.6, 0.6)
                cos_n, sin_n = math.cos(noise_angle), math.sin(noise_angle)
                direction = np.array([
                    direction[0] * cos_n - direction[1] * sin_n,
                    direction[0] * sin_n + direction[1] * cos_n
                ])

                new_point = points[idx] + direction * step_size
                if np.hypot(new_point[0], new_point[1]) < 1.0:
                    trial = points.copy()
                    trial[idx] = new_point
                    trial_score = compute_score(trial)
                    if trial_score > current_score + 1e-12:
                        points = trial
                        current_score = trial_score
                        improved = True
                        if trial_score > best_score:
                            best_score = trial_score
                            best_points = points.copy()

            if improved:
                no_improve_count = 0
                step_size = min(step_size * 1.08, 0.03)
            else:
                no_improve_count += 1
                step_size = max(step_size * 0.85, min_step)

            if no_improve_count >= max_no_improve:
                no_improve_count = 0
                step_size = 0.005

        return best_points, best_score

    def polish(points, time_budget):
        """High-precision polish pass with very small steps to tighten the solution."""
        points = points.copy().astype(np.float64)
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        end_time = time.time() + time_budget

        step_size = 1e-5
        min_step = 1e-9
        no_improve_count = 0

        while time.time() < end_time and step_size > min_step:
            radii = np.hypot(points[:, 0], points[:, 1])
            tree = cKDTree(points)
            dists, _ = tree.query(points, k=2)
            min_dists = dists[:, 1]

            min_pair_dist = np.min(min_dists)
            min_boundary = 1.0 - np.max(radii)
            current_score = min(min_pair_dist / 2.0, min_boundary)

            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()

            improved = False
            for _ in range(40):
                idx = np.random.randint(N)
                if min_pair_dist / 2.0 < min_boundary:
                    _, nn_idx = tree.query(points[idx], k=2)
                    nn_idx = nn_idx[1]
                    diff = points[idx] - points[nn_idx]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-10:
                        direction = diff / norm
                    else:
                        angle = np.random.uniform(0, 2 * math.pi)
                        direction = np.array([math.cos(angle), math.sin(angle)])
                else:
                    r = np.linalg.norm(points[idx])
                    if r > 1e-10:
                        direction = -points[idx] / r
                    else:
                        angle = np.random.uniform(0, 2 * math.pi)
                        direction = np.array([math.cos(angle), math.sin(angle)])

                new_point = points[idx] + direction * step_size
                if np.hypot(new_point[0], new_point[1]) < 1.0:
                    trial = points.copy()
                    trial[idx] = new_point
                    trial_score = compute_score(trial)
                    if trial_score > current_score + 1e-14:
                        points = trial
                        current_score = trial_score
                        improved = True
                        if trial_score > best_score:
                            best_score = trial_score
                            best_points = points.copy()

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                step_size *= 0.7
                if no_improve_count > 15:
                    break

        return best_points, best_score

    # --- Phase 1: Explore affine-transformed superlattices ---
    best_points = None
    best_score = -1.0

    # Estimate a good base spacing
    s_est = math.sqrt(2 * math.pi * 0.95 ** 2 / (N * math.sqrt(3)))

    # Try a range of scale factors and rotations/shears
    scale_trials = [s_est * (0.98 + 0.01 * t) for t in range(15)]
    rot_trials = [0.0, math.pi / 6, math.pi / 12, -math.pi / 12, math.pi / 18, -math.pi / 18]
    shear_trials = [0.0, 0.05, -0.05, 0.1, -0.1]

    # Deterministic initial best from plain hex lattice
    for scale in scale_trials:
        if time.time() > deadline - 30:
            break
        for rot in rot_trials:
            if time.time() > deadline - 30:
                break
            for shear in shear_trials:
                if time.time() > deadline - 30:
                    break
                cand = gen_hex_superlattice(scale, rot, shear)
                if cand is not None:
                    sc = compute_score(cand)
                    if sc > best_score:
                        best_score = sc
                        best_points = cand.copy()

    if best_points is None:
        # Fallback: random
        best_points = np.random.randn(N, 2) * 0.8
        max_r = np.max(np.hypot(best_points[:, 0], best_points[:, 1]))
        if max_r > 0.9:
            best_points *= 0.9 / max_r
        best_score = compute_score(best_points)

    # --- Phase 2: Local optimization on the best initial configuration ---
    time_left = deadline - time.time() - 20.0  # reserve 20s for polish
    if time_left > 5.0:
        best_points, best_score = local_optimize(best_points, time_left)

    # --- Phase 3: High-precision polish ---
    time_left = deadline - time.time() - 3.0
    if time_left > 2.0:
        best_points, polished_score = polish(best_points, time_left)
        if polished_score > best_score:
            best_score = polished_score

    # Final exact verification
    final_score = compute_score_full(best_points)
    if final_score > best_score:
        best_score = final_score

    return best_points
# EVOLVE-BLOCK-END
