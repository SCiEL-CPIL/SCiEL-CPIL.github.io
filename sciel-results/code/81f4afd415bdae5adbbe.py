"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    def compute_score(points):
        n = len(points)
        if n < 2:
            return 0.0
        # Compute minimum pairwise distance
        tree = cKDTree(points)
        pairs = tree.query_pairs(2.0)
        if len(pairs) == 0:
            min_dist = np.inf
        else:
            min_dist = np.inf
            for i, j in pairs:
                d = np.linalg.norm(points[i] - points[j])
                if d < min_dist:
                    min_dist = d
        # Compute minimum distance to boundary
        radii = np.hypot(points[:, 0], points[:, 1])
        min_boundary = 1.0 - np.max(radii)
        return min(min_dist / 2.0, min_boundary)

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

    # Start with hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        points = gen_hex_lattice(spacing)
        if len(points) >= N:
            # Sort by radius and take first N
            radii = np.hypot(points[:, 0], points[:, 1])
            points = points[np.argsort(radii, kind='stable')[:N]]
            break
        spacing *= 0.98

    best_score = compute_score(points)
    best_points = points.copy()

    # Local optimization using gradient-free method
    n_iter = 0
    while time.time() < deadline - 5.0:
        n_iter += 1
        # Try to improve by moving points
        current_score = compute_score(points)
        
        # Find the most constrained point
        radii = np.hypot(points[:, 0], points[:, 1])
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        min_dists = dists[:, 1]
        
        # Score components
        min_pair_dist = np.min(min_dists)
        min_boundary = 1.0 - np.max(radii)
        
        # Determine which constraint is limiting
        if min_pair_dist / 2.0 < min_boundary:
            # Pairwise constraint is limiting
            idx = np.argmin(min_dists)
            # Find nearest neighbor
            _, nn_idx = tree.query(points[idx], k=2)
            nn_idx = nn_idx[1]
            # Move point away from nearest neighbor
            diff = points[idx] - points[nn_idx]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                direction = diff / norm
            else:
                direction = np.array([1.0, 0.0])
            # Move outward
            step = 0.001
            new_point = points[idx] + direction * step
            # Check if still inside
            if np.hypot(new_point[0], new_point[1]) < 1.0:
                trial = points.copy()
                trial[idx] = new_point
                trial_score = compute_score(trial)
                if trial_score > current_score:
                    points = trial
                    if points_score > best_score:
                        best_score = points_score
                        best_points = points.copy()
        else:
            # Boundary constraint is limiting
            idx = np.argmax(radii)
            # Move inward
            direction = -points[idx] / (np.linalg.norm(points[idx]) + 1e-10)
            step = 0.001
            new_point = points[idx] + direction * step
            trial = points.copy()
            trial[idx] = new_point
            trial_score = compute_score(trial)
            if trial_score > current_score:
                points = trial
                if compute_score(points) > best_score:
                    best_score = compute_score(points)
                    best_points = points.copy()

    # Final check
    final_score = compute_score(best_points)
    if final_score > best_score:
        best_score = final_score

    return best_points
# EVOLVE-BLOCK-END
