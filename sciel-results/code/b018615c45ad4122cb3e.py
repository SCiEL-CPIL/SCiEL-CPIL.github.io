# EVOLVE-BLOCK-START
import numpy as np
import time


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon with equidistant vertices.
    Optimizes radial distances to maximize the minimum vertex score.
    """
    n = 20
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill

    def compute_score(polygon):
        """Compute the minimum vertex score for a given polygon. Returns 0.0 if invalid."""
        pts = np.array(polygon, dtype=float)
        if pts.shape != (n, 2):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0

        # Normalize: subtract centroid, divide by mean distance to centroid
        centroid = pts.mean(axis=0)
        pts_norm = pts - centroid
        dists_to_centroid = np.linalg.norm(pts_norm, axis=1)
        mean_dist = dists_to_centroid.mean()
        if mean_dist <= 0.01:
            return 0.0
        pts_norm = pts_norm / mean_dist

        # Check for repeated points
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pts_norm[i] - pts_norm[j]) < 0.05:
                    return 0.0

        # Check strict convexity: all cross products same sign and nonzero
        cross_signs = []
        for i in range(n):
            p0 = pts_norm[i]
            p1 = pts_norm[(i + 1) % n]
            p2 = pts_norm[(i + 2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) < 1e-12:
                return 0.0
            cross_signs.append(np.sign(cross))

        if len(set(cross_signs)) != 1:
            return 0.0

        # Compute vertex scores
        min_score = 1.0
        for i in range(n):
            v = pts_norm[i]
            dists = np.linalg.norm(pts_norm - v, axis=1)
            dists[i] = np.inf  # exclude self
            d = np.sort(dists)  # 19 distances

            if d[0] < 0.05:
                return 0.0

            gaps = d[1:] - d[:-1]  # 18 gaps
            k = np.argmin(gaps)  # first index minimizing gap

            D = (d[k] + d[k + 1]) / 2.0
            if D < 0.1:
                return 0.0

            if k == 0:
                d_sel = d[0:4]
            elif k == 17:
                d_sel = d[15:19]
            else:
                d_sel = d[k - 1:k + 3]

            ratios = np.maximum(d_sel / D, D / d_sel)
            raw = np.mean(ratios)
            score = min(1.0, 1.0 / raw)

            if score < min_score:
                min_score = score
                if min_score == 0.0:
                    return 0.0

        return min_score

    # Initialize with regular 20-gon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    def get_polygon(radii):
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.stack([x, y], axis=1)

    best_polygon = get_polygon(np.ones(n))
    best_score = compute_score(best_polygon)

    # Local optimization with adaptive step sizes and multi-scale approach
    rng = np.random.default_rng(42)
    
    # Phase 1: Coarse search
    step_size = 0.1
    max_iterations_phase1 = 5000
    
    radii = np.ones(n).copy()
    
    for iteration in range(max_iterations_phase1):
        if time.time() > deadline - 120.0:
            break

        # Pick 1-3 random vertices to perturb
        num_perturb = rng.integers(1, 4)
        idxs = rng.choice(n, num_perturb, replace=False)

        new_radii = radii.copy()
        for idx in idxs:
            delta = rng.uniform(-step_size, step_size)
            new_radii[idx] += delta
            if new_radii[idx] < 0.1:
                new_radii[idx] = 0.1 + rng.uniform(0, 0.05)

        new_polygon = get_polygon(new_radii)
        new_score = compute_score(new_polygon)

        if new_score > best_score:
            best_score = new_score
            best_polygon = new_polygon
            radii = new_radii
            step_size *= 0.97
        else:
            if iteration % 50 == 0:
                step_size = max(0.01, step_size * 1.05)

    # Phase 2: Fine search around best found
    step_size = 0.02
    max_iterations_phase2 = 10000

    for iteration in range(max_iterations_phase2):
        if time.time() > deadline - 30.0:
            break

        # Focus on worst-scoring vertices
        # Compute per-vertex scores to identify worst
        pts_norm = best_polygon - best_polygon.mean(axis=0)
        dists_to_centroid = np.linalg.norm(pts_norm, axis=1)
        mean_dist = dists_to_centroid.mean()
        if mean_dist > 0.01:
            pts_norm = pts_norm / mean_dist
        
        vertex_scores = []
        for i in range(n):
            v = pts_norm[i]
            dists = np.linalg.norm(pts_norm - v, axis=1)
            dists[i] = np.inf
            d = np.sort(dists)
            if d[0] < 0.05:
                vertex_scores.append(0.0)
                continue
            gaps = d[1:] - d[:-1]
            k = np.argmin(gaps)
            D = (d[k] + d[k + 1]) / 2.0
            if D < 0.1:
                vertex_scores.append(0.0)
                continue
            if k == 0:
                d_sel = d[0:4]
            elif k == 17:
                d_sel = d[15:19]
            else:
                d_sel = d[k - 1:k + 3]
            ratios = np.maximum(d_sel / D, D / d_sel)
            raw = np.mean(ratios)
            score = min(1.0, 1.0 / raw)
            vertex_scores.append(score)
        
        # Identify worst vertices
        worst_idx = np.argmin(vertex_scores)
        
        # Perturb worst vertex and its neighbors
        num_perturb = rng.integers(1, 3)
        # Include worst vertex with high probability
        if rng.random() < 0.7:
            idxs = [worst_idx]
            if num_perturb > 1:
                other_idxs = rng.choice(n - 1, num_perturb - 1, replace=False)
                other_idxs = other_idxs[other_idxs < worst_idx] + (other_idxs >= worst_idx)
                idxs = list(set(idxs + list(other_idxs)))[:num_perturb]
        else:
            idxs = rng.choice(n, num_perturb, replace=False)

        new_radii = radii.copy()
        for idx in idxs:
            delta = rng.uniform(-step_size, step_size)
            new_radii[idx] += delta
            if new_radii[idx] < 0.1:
                new_radii[idx] = 0.1 + rng.uniform(0, 0.05)

        new_polygon = get_polygon(new_radii)
        new_score = compute_score(new_polygon)

        if new_score > best_score:
            best_score = new_score
            best_polygon = new_polygon
            radii = new_radii
            step_size *= 0.98
        else:
            if iteration % 100 == 0:
                step_size = max(0.005, step_size * 1.03)

    # Final verification
    final_score = compute_score(best_polygon)
    if final_score < best_score - 1e-6:
        # Something went wrong, return regular polygon
        best_polygon = get_polygon(np.ones(n))
        best_score = compute_score(best_polygon)

    return best_polygon


# EVOLVE-BLOCK-END
