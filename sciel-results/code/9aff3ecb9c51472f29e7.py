import time
import itertools
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 85.0
    N = 30

    # Precompute all triples of indices 0..29
    all_triples = list(itertools.combinations(range(N), 3))

    def score(points):
        """Return the maximum multiplicity of a positive doubled area."""
        points_arr = np.array(points, dtype=np.int64)
        counts = {}
        for i, j, k in all_triples:
            # 2x area = |x_i(y_j - y_k) - x_j(y_i - y_k) + x_k(y_i - y_j)|
            x_i, y_i = points_arr[i]
            x_j, y_j = points_arr[j]
            x_k, y_k = points_arr[k]
            det = x_i * (y_j - y_k) - x_j * (y_i - y_k) + x_k * (y_i - y_j)
            a = abs(det)
            if a > 0:
                counts[a] = counts.get(a, 0) + 1
        return max(counts.values()) if counts else 0

    def score_fast(points):
        """Faster score using numpy vectorization for area computation."""
        points_arr = np.array(points, dtype=np.int64)
        max_mult = 0
        # Process in chunks to avoid memory issues
        chunk_size = 500
        for start in range(0, len(all_triples), chunk_size):
            chunk = all_triples[start:start+chunk_size]
            idx = np.array(chunk)
            p1 = points_arr[idx[:, 0]]
            p2 = points_arr[idx[:, 1]]
            p3 = points_arr[idx[:, 2]]
            # 2x area via cross product
            dets = np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) - 
                          p2[:, 0] * (p1[:, 1] - p3[:, 1]) + 
                          p3[:, 0] * (p1[:, 1] - p2[:, 1]))
            dets = dets[dets > 0]
            if len(dets) == 0:
                continue
            # Count multiplicities
            vals, counts = np.unique(dets, return_counts=True)
            chunk_max = counts.max()
            if chunk_max > max_mult:
                max_mult = chunk_max
        return max_mult

    # Initial point: random grid
    def make_grid():
        pts = []
        # 6x5 grid
        for x in range(6):
            for y in range(5):
                pts.append((x, y))
        return pts

    def random_point():
        return (random.randint(-30, 30), random.randint(-30, 30))

    # Hill climbing / local search
    def local_search(points, time_budget=0.5):
        points = list(points)
        current_score = score(points)
        deadline = start_time + time_budget
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            # Try moving each point
            random.shuffle(points)  # randomize order for diversity
            for idx in range(N):
                if (time.time() - start_time) >= deadline:
                    break
                x0, y0 = points[idx]
                # Try all possible moves in a neighborhood
                # Sample some moves to keep it fast
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x0 + dx, y0 + dy
                        if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                            continue
                        # Check for duplicates
                        if (nx, ny) in points:
                            continue
                        new_points = points.copy()
                        new_points[idx] = (nx, ny)
                        new_score = score(new_points)
                        if new_score > current_score:
                            points = new_points
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
        return points, current_score

    # Start with grid
    points = make_grid()
    best_points = list(points)
    best_score = score(points)

    # Do some rounds of local search with random restarts
    num_restarts = 5
    for r in range(num_restarts):
        if (time.time() - start_time) >= time_limit - 5:
            break
        # Perturb the current best
        if r == 0:
            start_pts = list(best_points)
        else:
            start_pts = list(best_points)
            # Random perturbation
            for _ in range(random.randint(1, 5)):
                idx = random.randint(0, N-1)
                start_pts[idx] = random_point()
                # Ensure uniqueness
                while start_pts.count(start_pts[idx]) > 1:
                    start_pts[idx] = random_point()

        # Local search
        result_pts, result_score = local_search(start_pts, time_budget=15.0)
        if result_score > best_score:
            best_score = result_score
            best_points = list(result_pts)
            print(f"New best: {best_score} at restart {r}")

    # Final: try to improve with full local search on best
    if (time.time() - start_time) < time_limit - 3:
        result_pts, result_score = local_search(best_points, time_budget=75.0)
        if result_score > best_score:
            best_score = result_score
            best_points = list(result_pts)

    return best_points
