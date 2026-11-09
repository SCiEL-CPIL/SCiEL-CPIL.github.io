"""Optimized 14-point configuration maximizing minimum triangle area over convex hull area."""
import numpy as np
import time

N = 14


def _min_triangle_area(points):
    """Compute minimum area of all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs(
                    (points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1])
                    - (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0])
                )
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute convex hull area using scipy."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    return hull.area


def _score(points):
    """Score = min triangle area / convex hull area."""
    mta = _min_triangle_area(points)
    cha = _convex_hull_area(points)
    return mta / cha


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Start with regular 14-gon on unit circle
    angles = 2 * np.pi * np.arange(N) / N
    points = np.column_stack((np.cos(angles), np.sin(angles)))

    # Parameterize as perturbation of angles and radii
    # We'll do a local search on the points themselves in polar coordinates
    best_points = points.copy()
    best_score = _score(best_points)

    # Local improvement: try small random perturbations
    import random
    rng = random.Random(42)

    for iteration in range(50000):
        if time.time() > deadline:
            break

        # Perturb a random point slightly
        candidate = best_points.copy()
        idx = rng.randrange(N)
        dx = rng.gauss(0, 0.05)
        dy = rng.gauss(0, 0.05)
        candidate[idx, 0] += dx
        candidate[idx, 1] += dy

        # Ensure points are reasonably spread (not too close)
        if np.min(np.linalg.norm(candidate - candidate[idx], axis=1)) < 0.1:
            continue

        cand_score = _score(candidate)
        if cand_score > best_score:
            best_score = cand_score
            best_points = candidate

    return best_points
