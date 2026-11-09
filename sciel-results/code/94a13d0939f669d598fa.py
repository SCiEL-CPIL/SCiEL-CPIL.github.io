# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import combinations

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Minimize the number of convex 7-gons among 33 points in the unit square.
    Uses simulated annealing with an efficient convex-position count.
    """
    start = time.time()
    deadline = start + 580.0  # 50s margin before 630s limit
    n = 33
    k = 7

    # ---- helpers ----
    def is_valid(pts):
        if pts.shape != (n, 2):
            return False
        if not np.all(np.isfinite(pts)):
            return False
        if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
            return False
        # distinct points
        for i in range(n):
            for j in range(i + 1, n):
                d2 = (pts[i, 0] - pts[j, 0]) ** 2 + (pts[i, 1] - pts[j, 1]) ** 2
                if d2 < 1e-12:
                    return False
        # no three collinear
        for i in range(n):
            for j in range(i + 1, n):
                for l in range(j + 1, n):
                    area = abs(pts[i, 0] * (pts[j, 1] - pts[l, 1]) +
                               pts[j, 0] * (pts[l, 1] - pts[i, 1]) +
                               pts[l, 0] * (pts[i, 1] - pts[j, 1]))
                    if area < 1e-9:
                        return False
        return True

    def count_convex_kgoms(points, k=7):
        """Count the number of k-point subsets in convex position."""
        n_pts = len(points)
        count = 0
        # For each combination of k points, check if all are on the convex hull
        for combo in combinations(range(n_pts), k):
            sub = points[list(combo)]
            if all_points_extreme(sub):
                count += 1
        return count

    def all_points_extreme(pts):
        """Check if all points in a small set are on the convex hull."""
        m = len(pts)
        if m <= 3:
            return True
        # Sort by x, then y
        order = sorted(range(m), key=lambda i: (pts[i, 0], pts[i, 1]))
        # Build convex hull
        hull = []
        # Lower hull
        for i in order:
            while len(hull) >= 2:
                p0 = hull[-2]
                p1 = hull[-1]
                cross = (pts[p1, 0] - pts[p0, 0]) * (pts[i, 1] - pts[p0, 1]) - \
                        (pts[p1, 1] - pts[p0, 1]) * (pts[i, 0] - pts[p0, 0])
                if cross <= 1e-9:
                    hull.pop()
                else:
                    break
            hull.append(i)
        # Upper hull
        upper = []
        for i in reversed(order):
            while len(upper) >= 2:
                p0 = upper[-2]
                p1 = upper[-1]
                cross = (pts[p1, 0] - pts[p0, 0]) * (pts[i, 1] - pts[p0, 1]) - \
                        (pts[p1, 1] - pts[p0, 1]) * (pts[i, 0] - pts[p0, 0])
                if cross <= 1e-9:
                    upper.pop()
                else:
                    break
            upper.append(i)
        full_hull = set(hull[:-1] + upper[:-1])
        return len(full_hull) == m

    def score(points):
        C = count_convex_kgoms(points)
        return 1.0 / (1.0 + C)

    # ---- initial point generation ----
    def generate_initial():
        """Generate an initial point set with structure to reduce convex 7-gons."""
        rng = random.Random(42)
        pts = np.zeros((n, 2))
        # Use a few clusters to reduce convex positions
        n_clusters = 5
        cluster_centers = [(0.2, 0.2), (0.5, 0.5), (0.8, 0.3), (0.3, 0.7), (0.7, 0.8)]
        pts_per_cluster = [n // n_clusters + (1 if i < n % n_clusters else 0) for i in range(n_clusters)]
        
        idx = 0
        for c in range(n_clusters):
            cx, cy = cluster_centers[c]
            for _ in range(pts_per_cluster[c]):
                pts[idx, 0] = cx + np.random.randn() * 0.08
                pts[idx, 1] = cy + np.random.randn() * 0.08
                idx += 1
        pts = np.clip(pts, 0.01, 0.99)
        return pts

    # ---- main search ----
    pts = generate_initial()
    if not is_valid(pts):
        np.random.seed(42)
        pts = np.random.rand(n, 2)
        pts = np.clip(pts, 0.01, 0.99)
    
    current_score = score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    alpha = 0.9995
    rng = random.Random(12345)
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = rng.randrange(n)
        orig = pts[idx].copy()
        
        # Random step with adaptive size
        step_size = 0.03 + 0.04 * (1 - (time.time() - start) / (deadline - start))
        pts[idx] = orig + np.random.randn(2) * step_size
        pts[idx] = np.clip(pts[idx], 0.01, 0.99)
        
        if not is_valid(pts):
            pts[idx] = orig
            continue
        
        new_score = score(pts)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    current_score = new_score
                else:
                    pts[idx] = orig
            else:
                current_score = new_score
        
        T = max(T_min, T * alpha)
    
    # Final validity check
    if not is_valid(best_pts):
        np.random.seed(0)
        best_pts = np.random.rand(n, 2)
        best_pts = np.clip(best_pts, 0.01, 0.99)
    
    return best_pts.astype(np.float64)

# EVOLVE-BLOCK-END
