# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def erdos_szekeres_problem7() -> np.ndarray:
    n = 33
    k = 7
    deadline = time.time() + 580  # Safety margin

    # Precompute combinations of 6 indices from 32
    # This is constant for n=33
    all_combs = np.array(list(combinations(range(32), 6)))
    num_combs = len(all_combs)

    def is_valid(points):
        if points.shape != (n, 2):
            return False
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        # Check no two points too close
        # Vectorized distance check
        # dist_sq[i,j] = (pi - pj)^2
        # We can do this efficiently
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Check upper triangle
        iu = np.triu_indices(n, 1)
        if np.any(dist_sq[iu] < 1e-12):
            return False
        # Check no three collinear
        # area = x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
        # Vectorized for all triples is O(N^3) memory if not careful.
        # N=33, C(33,3) = 5456. Small.
        # We can iterate or vectorize.
        # Let's iterate for simplicity and safety, it's fast enough for 5456 checks.
        for i in range(n):
            for j in range(i + 1, n):
                for l in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[l]
                    area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                    if abs(area) < 1e-9:
                        return False
        return True

    def count_convex7(points):
        """Fast vectorized count of convex 7-gons."""
        # Sort lexicographically
        idx = np.lexsort((points[:, 1], points[:, 0]))
        sorted_points = points[idx]
        
        C = 0
        for pivot_idx in range(n):
            pivot = sorted_points[pivot_idx]
            others = np.delete(sorted_points, pivot_idx, axis=0)
            
            # Sort by angle
            angles = np.arctan2(others[:, 1] - pivot[1], others[:, 0] - pivot[0])
            order = np.argsort(angles)
            s = others[order]
            
            # Get points for all combinations
            # s shape (32, 2), all_combs shape (906192, 6)
            # pts shape (906192, 6, 2)
            pts = s[all_combs]
            
            v1 = pts[:, 0, :]
            v2 = pts[:, 1, :]
            v3 = pts[:, 2, :]
            v4 = pts[:, 3, :]
            v5 = pts[:, 4, :]
            v6 = pts[:, 5, :]
            
            # Calculate cross products
            # Turn at v1: (pivot, v1, v2)
            c1 = (v1[:, 0] - pivot[0]) * (v2[:, 1] - v1[:, 1]) - (v1[:, 1] - pivot[1]) * (v2[:, 0] - v1[:, 0])
            if not np.any(c1 > 1e-9):
                continue # Optimization: if no first turn is valid, skip rest?
                # Actually, we need to check all. But if c1 is never > 1e-9, C adds 0.
            
            # We can compute all and mask.
            c2 = (v2[:, 0] - v1[:, 0]) * (v3[:, 1] - v2[:, 1]) - (v2[:, 1] - v1[:, 1]) * (v3[:, 0] - v2[:, 0])
            c3 = (v3[:, 0] - v2[:, 0]) * (v4[:, 1] - v3[:, 1]) - (v3[:, 1] - v2[:, 1]) * (v4[:, 0] - v3[:, 0])
            c4 = (v4[:, 0] - v3[:, 0]) * (v5[:, 1] - v4[:, 1]) - (v4[:, 1] - v3[:, 1]) * (v5[:, 0] - v4[:, 0])
            c5 = (v5[:, 0] - v4[:, 0]) * (v6[:, 1] - v5[:, 1]) - (v5[:, 1] - v4[:, 1]) * (v6[:, 0] - v5[:, 0])
            c6 = (v6[:, 0] - v5[:, 0]) * (pivot[1] - v6[:, 1]) - (v6[:, 1] - v5[:, 1]) * (pivot[0] - v6[:, 0])
            c7 = (pivot[0] - v6[:, 0]) * (v1[:, 1] - pivot[1]) - (pivot[1] - v6[:, 1]) * (v1[:, 0] - pivot[0])
            
            valid = (c1 > 1e-9) & (c2 > 1e-9) & (c3 > 1e-9) & (c4 > 1e-9) & (c5 > 1e-9) & (c6 > 1e-9) & (c7 > 1e-9)
            C += np.sum(valid)
            
        return C

    # Initialize points on a line to minimize convex 7-gons
    # y = 0.5 + 0.1*x + noise
    np.random.seed(42)
    x = np.linspace(0.1, 0.9, n)
    y = 0.5 + 0.1 * x + np.random.normal(0, 0.01, n)
    points = np.column_stack((x, y))
    
    # Ensure validity
    if not is_valid(points):
        # Perturb if invalid
        for _ in range(100):
            points += np.random.normal(0, 0.01, (n, 2))
            points = np.clip(points, 0.01, 0.99)
            if is_valid(points):
                break

    best_points = points.copy()
    best_score = 1.0 / (1.0 + count_convex7(points))
    
    # Search
    start_time = time.time()
    step_size = 0.05
    no_improve_count = 0
    
    while time.time() - start_time < 550:
        # Random perturbation
        mask = np.random.rand(n, 2) < 0.2
        perturbation = (np.random.rand(n, 2) - 0.5) * 2 * step_size * mask
        new_points = best_points + perturbation
        new_points = np.clip(new_points, 0.001, 0.999)
        
        if is_valid(new_points):
            C = count_convex7(new_points)
            score = 1.0 / (1.0 + C)
            if score > best_score:
                best_score = score
                best_points = new_points.copy()
                no_improve_count = 0
                # Reduce step size slightly on improvement
                step_size *= 0.95
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    # Reset step size or jump
                    step_size = 0.05
                    no_improve_count = 0
                    # Random jump
                    best_points += np.random.normal(0, 0.1, (n, 2))
                    best_points = np.clip(best_points, 0.001, 0.999)
                    if not is_valid(best_points):
                         best_points = np.clip(best_points + np.random.normal(0, 0.01, (n, 2)), 0.001, 0.999)

    return best_points

# EVOLVE-BLOCK-END
