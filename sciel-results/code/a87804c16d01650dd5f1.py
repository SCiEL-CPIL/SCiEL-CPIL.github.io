"""Improved program using greedy hypergraph construction for fixed area targets."""
import numpy as np
from collections import Counter
import random

def construct():
    """
    Find 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area among all triples.
    
    Approach:
    1. Define the grid of all valid points.
    2. For a range of target doubled areas K, use a greedy algorithm to select
       30 points that maximize the number of triples with area K/2.
    3. The greedy step selects the point that forms the most new K-areas with
       the already selected points.
    4. Evaluate the final sets and return the best one.
    """
    
    # Precompute grid
    grid_points = []
    for x in range(-30, 31):
        for y in range(-30, 31):
            grid_points.append((x, y))
    N = len(grid_points)
    point_to_idx = {p: i for i, p in enumerate(grid_points)}
    
    def compute_max_multiplicity(points):
        """Compute the maximum multiplicity of any positive doubled area."""
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        
        idx_i, idx_j, idx_k = [], [], []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    idx_i.append(i)
                    idx_j.append(j)
                    idx_k.append(k)
        
        idx_i = np.array(idx_i)
        idx_j = np.array(idx_j)
        idx_k = np.array(idx_k)
        
        xi = pts[idx_i, 0]
        yi = pts[idx_i, 1]
        xj = pts[idx_j, 0]
        yj = pts[idx_j, 1]
        xk = pts[idx_k, 0]
        yk = pts[idx_k, 1]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        
        positive_mask = areas > 0
        if not np.any(positive_mask):
            return 0
        
        positive_areas = areas[positive_mask]
        counter = Counter(positive_areas.tolist())
        return max(counter.values())
    
    def get_points_on_line(u, v, K):
        """
        Find all grid points p such that area(u, v, p) == K/2.
        Returns a list of indices into grid_points.
        """
        x1, y1 = u
        x2, y2 = v
        dx = x2 - x1
        dy = y2 - y1
        
        points = []
        
        # Case 1: dx == 0 (vertical line uv)
        # Area = 0.5 * |dx * (y - y1) - dy * (x - x1)| = 0.5 * | -dy * (x - x1) | = 0.5 * |dy| * |x - x1|
        # We need 0.5 * |dy| * |x - x1| = K/2  => |x - x1| = K / |dy|
        # So x = x1 +/- K/|dy|.
        if dx == 0:
            if dy != 0 and K % abs(dy) == 0:
                dist = K // abs(dy)
                for sign in [1, -1]:
                    x = x1 + sign * dist
                    if -30 <= x <= 30:
                        for y in range(-30, 31):
                            if (x, y) != u and (x, y) != v:
                                points.append((x, y))
            return points
        
        # Case 2: dy == 0 (horizontal line uv)
        # Area = 0.5 * |dx * (y - y1)|.
        # We need 0.5 * |dx| * |y - y1| = K/2 => |y - y1| = K / |dx|.
        if dy == 0:
            if dx != 0 and K % abs(dx) == 0:
                dist = K // abs(dx)
                for sign in [1, -1]:
                    y = y1 + sign * dist
                    if -30 <= y <= 30:
                        for x in range(-30, 31):
                            if (x, y) != u and (x, y) != v:
                                points.append((x, y))
            return points
        
        # General Case
        # Equation: dx * (y - y1) - dy * (x - x1) = +/- K
        # -dy * x + dx * y + (dy * x1 - dx * y1) = +/- K
        # A * x + B * y + C = 0
        # A = -dy, B = dx, C = dy * x1 - dx * y1
        
        A = -dy
        B = dx
        C_base = dy * x1 - dx * y1
        
        for sign in [1, -1]:
            C = C_base - sign * K
            # A x + B y = -C
            # We iterate x from -30 to 30
            # y = (-C - A x) / B
            # Check if integer
            
            # Optimization: use gcd?
            # For small grid, direct iteration is fine.
            for x in range(-30, 31):
                numerator = -C - A * x
                if numerator % B == 0:
                    y = numerator // B
                    if -30 <= y <= 30:
                        p = (x, y)
                        if p != u and p != v:
                            points.append(p)
        
        return points
    
    best_score = 0
    best_points = None
    random.seed(42)
    
    # Try different target areas K
    # K is the doubled area. Must be even? No, area is 0.5 * det. Det is integer.
    # So doubled area is integer.
    # Small K values are good for dense clusters.
    for K in range(2, 102, 2):
        # Greedy construction
        selected = []
        counts = np.zeros(N, dtype=np.int32)
        
        # Initialize with a random point to break symmetry? Or just pick max count (all 0).
        # Let's pick a random start point.
        start_idx = random.randint(0, N - 1)
        start_point = grid_points[start_idx]
        selected.append(start_point)
        
        # We don't need to update counts for the first point as there are no pairs.
        
        for step in range(1, 30):
            # Find point with max count
            # If all counts are 0, pick random
            max_count = np.max(counts)
            if max_count == 0:
                # Pick random unselected point
                available = [i for i in range(N) if grid_points[i] not in selected]
                if not available:
                    break
                next_idx = random.choice(available)
            else:
                max_indices = np.where(counts == max_count)[0]
                # Filter out selected points (counts for selected might be non-zero if we didn't zero them)
                # We should zero out counts for selected points after selection to be safe, 
                # or just check.
                valid_indices = [i for i in max_indices if grid_points[i] not in selected]
                if not valid_indices:
                    available = [i for i in range(N) if grid_points[i] not in selected]
                    if not available:
                        break
                    next_idx = random.choice(available)
                else:
                    next_idx = random.choice(valid_indices)
            
            new_point = grid_points[next_idx]
            selected.append(new_point)
            
            # Update counts
            # For each existing point u in selected (except new_point)
            for u in selected[:-1]:
                # Find all points p that form area K with u and new_point
                pts_on_line = get_points_on_line(u, new_point, K)
                for p in pts_on_line:
                    if p in point_to_idx:
                        idx = point_to_idx[p]
                        counts[idx] += 1
            
            # Zero out count for the selected point to prevent re-selection
            # Actually, we check `not in selected` during selection, so it's fine.
            # But counts might accumulate. Let's zero it to keep counts clean for future?
            # No, counts represent "how many pairs in S form area K with this point".
            # Once selected, it doesn't matter.
            counts[next_idx] = 0 
        
        if len(selected) == 30:
            score = compute_max_multiplicity(selected)
            if score > best_score:
                best_score = score
                best_points = selected[:]
    
    if best_points is None:
        # Fallback
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = compute_max_multiplicity(best_points)
        
    return best_points
