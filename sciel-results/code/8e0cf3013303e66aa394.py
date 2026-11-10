"""Combined approach: Precomputed indices + Incremental scoring + Structured search."""

import time
import random
import math
from collections import Counter
import numpy as np

# Precompute triple indices once (i < j < k for n=30)
_n_points = 30
_triple_indices = None

def _get_triple_indices():
    global _triple_indices
    if _triple_indices is None:
        i_idx, j_idx, k_idx = np.indices((_n_points, _n_points, _n_points))
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        _triple_indices = (i_idx[mask], j_idx[mask], k_idx[mask])
    return _triple_indices

def _compute_all_areas(pts):
    """Compute all doubled areas for a set of points. Returns array of areas (zeros for collinear)."""
    ti, tj, tk = _get_triple_indices()
    xi = pts[ti, 0]
    yi = pts[ti, 1]
    xj = pts[tj, 0]
    yj = pts[tj, 1]
    xk = pts[tk, 0]
    yk = pts[tk, 1]
    areas = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    return areas

def _max_multiplicity(points):
    """Compute the maximum multiplicity of any positive doubled area among all triples."""
    if len(points) < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    areas = _compute_all_areas(pts)
    areas = areas[areas > 0]
    if len(areas) == 0:
        return 0
    vals, counts = np.unique(areas, return_counts=True)
    return int(counts.max())

def _generate_initial_candidates():
    """Generate diverse initial point sets focused on high-symmetry structures."""
    candidates = []
    
    # 1. Rectangular Grids - these naturally produce many equal-area triangles
    grid_dims = [(5, 6), (6, 5), (4, 8), (8, 4), (3, 10), (10, 3), (2, 15), (15, 2), (5, 5), (6, 4)]
    for w, h in grid_dims:
        if w * h == 30:
            pts = [(i, j) for i in range(w) for j in range(h)]
            cx = (w - 1) // 2
            cy = (h - 1) // 2
            centered = [(x - cx, y - cy) for x, y in pts]
            candidates.append(centered)
    
    # 2. Dense small grids with perturbations
    for _ in range(20):
        # Start with a 6x5 grid
        base = [(i, j) for i in range(6) for j in range(5)]
        pts = []
        used = set()
        for x, y in base:
            # Small random perturbation
            if random.random() < 0.3:
                dx = random.randint(-1, 1)
                dy = random.randint(-1, 1)
                new_pt = (x + dx, y + dy)
                if new_pt not in used and -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                    pts.append(new_pt)
                    used.add(new_pt)
                else:
                    pts.append((x, y))
                    used.add((x, y))
            else:
                pts.append((x, y))
                used.add((x, y))
        if len(set(pts)) == 30:
            candidates.append(pts)
    
    # 3. Hexagonal-like lattice
    for scale in [1, 2]:
        hex_pts = []
        for row in range(6):
            for col in range(5):
                x = col * 2 * scale + (row % 2) * scale
                y = row * scale
                hex_pts.append((x, y))
        if len(hex_pts) == 30:
            cx = sum(p[0] for p in hex_pts) // 30
            cy = sum(p[1] for p in hex_pts) // 30
            candidates.append([(x - cx, y - cy) for x, y in hex_pts])
    
    # 4. Triangular lattice
    for scale in [1, 2]:
        tri_pts = []
        for row in range(6):
            for col in range(5):
                x = col * scale
                y = row * scale
                tri_pts.append((x, y))
        if len(tri_pts) == 30:
            cx = sum(p[0] for p in tri_pts) // 30
            cy = sum(p[1] for p in tri_pts) // 30
            candidates.append([(x - cx, y - cy) for x, y in tri_pts])
    
    # 5. Concentric patterns - points on small circles
    for r in range(1, 8):
        circle_pts = []
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                if x * x + y * y <= r * r:
                    circle_pts.append((x, y))
        if len(circle_pts) >= 30:
            random.shuffle(circle_pts)
            candidates.append(circle_pts[:30])
    
    # 6. Diamond patterns
    for r in range(4, 8):
        diamond_pts = []
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                if abs(x) + abs(y) <= r:
                    diamond_pts.append((x, y))
        if len(diamond_pts) >= 30:
            random.shuffle(diamond_pts)
            candidates.append(diamond_pts[:30])
    
    # 7. Random from small grid
    for _ in range(15):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-4, 4)
            y = random.randint(-4, 4)
            pts.add((x, y))
        candidates.append(list(pts))
    
    # 8. Random from medium grid
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-6, 6)
            y = random.randint(-6, 6)
            pts.add((x, y))
        candidates.append(list(pts))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # 1. Initialize with diverse candidates
    candidates = _generate_initial_candidates()
    
    best_score = -1
    best_points = None
    best_areas = None
    
    # Evaluate initial candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        score = _max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]
            best_areas = _compute_all_areas(np.array(pts, dtype=np.int64))
            
    if best_points is None:
        # Fallback
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _max_multiplicity(best_points)
        best_areas = _compute_all_areas(np.array(best_points, dtype=np.int64))
        
    # 2. Local Search with incremental updates
    current_points = best_points[:]
    current_score = best_score
    current_areas = best_areas.copy()
    occupied = set(current_points)
    
    # Track the current best area value and its count
    def _get_best_area_info(areas):
        """Get the best area value and its multiplicity."""
        pos_areas = areas[areas > 0]
        if len(pos_areas) == 0:
            return 0, 0
        vals, counts = np.unique(pos_areas, return_counts=True)
        max_idx = np.argmax(counts)
        return int(vals[max_idx]), int(counts[max_idx])
    
    current_best_area, current_best_count = _get_best_area_info(current_areas)
    
    # Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.998
    no_improve_count = 0
    max_no_improve = 300
    
    # Precompute which triples involve each point index for faster updates
    ti, tj, tk = _get_triple_indices()
    
    # For each point index, find which triple indices involve it
    point_triple_map = [set() for _ in range(_n_points)]
    for t_idx in range(len(ti)):
        point_triple_map[ti[t_idx]].add(t_idx)
        point_triple_map[tj[t_idx]].add(t_idx)
        point_triple_map[tk[t_idx]].add(t_idx)
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = current_points[idx]
        
        # Generate a neighbor with adaptive radius
        if no_improve_count > max_no_improve // 2:
            radius = random.choice([1, 2, 3, 4, 5, 7, 10])
        else:
            if random.random() < 0.75:
                radius = 1
            elif random.random() < 0.9:
                radius = random.choice([2, 3])
            else:
                radius = random.choice([4, 5, 7])
        
        dx = random.randint(-radius, radius)
        dy = random.randint(-radius, radius)
        
        if dx == 0 and dy == 0:
            continue
            
        new_p = (old_p[0] + dx, old_p[1] + dy)
        
        if -30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30 and new_p not in occupied:
            # Perform move
            occupied.remove(old_p)
            occupied.add(new_p)
            current_points[idx] = new_p
            
            # Update only the affected triples (those involving point idx)
            affected_triples = point_triple_map[idx]
            pts_arr = np.array(current_points, dtype=np.int64)
            
            for t_idx in affected_triples:
                i, j, k = ti[t_idx], tj[t_idx], tk[t_idx]
                xi, yi = pts_arr[i, 0], pts_arr[i, 1]
                xj, yj = pts_arr[j, 0], pts_arr[j, 1]
                xk, yk = pts_arr[k, 0], pts_arr[k, 1]
                current_areas[t_idx] = abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
            
            # Quick check: can we beat current best?
            # Count occurrences of current_best_area
            new_count = np.sum(current_areas == current_best_area)
            
            # Check if there's a new area with higher count
            # Only do full unique if the quick check suggests improvement
            if new_count >= current_best_count:
                # Full evaluation to be sure
                new_score = _max_multiplicity(current_points)
                new_best_area, new_best_count = _get_best_area_info(current_areas)
                current_best_area, current_best_count = new_best_area, new_best_count
            else:
                new_score = current_best_count  # Conservative estimate
            
            delta = new_score - current_score
            
            if delta > 0:
                # Accept and update best
                current_score = new_score
                best_score = new_score
                best_points = current_points[:]
                best_areas = current_areas.copy()
                no_improve_count = 0
            elif delta == 0:
                # Accept (neutral move)
                current_score = new_score
                no_improve_count += 1
            else:
                # Reject with probability exp(-delta/T)
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_score = new_score
                    no_improve_count += 1
                else:
                    # Revert
                    occupied.remove(new_p)
                    occupied.add(old_p)
                    current_points[idx] = old_p
                    # Revert areas
                    pts_arr = np.array(current_points, dtype=np.int64)
                    for t_idx in affected_triples:
                        i, j, k = ti[t_idx], tj[t_idx], tk[t_idx]
                        xi, yi = pts_arr[i, 0], pts_arr[i, 1]
                        xj, yj = pts_arr[j, 0], pts_arr[j, 1]
                        xk, yk = pts_arr[k, 0], pts_arr[k, 1]
                        current_areas[t_idx] = abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
                    no_improve_count += 1
            
            # Cool down
            T = max(T_min, T * cooling_rate)
            
            # Restart if stuck
            if no_improve_count > max_no_improve or T <= T_min * 1.05:
                if random.random() < 0.3:
                    # Restart from best
                    current_points = best_points[:]
                    current_score = best_score
                    current_areas = best_areas.copy()
                    occupied = set(current_points)
                    current_best_area, current_best_count = _get_best_area_info(current_areas)
                    T = 1.0
                    no_improve_count = 0
                else:
                    # Perturb best
                    current_points = best_points[:]
                    occupied = set(current_points)
                    # Make 1-3 random moves
                    for _ in range(random.randint(1, 3)):
                        ridx = random.randint(0, 29)
                        old = current_points[ridx]
                        r = random.randint(1, 4)
                        ndx = random.randint(-r, r)
                        ndy = random.randint(-r, r)
                        np_ = (old[0] + ndx, old[1] + ndy)
                        if -30 <= np_[0] <= 30 and -30 <= np_[1] <= 30 and np_ not in occupied:
                            occupied.remove(old)
                            occupied.add(np_)
                            current_points[ridx] = np_
                    current_score = _max_multiplicity(current_points)
                    current_areas = _compute_all_areas(np.array(current_points, dtype=np.int64))
                    current_best_area, current_best_count = _get_best_area_info(current_areas)
                    T = 1.0
                    no_improve_count = 0
    
    return best_points
