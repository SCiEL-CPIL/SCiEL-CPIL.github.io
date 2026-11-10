import time
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    # Precompute triple indices for n=30
    _N = 30
    _i_idx, _j_idx, _k_idx = [], [], []
    for _i in range(_N - 2):
        for _j in range(_i + 1, _N - 1):
            for _k in range(_j + 1, _N):
                _i_idx.append(_i)
                _j_idx.append(_j)
                _k_idx.append(_k)
    _I_IDX = np.array(_i_idx, dtype=np.int32)
    _J_IDX = np.array(_j_idx, dtype=np.int32)
    _K_IDX = np.array(_k_idx, dtype=np.int32)
    _NUM_TRIPLES = len(_I_IDX)  # C(30,3) = 4060

    def _score_fast(points):
        """Fast scoring using precomputed indices. Returns (max_count, area_counts_dict)."""
        pts = np.array(points, dtype=np.int64)
        xi = pts[_I_IDX, 0]; yi = pts[_I_IDX, 1]
        xj = pts[_J_IDX, 0]; yj = pts[_J_IDX, 1]
        xk = pts[_K_IDX, 0]; yk = pts[_K_IDX, 1]
        det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
        det = det[det > 0]
        if len(det) == 0:
            return 0, {}
        vals, counts = np.unique(det, return_counts=True)
        area_dict = dict(zip(vals.tolist(), counts.tolist()))
        return int(counts.max()), area_dict

    def _gen_rect_grid(rows, cols, dx, dy):
        """Generate a centered rectangular grid, return exactly 30 points or None."""
        pts = set()
        total_w = (cols - 1) * dx
        total_h = (rows - 1) * dy
        if total_w > 60 or total_h > 60:
            return None
        ox = -total_w // 2
        oy = -total_h // 2
        for i in range(rows):
            for j in range(cols):
                x = ox + i * dx
                y = oy + j * dy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.add((x, y))
        if len(pts) < 30:
            return None
        return sorted(pts)[:30]

    def _gen_hex_config(radius):
        """Generate a hexagonal lattice configuration."""
        pts = set()
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                s = -q - r
                if abs(s) > radius:
                    continue
                x = q + s // 2
                y = r
                cx = x * 2
                cy = y * 2 + (q % 2)
                if -30 <= cx <= 30 and -30 <= cy <= 30:
                    pts.add((cx, cy))
        if len(pts) < 30:
            return None
        return sorted(pts)[:30]

    def _gen_lattice(a, b, c, d, count=30):
        """Generate points on a 2D lattice spanned by vectors (a,b) and (c,d)."""
        pts = set()
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * a + j * c
                y = i * b + j * d
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.add((x, y))
            if len(pts) >= count:
                break
        if len(pts) < count:
            return None
        return sorted(pts)[:count]

    def _gen_initial_configs():
        """Generate diverse initial configurations."""
        configs = []
        
        # Rectangular grids
        for rows in range(3, 9):
            for cols in range(3, 11):
                if rows * cols < 30:
                    continue
                for dx in [1, 2, 3]:
                    for dy in [1, 2, 3]:
                        cfg = _gen_rect_grid(rows, cols, dx, dy)
                        if cfg and len(cfg) == 30:
                            configs.append(cfg)
        
        # Hexagonal
        for r in range(3, 6):
            cfg = _gen_hex_config(r)
            if cfg:
                configs.append(cfg)
        
        # Various lattices
        for a, b in [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (3, 1)]:
            for c, d in [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (1, 3)]:
                if a * d - b * c == 0:
                    continue
                cfg = _gen_lattice(a, b, c, d)
                if cfg:
                    configs.append(cfg)
        
        # Random configurations
        rng = np.random.RandomState(42)
        for _ in range(20):
            pts = set()
            while len(pts) < 30:
                x = rng.randint(-30, 31)
                y = rng.randint(-30, 31)
                pts.add((x, y))
            configs.append(sorted(pts))
        
        # Remove duplicates
        unique_configs = []
        seen = set()
        for cfg in configs:
            key = tuple(sorted(cfg))
            if key not in seen:
                seen.add(key)
                unique_configs.append(cfg)
        
        return unique_configs

    def _local_search(points, deadline, target_area=None):
        """Simulated annealing local search with target area focus."""
        current_points = list(points)
        current_score, current_areas = _score_fast(current_points)
        best_points = list(current_points)
        best_score = current_score
        best_areas = dict(current_areas)
        
        n = len(current_points)
        point_set = set(current_points)
        
        if target_area is None and current_areas:
            target_area = max(current_areas, key=current_areas.get)
        
        temp = 5.0
        cooling = 0.995
        min_temp = 0.01
        no_improve = 0
        steps = 0
        
        while time.time() < deadline and temp > min_temp:
            # Pick a point to move
            idx = np.random.randint(n)
            old_pt = current_points[idx]
            
            # Generate new position
            if np.random.random() < 0.6:
                # Small perturbation
                radius = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.25, 0.2, 0.15, 0.1])
                new_x = old_pt[0] + np.random.randint(-radius, radius + 1)
                new_y = old_pt[1] + np.random.randint(-radius, radius + 1)
            elif np.random.random() < 0.85:
                # Medium perturbation
                new_x = old_pt[0] + np.random.randint(-10, 11)
                new_y = old_pt[1] + np.random.randint(-10, 11)
            else:
                # Large perturbation
                new_x = np.random.randint(-30, 31)
                new_y = np.random.randint(-30, 31)
            
            new_pt = (new_x, new_y)
            
            if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                temp *= cooling
                continue
            if new_pt != old_pt and new_pt in point_set:
                temp *= cooling
                continue
            
            # Apply move
            point_set.discard(old_pt)
            point_set.add(new_pt)
            current_points[idx] = new_pt
            
            new_score, new_areas = _score_fast(current_points)
            steps += 1
            
            # Acceptance criterion
            if new_score > current_score:
                current_score = new_score
                current_areas = new_areas
                no_improve = 0
                if new_score > best_score:
                    best_score = new_score
                    best_points = list(current_points)
                    best_areas = dict(new_areas)
            elif target_area is not None and new_areas.get(target_area, 0) > best_areas.get(target_area, 0):
                # Accept if target area improves
                current_score = new_score
                current_areas = new_areas
                no_improve = 0
                if new_score > best_score:
                    best_score = new_score
                    best_points = list(current_points)
                    best_areas = dict(new_areas)
            else:
                # Revert move
                point_set.discard(new_pt)
                point_set.add(old_pt)
                current_points[idx] = old_pt
                no_improve += 1
            
            temp *= cooling
            
            # If stuck for too long, reset to best
            if no_improve > 500:
                current_points = list(best_points)
                current_score = best_score
                current_areas = dict(best_areas)
                point_set = set(current_points)
                no_improve = 0
                temp = max(temp, 2.0)
        
        return best_points, best_score, best_areas

    # Generate initial configurations
    configs = _gen_initial_configs()
    
    # Evaluate all initial configs
    best_overall = None
    best_score_overall = 0
    best_areas_overall = {}
    
    for cfg in configs:
        if time.time() > deadline:
            break
        if len(cfg) != 30 or len(set(cfg)) != 30:
            continue
        if any(abs(x) > 30 or abs(y) > 30 for x, y in cfg):
            continue
        
        s, areas = _score_fast(cfg)
        if s > best_score_overall:
            best_score_overall = s
            best_overall = list(cfg)
            best_areas_overall = dict(areas)
    
    if best_overall is None:
        # Fallback
        best_overall = [(x, y) for x in range(6) for y in range(5)]
        best_score_overall, best_areas_overall = _score_fast(best_overall)
    
    # Run local search from the best initial config
    if time.time() < deadline:
        improved_points, improved_score, improved_areas = _local_search(
            best_overall, deadline, 
            max(best_areas_overall, key=best_areas_overall.get) if best_areas_overall else None
        )
        if improved_score > best_score_overall:
            best_overall = improved_points
            best_score_overall = improved_score
            best_areas_overall = improved_areas
    
    # Try local search from a few other good configs if time permits
    other_configs = [c for c in configs if c != best_overall][:5]
    for cfg in other_configs:
        if time.time() > deadline - 10:
            break
        if len(cfg) != 30 or len(set(cfg)) != 30:
            continue
        
        s, areas = _score_fast(cfg)
        if s >= best_score_overall - 2:  # Only try configs close to best
            improved_points, improved_score, improved_areas = _local_search(
                cfg, min(deadline, time.time() + 15),
                max(areas, key=areas.get) if areas else None
            )
            if improved_score > best_score_overall:
                best_overall = improved_points
                best_score_overall = improved_score
                best_areas_overall = improved_areas
    
    return best_overall
