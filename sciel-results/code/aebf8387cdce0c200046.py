"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import combinations
from collections import Counter

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from collections import Counter

_DEADLINE = 50.0  # safety margin before 90s kill

# Precompute triple indices for n=30 once
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

def _score_incremental(points, moved_idx, old_pt, new_pt, old_areas=None):
    """
    Incremental scoring: only recompute triangles involving the moved point.
    Returns (max_count, area_counts_dict).
    """
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    
    # Compute all triangle areas
    det = np.abs((pts[_J_IDX, 0] - pts[_I_IDX, 0]) * (pts[_K_IDX, 1] - pts[_I_IDX, 1]) - 
                 (pts[_J_IDX, 1] - pts[_I_IDX, 1]) * (pts[_K_IDX, 0] - pts[_I_IDX, 0]))
    det = det[det > 0]
    
    if len(det) == 0:
        return 0, {}
    
    vals, counts = np.unique(det, return_counts=True)
    area_dict = dict(zip(vals.tolist(), counts.tolist()))
    return int(counts.max()), area_dict

def _score_max_only(points):
    """Return just the max multiplicity."""
    pts = np.array(points, dtype=np.int64)
    xi = pts[_I_IDX, 0]; yi = pts[_I_IDX, 1]
    xj = pts[_J_IDX, 0]; yj = pts[_J_IDX, 1]
    xk = pts[_K_IDX, 0]; yk = pts[_K_IDX, 1]
    det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    det = det[det > 0]
    if len(det) == 0:
        return 0
    vals, counts = np.unique(det, return_counts=True)
    return int(counts.max())

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
    # Take first 30 in sorted order for determinism
    return sorted(pts)[:30]

def _gen_hex_config(radius):
    """Generate a hexagonal lattice configuration."""
    pts = set()
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            s = -q - r
            if abs(s) > radius:
                continue
            # Convert axial to cartesian
            x = q + s // 2
            y = r
            # Scale and center
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
    for _ in range(15):
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

def _greedy_set_cover(points, deadline, target_area=None):
    """
    Greedy set-cover approach: for a fixed target area A, find which point
    replacements maximize the count of triangles with area A.
    
    Strategy: For each candidate replacement position, compute how many new
    triangles of area A would be formed. Greedily select the best replacement.
    """
    start_time = time.time()
    current_points = list(points)
    current_score, current_areas = _score_fast(current_points)
    best_points = list(current_points)
    best_score = current_score
    best_areas = dict(current_areas)
    
    n = len(current_points)
    point_set = set(current_points)
    
    if target_area is None and current_areas:
        target_area = max(current_areas, key=current_areas.get)
    
    # Precompute: for each point index, which other points form triangles
    # of area target_area with it (as the "reference")
    pts_arr = np.array(current_points, dtype=np.int64)
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve < max_no_improve:
        # For each point, find the best replacement
        best_overall_improvement = 0
        best_idx = -1
        best_new_pt = None
        
        for idx in range(n):
            old_pt = current_points[idx]
            
            # Try nearby positions
            for radius in [2, 3, 5, 8]:
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                        if not (-30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30):
                            continue
                        if new_pt != old_pt and new_pt in point_set:
                            continue
                        
                        # Compute how many triangles of area target_area this creates
                        # with the other n-1 points
                        new_count = 0
                        for j in range(n):
                            if j == idx:
                                continue
                            p1 = current_points[idx]  # new point
                            p2 = current_points[j]
                            # Need a third point - but we only have pairs here
                            # Instead, count triangles where new_pt is one vertex
                            # and the other two are from existing points
                            pass
                        
                        # Better approach: just score the full configuration
                        # but only if it looks promising based on pair counts
                        # For efficiency, use a heuristic: count pairs that
                        # would form area target_area with some third point
                        pair_count = 0
                        for j in range(n):
                            if j == idx:
                                continue
                            p2 = np.array(current_points[j])
                            p_new = np.array(new_pt)
                            # Vector from p_new to p2
                            v = p2 - p_new
                            # For each other point k, check if |det(v, pk - p_new)| == target_area
                            # But that's O(n) per pair, so O(n^2) total
                            # Too slow - use a simpler heuristic
                            pass
                        
                        # Simpler: just evaluate if the move improves target area count
                        # by checking a subset of triangles
                        point_set.discard(old_pt)
                        point_set.add(new_pt)
                        current_points[idx] = new_pt
                        
                        sc, ar = _score_fast(current_points)
                        tc = ar.get(target_area, 0) if target_area else 0
                        
                        point_set.discard(new_pt)
                        point_set.add(old_pt)
                        current_points[idx] = old_pt
                        
                        improvement = tc - current_areas.get(target_area, 0)
                        if improvement > best_overall_improvement:
                            best_overall_improvement = improvement
                            best_idx = idx
                            best_new_pt = new_pt
        
        if best_idx >= 0 and best_new_pt is not None:
            # Apply the best move
            old_pt = current_points[best_idx]
            point_set.discard(old_pt)
            point_set.add(best_new_pt)
            current_points[best_idx] = best_new_pt
            
            new_score, new_areas = _score_fast(current_points)
            current_score = new_score
            current_areas = new_areas
            
            if new_score > best_score:
                best_score = new_score
                best_points = list(current_points)
                best_areas = dict(new_areas)
            no_improve = 0
        else:
            no_improve += 1
    
    return best_points, best_score

def _local_search(points, deadline, target_area=None):
    """
    Hybrid local search: combine greedy set-cover with simulated annealing.
    Uses greedy approach first, then refines with SA.
    """
    start_time = time.time()
    
    # Phase 1: Greedy set-cover (first 40% of time)
    greedy_deadline = start_time + 0.4 * (deadline - start_time)
    greedy_points, greedy_score = _greedy_set_cover(points, greedy_deadline, target_area)
    
    # Phase 2: Simulated annealing refinement (remaining time)
    sa_deadline = deadline
    sa_points, sa_score = _sa_refine(greedy_points, sa_deadline, target_area)
    
    # Return the better of the two
    if sa_score >= greedy_score:
        return sa_points, sa_score
    else:
        return greedy_points, greedy_score

def _sa_refine(points, deadline, target_area=None):
    """Simulated annealing refinement for final polishing."""
    start_time = time.time()
    current_points = list(points)
    current_score, current_areas = _score_fast(current_points)
    best_points = list(current_points)
    best_score = current_score
    best_areas = dict(current_areas)
    
    n = len(current_points)
    point_set = set(current_points)
    
    if target_area is None and current_areas:
        target_area = max(current_areas, key=current_areas.get)
    
    temp = 3.0
    cooling = 0.997
    min_temp = 0.01
    no_improve = 0
    steps = 0
    
    while time.time() < deadline and temp > min_temp:
        # Pick a point to move
        idx = np.random.randint(n)
        old_pt = current_points[idx]
        
        # Generate new position - bias toward small moves for refinement
        r = np.random.random()
        if r < 0.6:
            # Small perturbation
            radius = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            new_x = old_pt[0] + np.random.randint(-radius, radius + 1)
            new_y = old_pt[1] + np.random.randint(-radius, radius + 1)
        elif r < 0.85:
            # Medium perturbation
            new_x = old_pt[0] + np.random.randint(-5, 6)
            new_y = old_pt[1] + np.random.randint(-5, 6)
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
        
        # Acceptance criterion - prioritize improving target area count
        target_improved = False
        if target_area is not None:
            old_target_count = current_areas.get(target_area, 0)
            new_target_count = new_areas.get(target_area, 0)
            target_improved = new_target_count > old_target_count
        
        if new_score > current_score:
            current_score = new_score
            current_areas = new_areas
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_points = list(current_points)
                best_areas = dict(new_areas)
        elif target_improved:
            # Accept if target area count improves (even if overall score same)
            current_score = new_score
            current_areas = new_areas
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_points = list(current_points)
                best_areas = dict(new_areas)
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / max(temp, 0.01))
                if np.random.random() > prob:
                    # Revert
                    current_points[idx] = old_pt
                    point_set.discard(new_pt)
                    point_set.add(old_pt)
                    no_improve += 1
                else:
                    current_score = new_score
                    current_areas = new_areas
                    no_improve += 1
            else:
                current_score = new_score
                current_areas = new_areas
                no_improve += 1
        
        temp *= cooling
        
        # Reheat if stuck
        if no_improve > 100:
            temp = max(temp, 2.0)
            no_improve = 0
    
    return best_points, best_score

def _polish(points, deadline):
    """Final high-precision polish: try all single-point moves to neighboring cells."""
    current_points = list(points)
    current_score, current_areas = _score_fast(current_points)
    best_points = list(current_points)
    best_score = current_score
    point_set = set(current_points)
    n = len(current_points)
    
    if current_areas:
        target_area = max(current_areas, key=current_areas.get)
    else:
        target_area = None
    
    # Multiple passes with decreasing neighborhood size
    for radius in [4, 3, 2, 1]:
        for idx in range(n):
            if time.time() > deadline:
                return best_points, best_score
            old_pt = current_points[idx]
            best_for_idx = None
            best_for_idx_score = current_score
            best_for_idx_target = current_areas.get(target_area, 0) if target_area else 0
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                    if not (-30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30):
                        continue
                    if new_pt != old_pt and new_pt in point_set:
                        continue
                    
                    # Try this move
                    point_set.discard(old_pt)
                    point_set.add(new_pt)
                    current_points[idx] = new_pt
                    
                    sc, ar = _score_fast(current_points)
                    tc = ar.get(target_area, 0) if target_area else 0
                    
                    if sc > best_for_idx_score or (sc == best_for_idx_score and tc > best_for_idx_target):
                        best_for_idx_score = sc
                        best_for_idx_target = tc
                        best_for_idx = new_pt
                    else:
                        # Revert
                        current_points[idx] = old_pt
                        point_set.discard(new_pt)
                        point_set.add(old_pt)
            
            if best_for_idx is not None:
                # Apply the best move for this index
                point_set.discard(old_pt)
                point_set.add(best_for_idx)
                current_points[idx] = best_for_idx
                current_score, current_areas = _score_fast(current_points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = list(current_points)
    
    return best_points, best_score

def _try_all_configs():
    """
    Main search: evaluate initial configs, then use hybrid local search
    (greedy set-cover + SA) on top ones, then polish.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Generate and score initial configs
    configs = _gen_initial_configs()
    
    scored_configs = []
    for config in configs:
        if time.time() > deadline - 20.0:
            break
        if len(set(config)) != 30:
            continue
        score, areas = _score_fast(config)
        scored_configs.append((score, areas, config))
    
    if not scored_configs:
        # Fallback
        fallback = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
        return fallback
    
    scored_configs.sort(key=lambda x: -x[0])
    
    best_points = scored_configs[0][2]
    best_score = scored_configs[0][0]
    best_areas = scored_configs[0][1]
    
    # Run hybrid local search on top configurations
    # Allocate time: top 4 configs get more time, next 4 get less
    top_k = min(8, len(scored_configs))
    for i in range(top_k):
        if time.time() > deadline - 10.0:
            break
        score, areas, config = scored_configs[i]
        target_area = max(areas, key=areas.get) if areas else None
        
        # Time allocation: first 4 get 8s each, next 4 get 4s each
        if i < 4:
            time_alloc = 8.0
        else:
            time_alloc = 4.0
        
        remaining = deadline - time.time() - 5.0
        if remaining < 2.0:
            break
        actual_time = min(time_alloc, remaining)
        
        try:
            lp, ls = _local_search(config, time.time() + actual_time, target_area=target_area)
            if ls > best_score:
                best_score = ls
                best_points = lp
                _, best_areas = _score_fast(lp)
        except Exception:
            pass
    
    # Polish the best result with remaining time
    if time.time() < deadline - 3.0:
        try:
            pp, ps = _polish(best_points, deadline - 1.0)
            if ps > best_score:
                best_score = ps
                best_points = pp
        except Exception:
            pass
    
    return best_points

def construct():
    """Main entry point: find 30 points maximizing max area multiplicity."""
    start_time = time.time()
    
    try:
        result = _try_all_configs()
    except Exception:
        result = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
    
    # Verify validity
    pts_set = set(result)
    if len(pts_set) != 30:
        result = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
    for (x, y) in result:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            result = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
            break
    
    return result
# EVOLVE-BLOCK-END
