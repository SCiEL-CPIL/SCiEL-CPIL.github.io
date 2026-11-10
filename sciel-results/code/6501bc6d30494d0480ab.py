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

_DEADLINE = 78.0  # safety margin before 90s kill

# Precompute triple indices for n=30 (C(30,3) = 4060)
def _precompute_triple_indices(n=30):
    """Precompute index arrays for all C(n,3) triples."""
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    return (np.array(i_idx, dtype=np.int32),
            np.array(j_idx, dtype=np.int32),
            np.array(k_idx, dtype=np.int32))

_TRIPLE_IDX = _precompute_triple_indices(30)

def _score_with_areas(points):
    """Return (max_count, area_counts_dict) for detailed analysis. Fast vectorized version."""
    n = len(points)
    if n < 3:
        return 0, {}
    pts = np.array(points, dtype=np.int64)
    
    i_idx, j_idx, k_idx = _TRIPLE_IDX
    
    xi = pts[i_idx, 0]; yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]; yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]; yk = pts[k_idx, 1]
    
    det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    det = det[det > 0]
    if len(det) == 0:
        return 0, {}
    vals, counts = np.unique(det, return_counts=True)
    max_count = int(counts.max())
    # Build dict only for areas with count >= 2 (to save time)
    area_dict = {}
    for v, c in zip(vals, counts):
        if c >= 2:
            area_dict[int(v)] = int(c)
    return max_count, area_dict

def _score_fast(points):
    """Fast score computation - just returns max multiplicity."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    
    i_idx, j_idx, k_idx = _TRIPLE_IDX
    
    xi = pts[i_idx, 0]; yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]; yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]; yk = pts[k_idx, 1]
    
    det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    det = det[det > 0]
    if len(det) == 0:
        return 0
    vals, counts = np.unique(det, return_counts=True)
    return int(counts.max())

def _gen_grid_config(rows, cols, dx=1, dy=1, offset=(0, 0)):
    """Generate a grid configuration with given parameters."""
    pts = set()
    ox, oy = offset
    for i in range(rows):
        for j in range(cols):
            x = ox + i * dx
            y = oy + j * dy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
    return list(pts)[:30]

def _gen_hex_config(center=(0, 0), radius=4):
    """Generate a hexagonal packing configuration."""
    pts = set()
    cx, cy = center
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x = cx + dx
            y = cy + dy
            # Hexagonal offset
            if dx % 2 == 1:
                y = cy + dy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
    # Also try with different spacing
    return list(pts)[:30]

def _gen_lattice_config(basis1, basis2, count=30):
    """Generate points on a 2D lattice defined by two basis vectors."""
    pts = set()
    b1x, b1y = basis1
    b2x, b2y = basis2
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * b1x + j * b2x
            y = i * b1y + j * b2y
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
            if len(pts) >= count:
                break
        if len(pts) >= count:
            break
    return list(pts)[:count]

def _gen_grid_variants():
    """Generate various grid configurations with different spacings."""
    configs = []
    
    # Standard grids with different dimensions and spacings
    for rows in range(3, 9):
        for cols in range(3, 9):
            if rows * cols < 30:
                continue
            for dx in [1, 2, 3, 4]:
                for dy in [1, 2, 3, 4]:
                    # Center the grid
                    total_w = (cols - 1) * dx
                    total_h = (rows - 1) * dy
                    if total_w > 60 or total_h > 60:
                        continue
                    ox = -total_w // 2
                    oy = -total_h // 2
                    config = _gen_grid_config(rows, cols, dx, dy, (ox, oy))
                    if len(config) == 30:
                        configs.append(config)
    
    # Lattice configurations
    lattices = [
        (1, 0, 0, 2), (1, 0, 0, 3), (2, 0, 0, 1), (2, 0, 0, 2),
        (1, 1, 1, -1), (2, 1, 1, -2), (1, 2, 1, -1),
        (3, 0, 0, 1), (3, 0, 0, 2), (4, 0, 0, 1),
        (1, 0, 2, 1), (1, 0, 1, 2), (2, 1, 1, 1),
    ]
    for b1x, b1y, b2x, b2y in lattices:
        config = _gen_lattice_config((b1x, b1y), (b2x, b2y))
        if len(config) == 30:
            configs.append(config)
    
    # Parabolic configurations
    for a in [1, 2, 3, 4, 5]:
        for offset_y in range(-20, 21, 2):
            pts = set()
            for x in range(-20, 21):
                y = (a * x * x) // 10 + offset_y
                if -30 <= y <= 30:
                    pts.add((x, y))
                if len(pts) >= 30:
                    break
            if len(pts) == 30:
                configs.append(list(pts))
    
    # Random configurations
    np.random.seed(42)
    for _ in range(15):
        pts = set()
        while len(pts) < 30:
            x = np.random.randint(-30, 31)
            y = np.random.randint(-30, 31)
            pts.add((x, y))
        configs.append(list(pts))
    
    return configs

def _local_search_targeted(points, time_limit=50.0, target_area=None):
    """Local search that targets a specific area value or maximizes max multiplicity.
    Uses fast scoring and more aggressive exploration."""
    start_time = time.time()
    best_points = list(points)
    best_score = _score_fast(best_points)
    current_points = list(points)
    current_score = best_score
    
    n = len(current_points)
    current_set = set(current_points)
    
    # If no target area, use the most common area from current config
    if target_area is None:
        _, area_dict = _score_with_areas(current_points)
        if area_dict:
            target_area = max(area_dict, key=area_dict.get)
    
    temp = 4.0
    cooling = 0.999
    min_temp = 0.1
    no_improve_count = 0
    best_target_count = 0
    if target_area:
        _, area_dict = _score_with_areas(current_points)
        best_target_count = area_dict.get(target_area, 0)
    
    # Pre-generate candidate positions for faster sampling
    all_positions = np.array([(x, y) for x in range(-30, 31) for y in range(-30, 31)], dtype=np.int32)
    
    while time.time() < time_limit:
        # Pick a random point to move
        idx = np.random.randint(n)
        old_pt = current_points[idx]
        
        # Generate candidate new positions: mix of local and global
        r = np.random.random()
        if r < 0.5:
            # Local move: within 3 units
            new_x = old_pt[0] + np.random.randint(-3, 4)
            new_y = old_pt[1] + np.random.randint(-3, 4)
        elif r < 0.8:
            # Medium move: within 10 units
            new_x = old_pt[0] + np.random.randint(-10, 11)
            new_y = old_pt[1] + np.random.randint(-10, 11)
        else:
            # Global random
            new_x = np.random.randint(-30, 31)
            new_y = np.random.randint(-30, 31)
        
        new_pt = (int(new_x), int(new_y))
        
        # Check bounds and uniqueness
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            temp *= cooling
            continue
        if new_pt != old_pt and new_pt in current_set:
            temp *= cooling
            continue
        
        # Try the move
        current_set.discard(old_pt)
        current_set.add(new_pt)
        current_points[idx] = new_pt
        new_score = _score_fast(current_points)
        
        # Evaluate acceptance
        accepted = False
        if new_score > current_score:
            accepted = True
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_points = list(current_points)
                if target_area:
                    _, area_dict = _score_with_areas(current_points)
                    best_target_count = max(best_target_count, area_dict.get(target_area, 0))
        elif new_score == current_score:
            # Accept equal score moves with some probability to explore
            if np.random.random() < 0.3:
                accepted = True
                no_improve_count += 1
        else:
            # Accept worse moves with probability based on temperature
            delta = new_score - current_score
            prob = np.exp(delta / max(temp, 0.01))
            if np.random.random() < prob:
                accepted = True
                no_improve_count += 1
        
        if not accepted:
            # Revert
            current_set.discard(new_pt)
            current_set.add(old_pt)
            current_points[idx] = old_pt
        
        temp *= cooling
        
        # Reset temperature if stuck for too long
        if no_improve_count > 200:
            temp = max(temp, 3.0)
            no_improve_count = 0
    
    return best_points, best_score

def _try_all_configs():
    """Try multiple configurations with targeted local search."""
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Generate initial configurations
    configs = _gen_grid_variants()
    
    # Evaluate all configs
    scored_configs = []
    for config in configs:
        if time.time() > deadline - 5.0:
            break
        if len(set(config)) != 30:
            continue
        score = _score_fast(config)
        scored_configs.append((score, config))
    
    # Sort by score
    scored_configs.sort(key=lambda x: -x[0])
    
    best_points = scored_configs[0][1] if scored_configs else _gen_grid_config(5, 6, 1, 1, (-2, -2))
    best_score = scored_configs[0][0] if scored_configs else 0
    
    # Run local search on top configurations
    # Use more time per search since scoring is faster now
    top_k = min(8, len(scored_configs))
    for score, config in scored_configs[:top_k]:
        if time.time() > deadline - 3.0:
            break
        
        # Get target area
        _, area_dict = _score_with_areas(config)
        if area_dict:
            target_area = max(area_dict, key=area_dict.get)
        else:
            target_area = None
        
        remaining = deadline - time.time() - 2.0
        if remaining > 3.0:
            try:
                lp, ls = _local_search_targeted(config, time_limit=remaining, target_area=target_area)
                if ls > best_score:
                    best_score = ls
                    best_points = lp
            except Exception:
                pass
    
    # If time remains, try more diverse random starts with local search
    while time.time() < deadline - 10.0:
        try:
            pts = set()
            while len(pts) < 30:
                x = np.random.randint(-30, 31)
                y = np.random.randint(-30, 31)
                pts.add((x, y))
            config = list(pts)
            score = _score_fast(config)
            
            if score > best_score:
                best_score = score
                best_points = config
            
            remaining = deadline - time.time() - 3.0
            if remaining > 5.0:
                _, area_dict = _score_with_areas(config)
                target_area = max(area_dict, key=area_dict.get) if area_dict else None
                lp, ls = _local_search_targeted(config, time_limit=remaining, target_area=target_area)
                if ls > best_score:
                    best_score = ls
                    best_points = lp
        except Exception:
            break
    
    return best_points

def construct():
    """Main entry point: find 30 points maximizing max area multiplicity."""
    start_time = time.time()
    
    try:
        result = _try_all_configs()
    except Exception:
        # Fallback to simple grid
        result = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    
    # Verify validity
    pts = set(result)
    if len(pts) != 30:
        result = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    for (x, y) in result:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            result = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
            break
    
    return result
# EVOLVE-BLOCK-END
