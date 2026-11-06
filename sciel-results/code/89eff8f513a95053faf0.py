import numpy as np
import time
import random
from collections import Counter

# Pre-compute triple indices for 30 points
def _get_triple_indices(n=30):
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    return i_idx[mask], j_idx[mask], k_idx[mask]

_TRIPLE_I, _TRIPLE_J, _TRIPLE_K = _get_triple_indices(30)
_N_TRIPLES = len(_TRIPLE_I)

def _vectorized_compute_areas(points):
    """
    Compute absolute doubled areas for all triples of 30 points.
    points: numpy array of shape (30, 2)
    Returns: numpy array of shape (4500,) with absolute determinants.
    """
    pi = points[_TRIPLE_I]
    pj = points[_TRIPLE_J]
    pk = points[_TRIPLE_K]
    dx_ji = pj[:, 0] - pi[:, 0]
    dy_ji = pj[:, 1] - pi[:, 1]
    dx_ki = pk[:, 0] - pi[:, 0]
    dy_ki = pk[:, 1] - pi[:, 1]
    dets = dx_ji * dy_ki - dy_ji * dx_ki
    return np.abs(dets)

def _get_max_multiplicity(areas):
    """
    Given an array of areas, return (max_count, most_frequent_area).
    Ignores zero areas.
    """
    if len(areas) == 0:
        return 0, 0
    non_zero = areas[areas > 0]
    if len(non_zero) == 0:
        return 0, 0
    unique_areas, counts = np.unique(non_zero, return_counts=True)
    max_idx = np.argmax(counts)
    return counts[max_idx], unique_areas[max_idx]

def _count_area(areas, target):
    """Count how many times target area appears in areas."""
    return np.sum(areas == target)

def _random_points_in_box(n=30, box=30):
    """Generate n distinct random points in [-box, box]^2."""
    points = set()
    while len(points) < n:
        x = random.randint(-box, box)
        y = random.randint(-box, box)
        points.add((x, y))
    return np.array(list(points), dtype=np.int32)

def _grid_points(rows, cols, offset_x=0, offset_y=0):
    """Generate a grid of points."""
    pts = []
    for i in range(rows):
        for j in range(cols):
            x = i * 2 + offset_x
            y = j * 2 + offset_y
            pts.append((x, y))
    return np.array(pts[:30], dtype=np.int32)

def _structured_points():
    """Generate structured points that might have good area properties."""
    patterns = []
    patterns.append(_grid_points(6, 5, -5, -4))
    patterns.append(_grid_points(5, 6, -4, -5))
    patterns.append(_random_points_in_box())
    pts = []
    for i in range(7):
        for j in range(5):
            if len(pts) < 30:
                pts.append((i * 3 - 9, j * 3 - 6))
    patterns.append(np.array(pts, dtype=np.int32))
    return patterns

def _incremental_move(areas, points, idx, new_point):
    """
    Update the areas array after moving point idx to new_point.
    Returns the updated areas array (modified in place for speed).
    """
    # Find all triples involving idx
    mask_i = _TRIPLE_I == idx
    mask_j = _TRIPLE_J == idx
    mask_k = _TRIPLE_K == idx
    mask = mask_i | mask_j | mask_k
    
    # Recompute only those triples
    if np.any(mask):
        ti = _TRIPLE_I[mask]
        tj = _TRIPLE_J[mask]
        tk = _TRIPLE_K[mask]
        
        pi = points[ti]
        pj = points[tj]
        pk = points[tk]
        
        dx_ji = pj[:, 0] - pi[:, 0]
        dy_ji = pj[:, 1] - pi[:, 1]
        dx_ki = pk[:, 0] - pi[:, 0]
        dy_ki = pk[:, 1] - pi[:, 1]
        
        new_dets = dx_ji * dy_ki - dy_ji * dx_ki
        areas[mask] = np.abs(new_dets)
    
    return areas

def _find_snapping_points(points, idx, target_area, occupied_set):
    """
    Find candidate positions for point idx such that it forms a triangle
    with area target_area with two other existing points.
    Returns a list of candidate (x, y) tuples.
    """
    candidates = []
    if target_area == 0:
        return candidates
        
    n = len(points)
    # We want to find p_i such that area(p_i, p_j, p_k) == target_area
    # We pick random pairs (j, k) and solve for p_i.
    # To be efficient, we only try a few random pairs per call.
    
    # Pick 2 random other points
    others = [i for i in range(n) if i != idx]
    if len(others) < 2:
        return candidates
    
    # Try a few random pairs
    for _ in range(5):
        j = random.choice(others)
        k = random.choice(others)
        if j == k:
            continue
            
        x_j, y_j = int(points[j, 0]), int(points[j, 1])
        x_k, y_k = int(points[k, 0]), int(points[k, 1])
        
        dx = x_k - x_j
        dy = y_k - y_j
        
        # Determinant equation: (x - x_j)*dy - (y - y_j)*dx = +/- target_area
        # dy*x - dx*y = +/- target_area + dy*x_j - dx*y_j
        # dy*x - dx*y = C
        
        # We iterate x in [-30, 30]
        # y = (dy*x - C) / dx
        # If dx == 0, we handle separately.
        
        for sign in [1, -1]:
            C = sign * target_area + dy * x_j - dx * y_j
            
            if dx == 0:
                # -dy * y = C => y = -C / dy
                if dy != 0:
                    if C % dy == 0:
                        y_val = -C // dy
                        # x can be anything? No, the line is vertical x = x_j? 
                        # Wait, if dx=0, the line jk is vertical. 
                        # The locus of points with fixed area is two lines parallel to jk.
                        # If jk is vertical, the lines are vertical.
                        # Equation: (x-x_j)*dy = C => x = x_j + C/dy.
                        if C % dy == 0:
                            x_val = x_j + C // dy
                            if -30 <= x_val <= 30:
                                # y can be any integer? 
                                # No, the determinant is (x-x_j)*dy - (y-y_j)*0 = (x-x_j)*dy.
                                # So y does not affect the area if dx=0? 
                                # Det = (x-x_j)(y_k-y_j) - (y-y_j)(x_k-x_j).
                                # If x_k=x_j, dx=0. Det = (x-x_j)dy.
                                # So for ANY y, the area is the same? 
                                # Yes, if the base is vertical, moving the third point vertically 
                                # along the line parallel to the base keeps the height (horizontal dist) constant.
                                # So we can pick any y in [-30, 30].
                                # Let's pick a few random y's or just one.
                                # Actually, we want to move the point to a specific spot.
                                # Any y works for the area constraint. 
                                # We should pick a y that is not occupied.
                                for y_try in [random.randint(-30, 30) for _ in range(3)]:
                                    pt = (x_val, y_try)
                                    if pt not in occupied_set:
                                        candidates.append(pt)
            else:
                # dy*x - dx*y = C
                # dx*y = dy*x - C
                # y = (dy*x - C) / dx
                for x_val in range(-30, 31):
                    num = dy * x_val - C
                    if num % dx == 0:
                        y_val = num // dx
                        if -30 <= y_val <= 30:
                            pt = (x_val, y_val)
                            if pt not in occupied_set:
                                candidates.append(pt)
    
    return candidates

def _sa_search_dynamic(points, deadline, initial_temp=3.0):
    """
    Phase 1: SA with dynamic target to find a good area and config.
    """
    curr_points = points.copy()
    curr_occupied = set(map(tuple, curr_points.tolist()))
    curr_areas = _vectorized_compute_areas(curr_points)
    curr_max_mult, curr_max_area = _get_max_multiplicity(curr_areas)
    target_area = int(curr_max_area)
    curr_target_count = _count_area(curr_areas, target_area)
    
    best_score = max(curr_target_count, curr_max_mult)
    best_points = curr_points.copy()
    best_area = target_area if curr_target_count >= curr_max_mult else curr_max_area
    
    temp = initial_temp
    cooling = 0.9999
    min_temp = 0.01
    no_improve_count = 0
    max_no_improve = 5000
    steps = 0
    last_recompute = 0
    recompute_interval = 2000
    
    while time.time() < deadline:
        steps += 1
        idx = random.randint(0, 29)
        old_x, old_y = int(curr_points[idx, 0]), int(curr_points[idx, 1])
        
        # Random move
        r = random.random()
        if r < 0.4:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        else:
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
            new_x = max(-30, min(30, new_x))
            new_y = max(-30, min(30, new_y))
        
        new_point = (new_x, new_y)
        if new_point == (old_x, old_y) or new_point in curr_occupied:
            continue
            
        curr_points[idx] = [new_x, new_y]
        curr_occupied.remove((old_x, old_y))
        curr_occupied.add(new_point)
        _incremental_move(curr_areas, curr_points, idx, new_point)
        
        if steps - last_recompute >= recompute_interval:
            curr_areas = _vectorized_compute_areas(curr_points)
            last_recompute = steps
            
        new_max_mult, new_max_area = _get_max_multiplicity(curr_areas)
        
        if new_max_area != target_area:
            target_area = int(new_max_area)
            
        new_target_count = _count_area(curr_areas, target_area)
        new_score = max(new_target_count, new_max_mult)
        curr_score = max(curr_target_count, curr_max_mult)
        
        improved = False
        if new_score > curr_score:
            improved = True
            no_improve_count = 0
        elif new_score == curr_score:
            if new_target_count > curr_target_count:
                improved = True
                no_improve_count = 0
            elif random.random() < 0.5:
                improved = True
        else:
            delta = curr_score - new_score
            if random.random() < np.exp(-delta / temp):
                improved = True
        
        if improved:
            curr_target_count = new_target_count
            curr_max_mult = new_max_mult
            curr_max_area = new_max_area
            curr_score = new_score
            if curr_score > best_score:
                best_score = curr_score
                best_points = curr_points.copy()
                best_area = target_area if curr_target_count >= curr_max_mult else curr_max_area
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            curr_points[idx] = [old_x, old_y]
            curr_occupied.remove(new_point)
            curr_occupied.add((old_x, old_y))
            _incremental_move(curr_areas, curr_points, idx, (old_x, old_y))
            no_improve_count += 1
            
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
        if no_improve_count > max_no_improve:
            temp = max(temp, 1.5)
            no_improve_count = 0
            
    return best_score, best_points, best_area

def _sa_search_targeted(points, target_area, deadline, initial_temp=2.0):
    """
    Phase 2: SA fixed on a specific target area, using constructive snapping.
    """
    curr_points = points.copy()
    curr_occupied = set(map(tuple, curr_points.tolist()))
    curr_areas = _vectorized_compute_areas(curr_points)
    curr_count = _count_area(curr_areas, target_area)
    
    best_score = curr_count
    best_points = curr_points.copy()
    
    temp = initial_temp
    cooling = 0.99995
    min_temp = 0.005
    no_improve_count = 0
    max_no_improve = 8000
    steps = 0
    last_recompute = 0
    recompute_interval = 2000
    
    while time.time() < deadline:
        steps += 1
        idx = random.randint(0, 29)
        old_x, old_y = int(curr_points[idx, 0]), int(curr_points[idx, 1])
        
        # Decide move type
        r = random.random()
        if r < 0.6:
            # Constructive Snapping
            candidates = _find_snapping_points(curr_points, idx, target_area, curr_occupied)
            if not candidates:
                # Fallback to local random if no snap found
                new_x = old_x + random.randint(-2, 2)
                new_y = old_y + random.randint(-2, 2)
                new_x = max(-30, min(30, new_x))
                new_y = max(-30, min(30, new_y))
                new_point = (new_x, new_y)
                if new_point == (old_x, old_y) or new_point in curr_occupied:
                    continue
            else:
                new_point = random.choice(candidates)
        elif r < 0.9:
            # Local Perturbation
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
            new_x = max(-30, min(30, new_x))
            new_y = max(-30, min(30, new_y))
            new_point = (new_x, new_y)
            if new_point == (old_x, old_y) or new_point in curr_occupied:
                continue
        else:
            # Random Jump
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_point = (new_x, new_y)
            if new_point == (old_x, old_y) or new_point in curr_occupied:
                continue

        curr_points[idx] = [new_point[0], new_point[1]]
        curr_occupied.remove((old_x, old_y))
        curr_occupied.add(new_point)
        _incremental_move(curr_areas, curr_points, idx, new_point)
        
        if steps - last_recompute >= recompute_interval:
            curr_areas = _vectorized_compute_areas(curr_points)
            last_recompute = steps
            
        new_count = _count_area(curr_areas, target_area)
        
        # Acceptance: Always accept if count increases.
        # If count same, accept with prob 0.5 (to shuffle).
        # If count decreases, accept with prob exp(-delta/temp).
        if new_count > curr_count:
            improved = True
            no_improve_count = 0
        elif new_count == curr_count:
            if random.random() < 0.5:
                improved = True
            else:
                improved = False
        else:
            delta = curr_count - new_count
            if random.random() < np.exp(-delta / temp):
                improved = True
            else:
                improved = False
                
        if improved:
            curr_count = new_count
            if curr_count > best_score:
                best_score = curr_count
                best_points = curr_points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            curr_points[idx] = [old_x, old_y]
            curr_occupied.remove(new_point)
            curr_occupied.add((old_x, old_y))
            _incremental_move(curr_areas, curr_points, idx, (old_x, old_y))
            no_improve_count += 1
            
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
        if no_improve_count > max_no_improve:
            temp = max(temp, 1.0)
            no_improve_count = 0
            
    return best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Hard cap
    
    # Phase 0: Init
    initial_configs = _structured_points()
    best_overall_score = 0
    best_overall_points = None
    best_overall_area = 0
    
    for config in initial_configs:
        areas = _vectorized_compute_areas(config)
        score, area = _get_max_multiplicity(areas)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = config.copy()
            best_overall_area = area
            
    if best_overall_points is None:
        best_overall_points = _random_points_in_box()
        areas = _vectorized_compute_areas(best_overall_points)
        best_overall_score, best_overall_area = _get_max_multiplicity(areas)

    # Phase 1: Dynamic SA (15s)
    phase1_deadline = start_time + 15
    if time.time() < phase1_deadline:
        score, points, area = _sa_search_dynamic(best_overall_points, phase1_deadline, initial_temp=3.0)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = points
            best_overall_area = area

    # Phase 2: Targeted SA with Snapping (Remaining time - 2s)
    phase2_deadline = deadline - 2
    if time.time() < phase2_deadline:
        # Use the best area found so far as the target
        target_a = int(best_overall_area)
        score, points = _sa_search_targeted(best_overall_points, target_a, phase2_deadline, initial_temp=2.0)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = points

    if best_overall_points is None:
        best_overall_points = _random_points_in_box()

    return [(int(x), int(y)) for x, y in best_overall_points.tolist()]
