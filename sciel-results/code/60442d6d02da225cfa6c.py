"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

def _compute_areas_fast(points):
    """Compute all doubled triangle areas (absolute determinants) using vectorized numpy."""
    n = len(points)
    if n < 3:
        return []
    pts = np.array(points, dtype=np.int64)
    x1, y1 = pts[:, 0], pts[:, 1]
    
    indices = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(indices, indices, indices, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    x1_f = x1[i_flat]
    y1_f = y1[i_flat]
    x2_f = x1[j_flat]
    y2_f = y1[j_flat]
    x3_f = x1[k_flat]
    y3_f = y1[k_flat]
    
    dets = np.abs(x1_f * (y2_f - y3_f) - x2_f * (y1_f - y3_f) + x3_f * (y1_f - y2_f))
    dets = dets[dets > 0]
    return dets.tolist()

def _score_config(points):
    """Return the maximum multiplicity of any positive doubled area."""
    areas = _compute_areas_fast(points)
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _score_target_fast(points, target_area):
    """Count how many triples have exactly target_area as doubled area using vectorized numpy."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    x1, y1 = pts[:, 0], pts[:, 1]
    
    indices = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(indices, indices, indices, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    x1_f = x1[i_flat]
    y1_f = y1[i_flat]
    x2_f = x1[j_flat]
    y2_f = y1[j_flat]
    x3_f = x1[k_flat]
    y3_f = y1[k_flat]
    
    dets = np.abs(x1_f * (y2_f - y3_f) - x2_f * (y1_f - y3_f) + x3_f * (y1_f - y2_f))
    count = np.sum(dets == target_area)
    return int(count)

def _guided_moves(points, idx, target_area, rng, max_candidates=50):
    """
    Generate candidate positions for points[idx] such that triangles
    involving points[idx] and two other fixed points have area target_area.
    Uses the linear equation: for fixed points p_j, p_k, find p_i such that
    |det(p_i, p_j, p_k)| = target_area.
    """
    n = len(points)
    candidates = set()
    px, py = points[idx]
    
    # For each pair (j, k) with j != idx, k != idx, j != k
    # The determinant condition: |x_i*(y_j - y_k) - x_j*(y_i - y_k) + x_k*(y_i - y_y)| = target_area
    # This is linear in x_i, y_i: A*x_i + B*y_i = C or A*x_i + B*y_i = -C
    for j in range(n):
        if j == idx:
            continue
        for k in range(j + 1, n):
            if k == idx:
                continue
            xj, yj = points[j]
            xk, yk = points[k]
            
            # det = x_i*(yj - yk) - xj*(y_i - yk) + xk*(y_i - yj)
            # = x_i*(yj - yk) + y_i*(xk - xj) + (-xj*yk + xk*yj)
            A = yj - yk
            B = xk - xj
            C = -xj * yk + xk * yj
            
            # A*x_i + B*y_i + C = ±target_area
            # A*x_i + B*y_i = -C ± target_area
            for sign in [1, -1]:
                rhs = -C + sign * target_area
                # Find integer solutions to A*x + B*y = rhs in [-30, 30]^2
                # Try a few values of x and solve for y
                for x in range(-30, 31):
                    val = rhs - A * x
                    if B == 0:
                        if val == 0:
                            # Any y works, pick a few
                            for y in range(-30, 31, 7):
                                candidates.add((x, y))
                        continue
                    if val % B == 0:
                        y = val // B
                        if -30 <= y <= 30:
                            candidates.add((x, y))
                    # Also try nearby y values for robustness
                    if B != 0:
                        y_float = val / B
                        y_round = int(round(y_float))
                        for dy in range(-2, 3):
                            y_test = y_round + dy
                            if -30 <= y_test <= 30:
                                candidates.add((x, y_test))
    
    # Convert to list, remove current position
    result = []
    for (cx, cy) in candidates:
        if (cx, cy) != (px, py):
            result.append((cx, cy))
    
    # Shuffle and limit
    rng.shuffle(result)
    return result[:max_candidates]

def _local_search_anneal(points, target_area, max_iter=20000, time_limit=25):
    """Local search with annealing to maximize count of triples with target_area."""
    start_time = time.time()
    rng = random.Random()
    
    best_points = list(points)
    best_count = _score_target_fast(best_points, target_area)
    
    current_points = list(best_points)
    current_count = best_count
    
    temp = 5.0
    cooling = 0.9995
    min_temp = 0.1
    
    n = len(current_points)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point to move
        idx = rng.randint(0, n - 1)
        old_x, old_y = current_points[idx]
        
        # Generate candidate moves: mix of guided and random
        use_guided = rng.random() < 0.6
        if use_guided:
            candidates = _guided_moves(current_points, idx, target_area, rng, max_candidates=20)
            if not candidates:
                candidates = [(rng.randint(-30, 30), rng.randint(-30, 30))]
        else:
            candidates = [(rng.randint(-30, 30), rng.randint(-30, 30))]
        
        # Try each candidate
        moved = False
        for new_x, new_y in candidates:
            # Check distinctness
            is_distinct = True
            for i, (px, py) in enumerate(current_points):
                if i != idx and (px, py) == (new_x, new_y):
                    is_distinct = False
                    break
            
            if not is_distinct:
                continue
            
            # Try the move
            current_points[idx] = (new_x, new_y)
            new_count = _score_target_fast(current_points, target_area)
            
            if new_count > current_count:
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_points = list(current_points)
                moved = True
                break
            elif new_count == current_count:
                # Accept with some probability
                if rng.random() < 0.3:
                    moved = True
                    break
            else:
                # Accept with annealing probability
                delta = current_count - new_count
                if temp > min_temp and rng.random() < np.exp(-delta / max(temp, 0.01)):
                    current_count = new_count
                    moved = True
                    break
            
            # Revert if not accepted
            current_points[idx] = (old_x, old_y)
        
        if not moved:
            current_points[idx] = (old_x, old_y)
        
        # Cool down
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
    
    return best_points, best_count

def _initial_grid_points():
    """Generate initial grid of 30 points."""
    points = []
    for x in range(6):
        for y in range(5):
            points.append((x, y))
    return points

def _initial_lattice_points():
    """Generate initial two-lattice configuration of 30 points."""
    points = []
    for x in range(5):
        for y in range(4):
            points.append((x, y))
    for x in range(5):
        for y in range(2):
            points.append((x, y + 5))
    int_points = []
    seen = set()
    for p in points:
        if p not in seen:
            seen.add(p)
            int_points.append(p)
    while len(int_points) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in seen:
            seen.add((x, y))
            int_points.append((x, y))
    return int_points[:30]

def _initial_small_grid():
    """Small dense grid that may produce many equal areas."""
    points = []
    for x in range(5):
        for y in range(6):
            points.append((x, y))
    return points

def construct():
    deadline = time.time() + 55  # Safety margin before 90s limit
    
    # Try multiple initial configurations
    initial_configs = [
        _initial_grid_points(),
        _initial_lattice_points(),
        _initial_small_grid()
    ]
    
    best_score = 0
    best_points = None
    
    for init_points in initial_configs:
        if time.time() > deadline:
            break
        
        current_score = _score_config(init_points)
        if current_score > best_score:
            best_score = current_score
            best_points = list(init_points)
    
    if best_points is None:
        best_points = initial_configs[0]
        best_score = _score_config(best_points)
    
    # Determine good target areas from initial config
    areas = _compute_areas_fast(best_points)
    if areas:
        counts = Counter(areas)
        target_areas = [a for a, c in counts.most_common(20)]
    else:
        target_areas = list(range(1, 20))
    
    # For each target area, do local search with annealing
    for target in target_areas:
        if time.time() > deadline:
            break
        
        remaining_time = max(1, int(deadline - time.time()) - 5)
        improved_points, improved_count = _local_search_anneal(
            best_points, target, max_iter=30000, time_limit=remaining_time
        )
        
        if time.time() > deadline:
            break
        
        actual_score = _score_config(improved_points)
        if actual_score > best_score:
            best_score = actual_score
            best_points = improved_points
    
    # Final polish: try a few more random restarts if time permits
    if time.time() < deadline - 10:
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            # Perturb best_points randomly
            perturbed = list(best_points)
            for _ in range(5):
                idx = random.randint(0, 29)
                perturbed[idx] = (random.randint(-30, 30), random.randint(-30, 30))
            
            # Ensure distinctness
            seen = set()
            valid_perturbed = []
            for p in perturbed:
                if p not in seen:
                    seen.add(p)
                    valid_perturbed.append(p)
            
            while len(valid_perturbed) < 30:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in seen:
                    seen.add((x, y))
                    valid_perturbed.append((x, y))
            
            perturbed = valid_perturbed[:30]
            
            # Score and possibly improve
            score = _score_config(perturbed)
            if score > best_score:
                best_score = score
                best_points = perturbed
            
            # Quick local search on the perturbed config
            if time.time() < deadline - 10:
                areas2 = _compute_areas_fast(perturbed)
                if areas2:
                    counts2 = Counter(areas2)
                    top_target = counts2.most_common(1)[0][0]
                    improved, _ = _local_search_anneal(perturbed, top_target, max_iter=10000, time_limit=8)
                    improved_score = _score_config(improved)
                    if improved_score > best_score:
                        best_score = improved_score
                        best_points = improved
    
    return best_points
# EVOLVE-BLOCK-END
