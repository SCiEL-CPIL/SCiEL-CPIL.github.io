import time
import random
from collections import Counter
import numpy as np

def _build_counter(points):
    """Build area counter for all non-zero doubled areas."""
    n = len(points)
    area_counts = Counter()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx1 = x2 - x1
            dy1 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = points[k]
                dx2 = x3 - x1
                dy2 = y3 - y1
                area = abs(dx1 * dy2 - dy1 * dx2)
                if area > 0:
                    area_counts[area] += 1
    return area_counts

def _max_of_counter(area_counts):
    """Return max multiplicity in counter."""
    if not area_counts:
        return 0
    return max(area_counts.values())

def _calculate_score(points):
    """Calculate the maximum multiplicity of any non-zero doubled area."""
    area_counts = _build_counter(points)
    return _max_of_counter(area_counts)

def _get_nearby_positions(point, occupied_set, radius=4):
    """Get valid positions near a given point."""
    x, y = point
    candidates = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if -30 <= nx <= 30 and -30 <= ny <= 30:
                p = (nx, ny)
                if p not in occupied_set:
                    candidates.append(p)
    return candidates

def _evaluate_move_fast(points, idx, new_p, area_counts):
    """
    Fast evaluation of moving point idx to new_p using NumPy.
    Returns (delta_score, new_area_counts) or (None, None) if invalid.
    """
    old_p = points[idx]
    if new_p == old_p:
        return None, None

    n = len(points)
    # Gather all other points as numpy array
    others_idx = [i for i in range(n) if i != idx]
    others = np.array([points[i] for i in others_idx], dtype=np.int32)
    m = len(others)  # 29

    old_x, old_y = old_p
    new_x, new_y = new_p

    # Precompute vectors from old point to all others
    # Shape: (29, 2)
    vecs_old = others - np.array([old_x, old_y], dtype=np.int32)
    # Precompute vectors from new point to all others
    vecs_new = others - np.array([new_x, new_y], dtype=np.int32)

    # For each pair (i, j) with i < j among others, compute area
    # We need C(29,2) = 406 pairs
    # Use broadcasting: for pair (a, b), area = |vx_a * vy_b - vy_a * vx_b|
    
    # Build all pairs
    pair_i = np.repeat(np.arange(m), m - 1)
    pair_j = np.concatenate([np.arange(i + 1, m) for i in range(m)])
    
    # Old areas
    va = vecs_old[pair_i]
    vb = vecs_old[pair_j]
    old_areas = np.abs(va[:, 0] * vb[:, 1] - va[:, 1] * vb[:, 0])
    
    # New areas
    va_n = vecs_new[pair_i]
    vb_n = vecs_new[pair_j]
    new_areas = np.abs(va_n[:, 0] * vb_n[:, 1] - va_n[:, 1] * vb_n[:, 0])
    
    # Build deltas
    deltas = {}
    for a in old_areas:
        if a > 0:
            deltas[a] = deltas.get(a, 0) - 1
    for a in new_areas:
        if a > 0:
            deltas[a] = deltas.get(a, 0) + 1
    
    # Apply deltas to get new counts
    new_counts = dict(area_counts)
    for area, delta in deltas.items():
        if delta == 0:
            continue
        cur = new_counts.get(area, 0)
        if cur + delta < 0:
            return None, None
        new_val = cur + delta
        if new_val == 0:
            del new_counts[area]
        else:
            new_counts[area] = new_val

    new_score = _max_of_counter(new_counts)
    old_score = _max_of_counter(area_counts)
    delta_score = new_score - old_score

    return delta_score, new_counts

def _local_search(initial_points, time_limit=55.0, seed=None):
    """
    Perform hill climbing to maximize the score using fast incremental updates.
    """
    if seed is not None:
        random.seed(seed)

    start_time = time.time()
    best_points = [p for p in initial_points]
    best_score = _calculate_score(best_points)
    best_counts = _build_counter(best_points)

    current_points = [p for p in initial_points]
    current_score = best_score
    current_counts = dict(best_counts)

    occupied_set = set(current_points)

    stuck_count = 0
    max_stuck = 200
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]

    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = current_points[idx]

        # Get nearby positions (prefer closer ones)
        nearby = _get_nearby_positions(old_p, occupied_set, radius=5)

        if not nearby:
            empty = [c for c in all_coords if c not in occupied_set]
            if not empty:
                break
            new_p = random.choice(empty)
        else:
            # Weight toward closer positions
            weights = [1.0 / (1.0 + sum((a - b) ** 2 for a, b in zip(old_p, p))) for p in nearby]
            new_p = random.choices(nearby, weights=weights, k=1)[0]

        # Evaluate move
        delta_score, new_counts = _evaluate_move_fast(current_points, idx, new_p, current_counts)

        if delta_score is None:
            stuck_count += 1
            if stuck_count > max_stuck:
                current_points = [p for p in best_points]
                occupied_set = set(current_points)
                current_score = best_score
                current_counts = dict(best_counts)
                stuck_count = 0
                # Perturbation to escape local optimum
                if random.random() < 0.6:
                    for _ in range(random.randint(2, 4)):
                        ridx = random.randint(0, 29)
                        old_rp = current_points[ridx]
                        nearby_r = _get_nearby_positions(old_rp, occupied_set, radius=6)
                        if nearby_r:
                            new_rp = random.choice(nearby_r)
                            current_points[ridx] = new_rp
                            occupied_set.remove(old_rp)
                            occupied_set.add(new_rp)
                    current_counts = _build_counter(current_points)
                    current_score = _max_of_counter(current_counts)
            continue

        # Apply move
        current_points[idx] = new_p
        occupied_set.remove(old_p)
        occupied_set.add(new_p)
        current_counts = new_counts
        current_score += delta_score

        if delta_score > 0:
            stuck_count = 0
            if current_score > best_score:
                best_score = current_score
                best_points = [p for p in current_points]
                best_counts = dict(current_counts)
        else:
            # Revert move
            current_points[idx] = old_p
            occupied_set.remove(new_p)
            occupied_set.add(old_p)
            current_counts = _build_counter(current_points)
            current_score = _max_of_counter(current_counts)
            stuck_count += 1
            if stuck_count > max_stuck:
                current_points = [p for p in best_points]
                occupied_set = set(current_points)
                current_score = best_score
                current_counts = dict(best_counts)
                stuck_count = 0
                if random.random() < 0.6:
                    for _ in range(random.randint(2, 4)):
                        ridx = random.randint(0, 29)
                        old_rp = current_points[ridx]
                        nearby_r = _get_nearby_positions(old_rp, occupied_set, radius=6)
                        if nearby_r:
                            new_rp = random.choice(nearby_r)
                            current_points[ridx] = new_rp
                            occupied_set.remove(old_rp)
                            occupied_set.add(new_rp)
                    current_counts = _build_counter(current_points)
                    current_score = _max_of_counter(current_counts)

    return best_points, best_score

def _hexagonal_lattice(center_x, center_y, rows, cols):
    """Generate a hexagonal lattice of points."""
    points = []
    for i in range(rows):
        for j in range(cols):
            x = center_x + j
            y = center_y + i
            if i % 2 == 1:
                x += 1
            points.append((x, y))
    return points

def _triangular_lattice(center_x, center_y, n):
    """Generate a triangular lattice of n points (rows 1,2,3,...)."""
    points = []
    row = 0
    count = 0
    while count < n:
        for j in range(row + 1):
            if count >= n:
                break
            x = center_x + j - row // 2
            y = center_y + row
            points.append((x, y))
            count += 1
        row += 1
    return points

def _rectangular_grid(cx, cy, w, h):
    """Generate a w x h rectangular grid centered at (cx, cy)."""
    points = []
    for i in range(h):
        for j in range(w):
            x = cx + j - w // 2
            y = cy + i - h // 2
            points.append((x, y))
    return points

def construct():
    # Generate diverse structured initial configurations
    candidates = []
    
    # Hexagonal lattices at various positions
    for cx in [-3, 0, 3]:
        for cy in [-3, 0, 3]:
            candidates.append(_hexagonal_lattice(cx, cy, 5, 6))
    
    # Triangular lattice
    candidates.append(_triangular_lattice(0, 0, 30))
    
    # Rectangular grids
    candidates.append(_rectangular_grid(0, 0, 5, 6))
    candidates.append(_rectangular_grid(0, 0, 6, 5))
    candidates.append(_rectangular_grid(0, 0, 3, 10))
    candidates.append(_rectangular_grid(0, 0, 10, 3))
    candidates.append(_rectangular_grid(0, 0, 2, 15))
    candidates.append(_rectangular_grid(0, 0, 15, 2))
    
    # Random points
    random.seed(42)
    all_pts = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    candidates.append(random.sample(all_pts, 30))
    
    # Score all candidates and pick top few
    scored = []
    for c in candidates:
        if len(c) == 30 and len(set(c)) == 30:
            s = _calculate_score(c)
            scored.append((s, c))
    
    scored.sort(key=lambda x: -x[0])
    
    # Run local search from top 3 initial configurations with time budgeting
    total_time_budget = 58.0  # Leave 32s safety margin for 90s kill
    time_per_start = total_time_budget / 3.0
    
    best_overall_points = scored[0][1]
    best_overall_score = scored[0][0]
    
    for i, (init_score, init_points) in enumerate(scored[:3]):
        if i > 0:
            time_left = total_time_budget - i * time_per_start
            if time_left < 10.0:
                break
        
        t_limit = min(time_per_start, time_left - 5.0)
        if t_limit < 5.0:
            break
            
        seed_val = 42 + i * 1000
        final_points, final_score = _local_search(init_points, time_limit=t_limit, seed=seed_val)
        
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall_points = final_points
    
    return best_overall_points
