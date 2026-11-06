import time
import numpy as np
import random
from collections import Counter

# Hard wall-clock budget for the whole construct() call (safety margin under 90s).
_DEADLINE = 85.0
_MAX_AREA = 4000  # Safe upper bound for |det| in [-30,30]^2
_N = 30


def _get_triple_indices(n):
    """Precompute indices for all unique triples (i, j, k) with i < j < k."""
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    return i[mask], j[mask], k[mask]

# Precompute triple indices for n=30
_I, _J, _K = _get_triple_indices(_N)

def _vectorized_compute_areas(pts):
    """
    Compute the absolute determinant (doubled area) for all triples.
    Returns an array of areas.
    """
    p = np.asarray(pts, dtype=np.int64)
    xi, yi = p[_I, 0], p[_I, 1]
    xj, yj = p[_J, 0], p[_J, 1]
    xk, yk = p[_K, 0], p[_K, 1]
    det = (xi * (yj - yk)) + (xj * (yk - yi)) + (xk * (yi - yj))
    return np.abs(det)

def _get_max_multiplicity(areas):
    """
    Given an array of areas, return the maximum multiplicity of any positive area.
    """
    pos = areas[areas > 0]
    if pos.size == 0:
        return 0, 0
    vals, counts = np.unique(pos, return_counts=True)
    max_idx = np.argmax(counts)
    return int(counts[max_idx]), int(vals[max_idx])

def _count_area(areas, target):
    """
    Count occurrences of a specific target area in the array.
    """
    if target <= 0:
        return 0
    return int(np.sum(areas == target))

def _random_points_in_box(n=30):
    """Generate n distinct random integer points in [-30, 30]^2."""
    pts = set()
    while len(pts) < n:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        pts.add((x, y))
    return list(pts)

def _grid_points(rows, cols, cx=0, cy=0, sx=3, sy=3):
    """Generate a grid of points."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = cx + (c - cols // 2) * sx
            y = cy + (r - rows // 2) * sy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    return pts[:30]

def _structured_points():
    """Generate a set of structured points known to have high multiplicities."""
    # A 6x5 grid with spacing 3
    return _grid_points(6, 5, cx=0, cy=0, sx=3, sy=3)

def _incremental_move(pts, idx, new_pos):
    """
    Apply a move to pts at index idx with new_pos.
    Returns the new pts list.
    """
    new_pts = pts.copy()
    new_pts[idx] = new_pos
    return new_pts

def _sa_search(pts, deadline, target_area=None):
    """
    Simulated Annealing search to maximize the multiplicity of a single area.
    If target_area is None, it dynamically tracks the current best area.
    """
    pts = [tuple(p) for p in pts]
    n = len(pts)
    
    # Initial evaluation
    areas = _vectorized_compute_areas(pts)
    cur_score, cur_target = _get_max_multiplicity(areas)
    
    if target_area is None:
        target_area = cur_target
    
    best_score = cur_score
    best_pts = pts.copy()
    
    # SA Parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.1
    
    start_time = time.time()
    
    while time.time() < deadline and T > min_T:
        # Select a random point to move
        idx = random.randint(0, n - 1)
        old_pos = pts[idx]
        
        # Generate a random new position
        # Bias towards small moves initially? Or just random in box?
        # Random in box is good for escaping local minima.
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        new_pos = (new_x, new_y)
        
        # Check validity (distinctness)
        if new_pos == old_pos:
            continue
        if new_pos in pts:
            continue
            
        # Evaluate new state
        # To speed up, we could update only affected triples, but vectorized re-computation
        # of 4495 triples is ~0.5ms. 1000 iterations is 0.5s.
        # 20000 iterations is 10s. This is feasible.
        
        new_pts = _incremental_move(pts, idx, new_pos)
        new_areas = _vectorized_compute_areas(new_pts)
        
        # Determine score
        # Objective: Maximize count of target_area.
        # If we dynamically update target, we might chase a moving target.
        # Strategy: Use the current global max as the target? Or a fixed target?
        # The inspiration says "dynamic targeting".
        # Let's define score as: count(new_target) where new_target is the max area in new_areas?
        # No, that's noisy.
        # Better: Score = count(target_area).
        # If we find a new area with higher count, we can update target_area.
        
        new_score, new_max_area = _get_max_multiplicity(new_areas)
        
        # Update target if the new configuration has a strictly better dominant area
        # or if the current target is no longer the dominant one and the new one is better.
        # Simple heuristic: If new_score > cur_score, accept and update target.
        # If new_score == cur_score, accept with probability 1.
        # If new_score < cur_score, accept with probability exp((new_score - cur_score)/T).
        
        # However, we want to maximize the *largest* multiplicity.
        # If we fix target_area, we might miss a better area.
        # Let's allow target_area to update if we find a configuration with a higher max multiplicity.
        
        delta_score = new_score - cur_score
        
        if delta_score > 0:
            # Always accept improvement
            pts = new_pts
            cur_score = new_score
            cur_target = new_max_area # Update target to the new dominant area
            target_area = cur_target
            
            if cur_score > best_score:
                best_score = cur_score
                best_pts = pts.copy()
        else:
            # Accept worse moves with probability
            if delta_score == 0:
                # Neutral move, accept to explore
                pts = new_pts
                # Keep target_area same? Or update if new_max_area is also dominant?
                # If counts are same, it's a tie. Keep current target to avoid oscillation.
                # But if new_max_area is different and has same count, maybe switch?
                # Let's stick to current target for stability.
            else:
                # Worse move
                prob = np.exp(delta_score / T)
                if random.random() < prob:
                    pts = new_pts
                    cur_score = new_score
                    # Do not update target_area for worse moves to maintain focus?
                    # Or update if it's a "promising" worse move?
                    # Standard SA keeps the state but objective is fixed or dynamic.
                    # If objective is dynamic (max multiplicity), then cur_score is the new max.
                    # So we should update cur_target to new_max_area.
                    cur_target = new_max_area
                    target_area = cur_target
        
        # Cool down
        T *= cooling_rate
        
    return best_pts, best_score

def _polish(pts, deadline):
    """
    Greedy local search to polish the result.
    Tries to move each point to any position that increases the score.
    """
    pts = [tuple(p) for p in pts]
    n = len(pts)
    
    areas = _vectorized_compute_areas(pts)
    cur_score, cur_target = _get_max_multiplicity(areas)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n):
            old_pos = pts[idx]
            best_new_pos = None
            best_new_score = cur_score
            
            # Try all possible positions? 61x61 = 3721.
            # For each, compute score.
            # 30 points * 3721 positions * 0.5ms eval = 5.5s per pass.
            # Too slow for multiple passes.
            
            # Sample positions instead?
            # Or just check nearby positions?
            
            # Let's try a random subset of positions to speed up.
            candidates = []
            for _ in range(100):
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) != old_pos and (x, y) not in pts:
                    candidates.append((x, y))
            
            for cand in candidates:
                new_pts = _incremental_move(pts, idx, cand)
                new_areas = _vectorized_compute_areas(new_pts)
                new_score, new_target = _get_max_multiplicity(new_areas)
                
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_pos = cand
                    # Update target if improved
                    cur_target = new_target
            
            if best_new_pos is not None:
                pts[idx] = best_new_pos
                cur_score = best_new_score
                improved = True
    
    return pts, cur_score

def construct():
    start = time.time()
    deadline = start + _DEADLINE
    random.seed(42)

    # Phase 1: Generate diverse seeds
    seeds = [
        _structured_points(),
        _random_points_in_box(),
        _random_points_in_box(),
        _grid_points(5, 6, cx=0, cy=0, sx=3, sy=3),
        _grid_points(10, 3, cx=0, cy=0, sx=5, sy=3),
    ]
    
    # Add some sheared grids
    def _sheared_grid(rows, cols, cx=0, cy=0, sx=3, sy=3, shear=1):
        pts = []
        for r in range(rows):
            for c in range(cols):
                x = cx + (c - cols // 2) * sx + r * shear
                y = cy + (r - rows // 2) * sy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        return pts[:30]
    
    seeds.append(_sheared_grid(6, 5, shear=1))
    seeds.append(_sheared_grid(6, 5, shear=-1))

    best_pts = None
    best_val = 0
    
    # Allocate time for SA and Polish
    # 70% for SA, 30% for Polish
    sa_deadline = start + (_DEADLINE * 0.7)
    
    # Run SA on each seed, allocating remaining time
    n_seeds = len(seeds)
    per_seed_time = (sa_deadline - time.time()) / n_seeds
    
    for seed in seeds:
        if time.time() >= sa_deadline:
            break
            
        # Ensure seed is valid
        valid_seed = []
        seen = set()
        for p in seed:
            if -30 <= p[0] <= 30 and -30 <= p[1] <= 30 and p not in seen:
                seen.add(p)
                valid_seed.append(p)
        if len(valid_seed) < 30:
            # Pad with random points
            while len(valid_seed) < 30:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in seen:
                    seen.add((x, y))
                    valid_seed.append((x, y))
        
        seed_deadline = time.time() + per_seed_time
        pts, val = _sa_search(valid_seed, seed_deadline)
        
        if val > best_val:
            best_val = val
            best_pts = pts

    # Phase 2: Polish the best result
    if best_pts and time.time() < deadline - 1:
        polish_deadline = deadline - 1
        best_pts, best_val = _polish(best_pts, polish_deadline)
        
    # Final validation
    if best_pts is None:
        best_pts = _structured_points()
        best_val = _get_max_multiplicity(_vectorized_compute_areas(best_pts))[0]
        
    # Ensure exactly 30 distinct points in range
    seen = set()
    out = []
    for p in best_pts:
        x, y = p
        if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in seen:
            seen.add((x, y))
            out.append((x, y))
            
    if len(out) < 30:
        for x in range(-30, 31):
            for y in range(-30, 31):
                if len(out) >= 30:
                    break
                if (x, y) not in seen:
                    seen.add((x, y))
                    out.append((x, y))
            if len(out) >= 30:
                break
                
    return out[:30]
