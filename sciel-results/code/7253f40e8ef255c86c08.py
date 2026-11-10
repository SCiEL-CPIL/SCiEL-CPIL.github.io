"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _line_key(p1, p2):
    """Compute canonical line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _build_line_map(points):
    """Build map from line key to set of points on that line."""
    line_map = defaultdict(set)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_map[key].add(i)
                line_map[key].add(j)
    return line_map

def _score_from_line_map(line_map):
    """Compute score from line map: count lines with exactly 4 points."""
    score = 0
    for pts in line_map.values():
        if len(pts) == 4:
            score += 1
    return score

def _is_valid(line_map):
    """Check if no line has 5+ points."""
    for pts in line_map.values():
        if len(pts) >= 5:
            return False
    return True

def _evaluate(points):
    """
    Returns (is_valid, score).
    """
    line_map = _build_line_map(points)
    if not _is_valid(line_map):
        return False, 0
    return True, _score_from_line_map(line_map)

def _backtracking_search(points_so_far, line_map, target_n=16, candidate_pool=None, deadline=None):
    """
    Backtracking search that adds points one at a time,
    pruning if adding a point would create a line with 5+ collinear points.
    Returns (best_points, best_score).
    """
    if deadline is None:
        deadline = time.time() + 10
    
    n = len(points_so_far)
    if n == target_n:
        score = _score_from_line_map(line_map)
        return list(points_so_far), score
    
    if time.time() > deadline:
        return None, -1
    
    if candidate_pool is None:
        # Generate candidate pool from grid
        candidate_pool = []
        for x in range(-5, 6):
            for y in range(-5, 6):
                candidate_pool.append((x, y))
        random.shuffle(candidate_pool)
    
    # Try adding each candidate point
    best_points = None
    best_score = -1
    
    for candidate in candidate_pool:
        if time.time() > deadline:
            break
        
        # Check if candidate is already in points_so_far
        if candidate in points_so_far:
            continue
        
        # Check if adding candidate would create a line with 5+ points
        new_line_map = defaultdict(set)
        for key, pts in line_map.items():
            new_line_map[key] = set(pts)
        
        valid = True
        for i in range(len(points_so_far)):
            key = _line_key(points_so_far[i], candidate)
            if key is not None:
                new_line_map[key].add(i)
                new_line_map[key].add(len(points_so_far))
                if len(new_line_map[key]) >= 5:
                    valid = False
                    break
        
        if not valid:
            continue
        
        # Recurse
        new_points = points_so_far + [candidate]
        result_points, result_score = _backtracking_search(
            new_points, new_line_map, target_n, candidate_pool, deadline
        )
        
        if result_points is not None and result_score > best_score:
            best_score = result_score
            best_points = result_points
        
        # Early termination if we found a good score
        if best_score >= 10:
            break
    
    if best_points is None:
        return None, -1
    
    return best_points, best_score

def _simulated_annealing(points, deadline):
    """Simulated annealing to improve a starting configuration."""
    current = [list(p) for p in points]
    current_valid, current_score = _evaluate(current)
    if not current_valid:
        return current, 0
    
    best_points = [list(p) for p in current]
    best_score = current_score
    
    temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Try a random move
        idx = random.randint(0, 15)
        old_x, old_y = current[idx]
        
        if random.random() < 0.8:
            new_x = old_x + random.randint(-10, 10)
            new_y = old_y + random.randint(-10, 10)
        else:
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        if any(p[0] == new_x and p[1] == new_y for p in current if p is not current[idx]):
            continue
        
        candidate = [list(p) for p in current]
        candidate[idx] = [new_x, new_y]
        
        cand_valid, cand_score = _evaluate(candidate)
        
        if not cand_valid:
            cand_score = -1
        
        if cand_score > best_score:
            best_points = candidate
            best_score = cand_score
        
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
        else:
            delta = cand_score - current_score
            if delta < 0:
                prob = pow(2.718281828, delta / temp)
                if random.random() < prob:
                    current = candidate
                    current_score = cand_score
        
        temp *= cooling
    
    return best_points, best_score

def _generate_initial_grid():
    """Generate a 4x4 grid of points, which has many 4-point lines."""
    return [[x, y] for x in range(4) for y in range(4)]

def _generate_perturbed_grid(perturbation=3):
    """Generate a perturbed 4x4 grid to break symmetry while keeping structure."""
    base = [[x, y] for x in range(4) for y in range(4)]
    points = []
    for x, y in base:
        # Add small random perturbation
        dx = random.randint(-perturbation, perturbation)
        dy = random.randint(-perturbation, perturbation)
        new_x = x * 10 + dx
        new_y = y * 10 + dy
        if abs(new_x) <= 1000 and abs(new_y) <= 1000:
            points.append([new_x, new_y])
    # Ensure distinctness
    unique = []
    seen = set()
    for p in points:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    while len(unique) < 16:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique.append([x, y])
    return unique

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    random.seed(42)
    
    best_points = None
    best_score = 0
    
    # Strategy 1: Start with 4x4 grid and refine with SA
    initial_grid = _generate_initial_grid()
    grid_valid, grid_score = _evaluate(initial_grid)
    if grid_valid and grid_score > best_score:
        best_score = grid_score
        best_points = [list(p) for p in initial_grid]
    
    # Strategy 2: Try perturbed grids
    for trial in range(3):
        if time.time() > deadline - 30:
            break
        
        perturbed = _generate_perturbed_grid(perturbation=2 + trial)
        p_valid, p_score = _evaluate(perturbed)
        if p_valid and p_score > best_score:
            best_score = p_score
            best_points = [list(p) for p in perturbed]
    
    # Strategy 3: Simulated annealing from best configuration
    if best_points is None:
        best_points = _generate_initial_grid()
        best_score = _evaluate(best_points)[1]
    
    refined_points, refined_score = _simulated_annealing(best_points, deadline)
    
    if refined_score > best_score:
        best_points = refined_points
        best_score = refined_score
    
    # Final validation
    final_valid, final_score = _evaluate(best_points)
    if not final_valid:
        # Fallback to 4x4 grid
        best_points = _generate_initial_grid()
        final_score = _evaluate(best_points)[1]
    
    return best_points
# EVOLVE-BLOCK-END
