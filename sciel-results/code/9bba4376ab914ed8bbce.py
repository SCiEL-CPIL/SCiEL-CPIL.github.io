"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict
from math import gcd

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 0, x1)
    if dy == 0:
        return (0, 1, y1)
    a = dy
    b = -dx
    c = a * x1 + b * y1
    g = gcd(abs(a), abs(b))
    if g > 1:
        a //= g
        b //= g
        c //= g
    if a < 0:
        a = -a
        b = -b
        c = -c
    elif a == 0 and b < 0:
        b = -b
        c = -c
    return (a, b, c)

def _build_line_map(points):
    """Build line map: line_key -> set of point indices."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_map[key].add(i)
                line_map[key].add(j)
    return line_map

def _score_from_line_map(line_map):
    """Count lines with exactly 4 points, and check no 5+ collinear. Returns (count_4, valid)."""
    count_4 = 0
    for key, idx_set in line_map.items():
        sz = len(idx_set)
        if sz >= 5:
            return 0, False
        if sz == 4:
            count_4 += 1
    return count_4, True

def _score(points):
    """Score: number of exactly-4 lines if valid, else 0."""
    if len(points) != 16:
        return 0
    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return 0
        if (x, y) in seen:
            return 0
        seen.add((x, y))
    line_map = _build_line_map(points)
    count_4, valid = _score_from_line_map(line_map)
    return count_4 if valid else 0

def _is_valid(points):
    """Check validity."""
    if len(points) != 16:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    line_map = _build_line_map(points)
    for key, idx_set in line_map.items():
        if len(idx_set) >= 5:
            return False
    return True

def _local_search(points, max_time, rng):
    """Local search with simulated annealing to maximize 4-point lines."""
    start_time = time.time()
    current = [list(p) for p in points]
    current_score = _score(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    # Temperature schedule
    T = 2.0
    cooling = 0.9995
    
    small_offsets = []
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            if dx != 0 or dy != 0:
                small_offsets.append((dx, dy))
    
    # Medium and large offsets for escaping local optima
    medium_offsets = []
    for dx in range(-8, 9):
        for dy in range(-8, 9):
            if dx != 0 or dy != 0:
                medium_offsets.append((dx, dy))
    
    large_offsets = []
    for _ in range(50):
        dx = rng.randint(-20, 20)
        dy = rng.randint(-20, 20)
        if dx != 0 or dy != 0:
            large_offsets.append((dx, dy))
    
    iteration = 0
    while time.time() - start_time < max_time:
        iteration += 1
        if T < 0.01:
            T = 0.01
        
        # Choose move size based on temperature
        r = rng.random()
        if r < 0.5:
            dx, dy = rng.choice(small_offsets)
        elif r < 0.85:
            dx, dy = rng.choice(medium_offsets)
        else:
            dx, dy = rng.choice(large_offsets)
        
        idx = rng.randint(0, 15)
        new_points = [list(p) for p in current]
        new_x = new_points[idx][0] + dx
        new_y = new_points[idx][1] + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        new_points[idx] = [new_x, new_y]
        
        # Check distinctness
        if len(set(tuple(p) for p in new_points)) != 16:
            continue
        
        new_score = _score(new_points)
        
        if new_score >= current_score:
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best = [list(p) for p in current]
                best_score = current_score
        else:
            # Metropolis acceptance
            delta = current_score - new_score
            if delta > 0:
                accept_prob = math.exp(-delta / T)
                if rng.random() < accept_prob:
                    current = new_points
                    current_score = new_score
        
        T *= cooling
    
    return best, best_score

def _try_structured_moves(points, max_time, rng):
    """Try structured moves: swap pairs, shift rows/cols, etc."""
    start_time = time.time()
    current = [list(p) for p in points]
    current_score = _score(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    while time.time() - start_time < max_time:
        move_type = rng.randint(0, 3)
        new_points = [list(p) for p in current]
        
        if move_type == 0:
            # Swap two points
            i, j = rng.sample(range(16), 2)
            new_points[i], new_points[j] = new_points[j], new_points[i]
        elif move_type == 1:
            # Move one point to a new random location
            idx = rng.randint(0, 15)
            new_x = rng.randint(-100, 100)
            new_y = rng.randint(-100, 100)
            new_points[idx] = [new_x, new_y]
        elif move_type == 2:
            # Shift a "row" of points (points with similar y)
            # Find points with the same y
            from collections import Counter
            y_counts = Counter(p[1] for p in current)
            common_ys = [y for y, c in y_counts.items() if c >= 3]
            if common_ys:
                target_y = rng.choice(common_ys)
                dx = rng.randint(-3, 3)
                for i, (x, y) in enumerate(current):
                    if y == target_y:
                        new_points[i][0] = x + dx
        else:
            # Move two points together
            i, j = rng.sample(range(16), 2)
            dx = rng.randint(-5, 5)
            dy = rng.randint(-5, 5)
            if dx != 0 or dy != 0:
                new_points[i][0] += dx
                new_points[i][1] += dy
                new_points[j][0] += dx
                new_points[j][1] += dy
        
        # Validate
        if len(set(tuple(p) for p in new_points)) != 16:
            continue
        valid = True
        for x, y in new_points:
            if abs(x) > 1000 or abs(y) > 1000:
                valid = False
                break
        if not valid:
            continue
        
        new_score = _score(new_points)
        if new_score > best_score:
            best = new_points
            best_score = new_score
            current = new_points
            current_score = new_score
        elif new_score == current_score and rng.random() < 0.3:
            current = new_points
            current_score = new_score
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    rng = random.Random(42)
    
    # Start with 4x4 grid
    initial = [[x, y] for x in range(4) for y in range(4)]
    
    best_points = [list(p) for p in initial]
    best_score = _score(initial)
    
    # Phase 1: Local search from grid
    remaining = deadline - time.time()
    if remaining > 10:
        pts, sc = _local_search(initial, min(30, remaining - 5), rng)
        if sc > best_score:
            best_points = pts
            best_score = sc
    
    # Phase 2: Structured moves
    remaining = deadline - time.time()
    if remaining > 10:
        pts, sc = _try_structured_moves(best_points, min(20, remaining - 5), rng)
        if sc > best_score:
            best_points = pts
            best_score = sc
    
    # Phase 3: More local search from best
    remaining = deadline - time.time()
    if remaining > 10:
        pts, sc = _local_search(best_points, min(30, remaining - 5), rng)
        if sc > best_score:
            best_points = pts
            best_score = sc
    
    # Phase 4: Try random restarts
    remaining = deadline - time.time()
    restart_count = 0
    while remaining > 15 and restart_count < 5:
        # Generate a random perturbation of the grid
        perturbed = []
        for x, y in initial:
            if rng.random() < 0.4:
                nx = x + rng.randint(-3, 3)
                ny = y + rng.randint(-3, 3)
                perturbed.append([nx, ny])
            else:
                perturbed.append([x, y])
        
        if len(set(tuple(p) for p in perturbed)) == 16:
            valid = True
            for px, py in perturbed:
                if abs(px) > 1000 or abs(py) > 1000:
                    valid = False
                    break
            if valid:
                remaining = deadline - time.time()
                if remaining > 10:
                    pts, sc = _local_search(perturbed, min(15, remaining - 5), rng)
                    if sc > best_score:
                        best_points = pts
                        best_score = sc
        restart_count += 1
    
    # Phase 5: Final polish
    remaining = deadline - time.time()
    if remaining > 5:
        pts, sc = _local_search(best_points, min(10, remaining - 2), rng)
        if sc > best_score:
            best_points = pts
            best_score = sc
    
    # Final validation
    if not _is_valid(best_points):
        return [list(p) for p in initial]
    
    return [list(p) for p in best_points]
# EVOLVE-BLOCK-END
