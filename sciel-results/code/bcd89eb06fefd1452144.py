"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import math
import time
from collections import defaultdict

# EVOLVE-BLOCK-START

def _line_key(x1, y1, x2, y2):
    """Compute canonical line key from two points."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    a = dy
    b = -dx
    c = a * x1 + b * y1
    g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _build_line_index(points):
    """Build line count index from points. Returns (line_counts, valid)."""
    n = len(points)
    if n != 16:
        return {}, False
    
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return {}, False
        if (x, y) in seen:
            return {}, False
        seen.add((x, y))
    
    line_counts = defaultdict(int)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            if key is not None:
                line_counts[key] += 1
                if line_counts[key] >= 5:
                    return line_counts, False
    
    return line_counts, True

def _evaluate_from_index(line_counts):
    """Count 4-point lines from pre-built index."""
    four_count = 0
    for count in line_counts.values():
        if count == 4:
            four_count += 1
    return four_count

def _evaluate(points):
    """Return the number of 4-point lines, or -1 if invalid."""
    line_counts, valid = _build_line_index(points)
    if not valid:
        return -1
    return _evaluate_from_index(line_counts)

def _incremental_update(points, old_idx, new_idx, new_x, new_y):
    """Update line counts when moving point old_idx to (new_x, new_y).
    Returns (new_line_counts, valid)."""
    n = len(points)
    old_x, old_y = points[old_idx]
    if old_x == new_x and old_y == new_y:
        return None, True  # No change
    
    # Check new position validity
    if abs(new_x) > 1000 or abs(new_y) > 1000:
        return None, False
    for i in range(n):
        if i == old_idx:
            continue
        if points[i][0] == new_x and points[i][1] == new_y:
            return None, False
    
    # Decrement lines through old point
    line_counts = defaultdict(int)
    for key, count in _build_line_index(points)[0].items():
        line_counts[key] = count
    
    # Remove old point's contributions
    for i in range(n):
        if i == old_idx:
            continue
        key = _line_key(old_x, old_y, points[i][0], points[i][1])
        if key is not None:
            line_counts[key] -= 1
            if line_counts[key] <= 0:
                del line_counts[key]
    
    # Add new point's contributions
    for i in range(n):
        if i == old_idx:
            continue
        key = _line_key(new_x, new_y, points[i][0], points[i][1])
        if key is not None:
            line_counts[key] += 1
            if line_counts[key] >= 5:
                return line_counts, False
    
    return line_counts, True

def _perturb(points, rng):
    """Perturb points with various moves."""
    new_points = [p[:] for p in points]
    move_type = rng.random()
    
    if move_type < 0.5:
        # Standard random jump for 1-2 points
        num_moves = rng.choice([1, 2])
        for _ in range(num_moves):
            idx = rng.randint(0, 15)
            new_points[idx][0] += rng.randint(-20, 20)
            new_points[idx][1] += rng.randint(-20, 20)
    elif move_type < 0.8:
        # Translate a small cluster
        center_idx = rng.randint(0, 15)
        cx, cy = new_points[center_idx]
        neighbors = [center_idx]
        for i in range(16):
            if i == center_idx: continue
            dx = abs(new_points[i][0] - cx)
            dy = abs(new_points[i][1] - cy)
            if dx <= 2 and dy <= 2:
                neighbors.append(i)
        
        dx = rng.randint(-10, 10)
        dy = rng.randint(-10, 10)
        for idx in neighbors:
            new_points[idx][0] += dx
            new_points[idx][1] += dy
    else:
        # Swap coordinates of two points
        i, j = rng.sample(range(16), 2)
        new_points[i][0], new_points[j][0] = new_points[j][0], new_points[i][0]
        new_points[i][1], new_points[j][1] = new_points[j][1], new_points[i][1]
        
    return new_points

def _maxsat_score(points, line_counts, valid):
    """Max-SAT-inspired score: reward 4-point lines, heavily penalize 5+ collinear.
    Uses smooth surrogate: score = sum(soft_max(0, 4-count) for lines) - penalty*violations"""
    if not valid:
        return -1000  # Heavy penalty for invalid
    
    # Count 4-point lines
    four_count = 0
    five_plus = 0
    for count in line_counts.values():
        if count == 4:
            four_count += 1
        elif count >= 5:
            five_plus += 1
    
    # Smooth surrogate: log-sum-exp style soft reward for exactly-4
    # Hard penalty for violations
    score = four_count - 10 * five_plus
    return score

def construct():
    """Find 16 integer points maximizing 4-point lines with no five collinear."""
    deadline = time.time() + 110
    
    # Initial configurations: affine plane of order 4 (4x4 grid)
    init_grid = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in init_grid]
    best_line_counts, best_valid = _build_line_index(best_points)
    best_score = _evaluate_from_index(best_line_counts) if best_valid else -1
    
    rng = random.Random(42)
    
    # Multi-start with incremental updates
    while time.time() < deadline - 5:
        # Start from best or random
        if rng.random() < 0.7 or best_score <= 0:
            current_points = [p[:] for p in best_points]
            current_line_counts = dict(best_line_counts)
            current_valid = best_valid
            current_score = best_score
        else:
            current_points = [[rng.randint(-20, 20), rng.randint(-20, 20)] for _ in range(16)]
            current_line_counts, current_valid = _build_line_index(current_points)
            current_score = _evaluate_from_index(current_line_counts) if current_valid else -1
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                best_line_counts = dict(current_line_counts)
                best_valid = current_valid
        
        # Simulated Annealing with incremental updates
        temperature = 30.0
        cooling = 0.995
        min_temperature = 0.5
        max_iter = 5000
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if temperature < min_temperature:
                break
                
            # Generate perturbation
            new_points = [p[:] for p in current_points]
            move_type = rng.random()
            
            if move_type < 0.5:
                num_moves = rng.choice([1, 2])
                for _ in range(num_moves):
                    idx = rng.randint(0, 15)
                    new_points[idx][0] += rng.randint(-20, 20)
                    new_points[idx][1] += rng.randint(-20, 20)
            elif move_type < 0.8:
                center_idx = rng.randint(0, 15)
                cx, cy = new_points[center_idx]
                neighbors = [center_idx]
                for i in range(16):
                    if i == center_idx: continue
                    dx = abs(new_points[i][0] - cx)
                    dy = abs(new_points[i][1] - cy)
                    if dx <= 2 and dy <= 2:
                        neighbors.append(i)
                
                dx = rng.randint(-10, 10)
                dy = rng.randint(-10, 10)
                for idx in neighbors:
                    new_points[idx][0] += dx
                    new_points[idx][1] += dy
            else:
                i, j = rng.sample(range(16), 2)
                new_points[i][0], new_points[j][0] = new_points[j][0], new_points[i][0]
                new_points[i][1], new_points[j][1] = new_points[j][1], new_points[i][1]
            
            # Evaluate new configuration
            new_line_counts, new_valid = _build_line_index(new_points)
            new_score = _evaluate_from_index(new_line_counts) if new_valid else -1
            
            if new_score > current_score:
                current_points = new_points
                current_line_counts = new_line_counts
                current_valid = new_valid
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                    best_line_counts = dict(current_line_counts)
                    best_valid = current_valid
            elif new_score >= 0:
                delta = current_score - new_score
                if rng.random() < math.exp(-delta / max(temperature, 0.01)):
                    current_points = new_points
                    current_line_counts = new_line_counts
                    current_valid = new_valid
                    current_score = new_score
            
            temperature *= cooling

    return best_points

# EVOLVE-BLOCK-END
