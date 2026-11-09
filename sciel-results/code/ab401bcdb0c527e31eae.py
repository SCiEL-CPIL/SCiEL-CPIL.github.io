"""Improved program: Hypergraph-based optimization with incremental collinearity tracking
and direct minimization of the 3-free subset size using structured initial configurations."""
import random
import math
import time
from collections import defaultdict

# Deadline guard
_START_TIME = None
_DEADLINE = 145.0  # seconds (safety margin before 150s kill)

def _time_left():
    if _START_TIME is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START_TIME)

def _is_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

def _has_four_collinear(points):
    """Check if any four points are collinear. Returns True if violation found."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
                    if count >= 4:
                        return True
    return False

def _count_collinear_triples(points):
    """Count the number of collinear triples."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _max_3_free_subset(points):
    """Exact branch and bound for the largest subset with no three collinear."""
    n = len(points)
    if n <= 2:
        return n
    
    # Precompute for each pair (i,j), the set of points collinear with both
    collinear_with_pair = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if k != i and k != j and _is_collinear(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinear_with_pair[i][j] = mask
            collinear_with_pair[j][i] = mask
    
    best = [0]
    
    def backtrack(i, subset_mask, size):
        if size > best[0]:
            best[0] = size
        if i == n:
            return
        
        # Upper bound
        if size + (n - i) <= best[0]:
            return
        
        # Option 1: exclude point i
        backtrack(i + 1, subset_mask, size)
        
        # Option 2: include point i - check if it creates a collinear triple
        can_add = True
        temp_mask = subset_mask
        while temp_mask:
            a_bit = temp_mask & (-temp_mask)
            a = a_bit.bit_length() - 1
            if collinear_with_pair[a][i] & subset_mask:
                can_add = False
                break
            temp_mask ^= a_bit
        
        if can_add:
            backtrack(i + 1, subset_mask | (1 << i), size + 1)
    
    backtrack(0, 0, 0)
    return best[0]

def _generate_line_based_config():
    """Generate points on a set of lines to create many 3-collinearities.
    Use 7 lines, each with ~3 points, sharing points to get 20 total."""
    points = []
    used = set()
    
    # Define lines by two points each, add a third point on each line
    # Use a structured pattern
    base_points = [
        (0, 0), (1, 1), (2, 2),  # line y=x
        (0, 1), (1, 0), (2, -1),  # line y=-x+1
        (0, 2), (1, 3), (2, 4),  # line y=x+2
        (3, 0), (4, 1), (5, 2),  # line y=x-3
        (0, -1), (1, -2), (2, -3),  # line y=-x-1
        (-1, 0), (-2, 1), (-3, 2),  # line y=-x+1 (different segment)
        (0, 3), (2, 1), (4, -1),  # line y=-x+3
        (1, -1), (3, 1), (5, 3),  # line y=x-2
    ]
    
    # Scale and shift to fit in [-100,100] and avoid duplicates
    for p in base_points:
        if p not in used:
            points.append(p)
            used.add(p)
    
    # Add more points to reach 20
    while len(points) < 20:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        if (x, y) not in used:
            points.append((x, y))
            used.add((x, y))
    
    return points[:20]

def _generate_parabola_perturbed():
    """Points near a parabola with deliberate collinearities."""
    random.seed(123)
    points = []
    # Start with points on y=x^2
    for x in range(-5, 6):
        points.append((x, x * x))
    # That's 11 points, add 9 more near the parabola
    for _ in range(9):
        x = random.randint(-8, 8)
        y = x * x + random.randint(-2, 2)
        if (x, y) not in points and abs(x) <= 100 and abs(y) <= 100:
            points.append((x, y))
    
    # Ensure 20 unique points
    while len(points) < 20:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        if (x, y) not in points:
            points.append((x, y))
    
    return points[:20]

def _generate_grid_with_lines():
    """Grid-like pattern with added collinearities."""
    random.seed(456)
    points = []
    # 4x5 grid with perturbations
    for i in range(4):
        for j in range(5):
            x = i * 6 - 9
            y = j * 6 - 12
            # Small perturbation
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
            p = (x + dx, y + dy)
            if p not in points:
                points.append(p)
    
    while len(points) < 20:
        x = random.randint(-15, 15)
        y = random.randint(-15, 15)
        if (x, y) not in points:
            points.append((x, y))
    
    return points[:20]

def _fix_four_collinear(points):
    """Try to fix any 4-collinear violations by moving points."""
    attempts = 0
    while _has_four_collinear(points) and attempts < 10000:
        # Find which points are in a 4-collinear set and move one
        found = False
        for i in range(20):
            for j in range(i + 1, 20):
                collinear = [points[i], points[j]]
                for k in range(j + 1, 20):
                    if _is_collinear(points[i], points[j], points[k]):
                        collinear.append(points[k])
                        if len(collinear) >= 4:
                            # Move point i
                            idx = i
                            new = (random.randint(-30, 30), random.randint(-30, 30))
                            while new in points or abs(new[0]) > 100 or abs(new[1]) > 100:
                                new = (random.randint(-30, 30), random.randint(-30, 30))
                            points[idx] = new
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if not found:
            # Random move
            idx = random.randint(0, 19)
            new = (random.randint(-30, 30), random.randint(-30, 30))
            while new in points or abs(new[0]) > 100 or abs(new[1]) > 100:
                new = (random.randint(-30, 30), random.randint(-30, 30))
            points[idx] = new
        attempts += 1
    return points

def _delta_triples(points, idx, old_point, new_point):
    """Compute the change in number of collinear triples when moving point idx."""
    n = len(points)
    old_count = 0
    new_count = 0
    
    for i in range(n):
        if i == idx:
            continue
        for j in range(i + 1, n):
            if j == idx:
                continue
            if _is_collinear(old_point, points[i], points[j]):
                old_count += 1
            if _is_collinear(new_point, points[i], points[j]):
                new_count += 1
    
    return new_count - old_count

def _count_lines_with_3(points):
    """Count lines that have exactly 3 points from the set."""
    n = len(points)
    lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            # Normalize line representation
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g == 0:
                g = 1
            dx_n = dx // g if g != 0 else dx
            dy_n = dy // g if g != 0 else dy
            # Line: (x,y) such that (x - xi, y - yi) is parallel to (dx_n, dy_n)
            # Normalize: use the smallest point on the line
            key = (dx_n, dy_n, min(points[i], points[j]))
            count = 0
            for k in range(n):
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
            if count == 3:
                lines.add(key)
    return len(lines)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    random.seed(42)
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Config 1: Line-based
    initial_configs.append(_generate_line_based_config())
    
    # Config 2: Parabola perturbed
    initial_configs.append(_generate_parabola_perturbed())
    
    # Config 3: Grid with lines
    initial_configs.append(_generate_grid_with_lines())
    
    # Config 4: Structured - points on few lines
    pts4 = []
    # 5 lines with 4 points each, sharing points
    line_defs = [
        [(0,0), (1,1), (2,2), (3,3)],
        [(0,1), (1,0), (2,-1), (3,-2)],
        [(4,0), (5,1), (6,2), (7,3)],
        [(0,4), (1,3), (2,2), (3,1)],
        [(4,4), (5,3), (6,2), (7,1)],
    ]
    used = set()
    for line in line_defs:
        for p in line:
            if p not in used:
                pts4.append(p)
                used.add(p)
    # Add remaining
    while len(pts4) < 20:
        x = random.randint(-5, 8)
        y = random.randint(-3, 5)
        if (x, y) not in used:
            pts4.append((x, y))
            used.add((x, y))
    initial_configs.append(pts4[:20])
    
    # Fix all configs and evaluate
    best_points = None
    best_score = 21
    
    for config in initial_configs:
        if _time_left() < 5.0:
            break
        config = _fix_four_collinear(config)
        if _has_four_collinear(config):
            continue
        
        score = _max_3_free_subset(config)
        if score < best_score:
            best_score = score
            best_points = list(config)
    
    if best_points is None:
        # Fallback: random
        best_points = []
        for _ in range(20):
            p = (random.randint(-50, 50), random.randint(-50, 50))
            while p in best_points:
                p = (random.randint(-50, 50), random.randint(-50, 50))
            best_points.append(p)
        while _has_four_collinear(best_points):
            idx = random.randint(0, 19)
            best_points[idx] = (random.randint(-50, 50), random.randint(-50, 50))
            while best_points[idx] in best_points:
                best_points[idx] = (random.randint(-50, 50), random.randint(-50, 50))
        best_score = _max_3_free_subset(best_points)
    
    # Local search: directly minimize the 3-free subset size
    points = list(best_points)
    current_score = best_score
    
    # Phase 1: Simulated annealing
    temperature = 10.0
    cooling_rate = 0.999
    min_temperature = 0.1
    no_improve_count = 0
    max_no_improve = 3000
    
    while temperature > min_temperature and _time_left() > 20.0 and no_improve_count < max_no_improve:
        idx = random.randint(0, 19)
        old_point = points[idx]
        
        # Generate new point: biased toward creating collinearities
        r = random.random()
        if r < 0.4:
            # Place on line through two existing points
            i = random.randint(0, 19)
            j = random.randint(0, 19)
            if i == idx or j == idx or i == j:
                i = (i + 1) % 20
                j = (j + 2) % 20
                if i == idx or j == idx or i == j:
                    i = (idx + 1) % 20
                    j = (idx + 2) % 20
            p_i = points[i]
            p_j = points[j]
            dx = p_j[0] - p_i[0]
            dy = p_j[1] - p_i[1]
            if dx == 0 and dy == 0:
                continue
            # Try several positions on this line
            best_new = None
            for t in range(-3, 4):
                if t == 0:
                    continue
                nx = p_i[0] + t * dx
                ny = p_i[1] + t * dy
                if abs(nx) <= 100 and abs(ny) <= 100 and (nx, ny) not in points:
                    # Check no 4-collinear
                    points[idx] = (nx, ny)
                    if not _has_four_collinear(points):
                        best_new = (nx, ny)
                        break
            points[idx] = old_point
            if best_new is None:
                continue
            new_point = best_new
        elif r < 0.7:
            # Near another point
            other_idx = random.randint(0, 19)
            if other_idx == idx:
                other_idx = (other_idx + 1) % 20
            other = points[other_idx]
            new_point = (other[0] + random.randint(-4, 4), other[1] + random.randint(-4, 4))
        else:
            # Random in box
            new_point = (random.randint(-30, 30), random.randint(-30, 30))
        
        if new_point in points or abs(new_point[0]) > 100 or abs(new_point[1]) > 100:
            continue
        
        # Check validity
        points[idx] = new_point
        if _has_four_collinear(points):
            points[idx] = old_point
            continue
        
        # Compute delta in triples
        delta_triples = _delta_triples(points, idx, old_point, new_point)
        
        # Compute new exact score
        new_score = _max_3_free_subset(points)
        delta_score = new_score - current_score
        
        if delta_score <= 0:
            # Improvement or equal
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = list(points)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Worse, accept with probability
            if random.random() < math.exp(-delta_score / temperature):
                current_score = new_score
                no_improve_count += 1
            else:
                points[idx] = old_point
                no_improve_count += 1
        
        temperature *= cooling_rate
    
    # Phase 2: Targeted refinement - try to reduce score by specific moves
    points = list(best_points)
    current_score = best_score
    
    for iteration in range(2000):
        if _time_left() < 5.0:
            break
        
        idx = random.randint(0, 19)
        old_point = points[idx]
        
        # Try to place point to create a new collinear triple with two existing points
        found_improvement = False
        for i in range(20):
            if i == idx:
                continue
            for j in range(i + 1, 20):
                if j == idx:
                    continue
                p_i = points[i]
                p_j = points[j]
                dx = p_j[0] - p_i[0]
                dy = p_j[1] - p_i[1]
                
                # Check if there's already a third point on this line
                third_count = 0
                for k in range(20):
                    if k != i and k != j and _is_collinear(p_i, p_j, points[k]):
                        third_count += 1
                
                if third_count >= 2:
                    continue  # Would create 4-collinear
                
                # Try to place idx on this line
                for t in range(-5, 6):
                    if t == 0:
                        continue
                    nx = p_i[0] + t * dx
                    ny = p_i[1] + t * dy
                    if abs(nx) <= 100 and abs(ny) <= 100 and (nx, ny) not in points:
                        points[idx] = (nx, ny)
                        if not _has_four_collinear(points):
                            new_score = _max_3_free_subset(points)
                            if new_score < current_score:
                                current_score = new_score
                                if current_score < best_score:
                                    best_score = current_score
                                    best_points = list(points)
                                    found_improvement = True
                            else:
                                points[idx] = old_point
                            break
                if found_improvement:
                    break
            if found_improvement:
                break
        
        if not found_improvement:
            # Try small random perturbation
            new_point = (old_point[0] + random.randint(-3, 3), old_point[1] + random.randint(-3, 3))
            if new_point not in points and abs(new_point[0]) <= 100 and abs(new_point[1]) <= 100:
                points[idx] = new_point
                if not _has_four_collinear(points):
                    new_score = _max_3_free_subset(points)
                    if new_score < current_score:
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_points = list(points)
                    else:
                        points[idx] = old_point
    
    # Phase 3: Final polish - try all single-point moves for best points
    points = list(best_points)
    current_score = best_score
    
    for idx in range(20):
        if _time_left() < 3.0:
            break
        old_point = points[idx]
        
        # Try nearby positions
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue
                nx = old_point[0] + dx
                ny = old_point[1] + dy
                if abs(nx) > 100 or abs(ny) > 100:
                    continue
                if (nx, ny) in points:
                    continue
                
                points[idx] = (nx, ny)
                if _has_four_collinear(points):
                    points[idx] = old_point
                    continue
                
                new_score = _max_3_free_subset(points)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_points = list(points)
                else:
                    points[idx] = old_point
    
    return [list(p) for p in best_points]
