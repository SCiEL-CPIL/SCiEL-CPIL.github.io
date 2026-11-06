"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly two of the points. Fast version."""
    n = len(points)
    if n < 2:
        return 0
    # Build line -> set of point indices
    line_pts = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                key = (0, 0, xi)
            elif dy == 0:
                key = (0, 0, yi)
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0:
                    dx_n = -dx_n
                    dy_n = -dy_n
                elif dx_n == 0 and dy_n < 0:
                    dy_n = -dy_n
                c = dy_n * xi - dx_n * yi
                key = (dx_n, dy_n, c)
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    ordinary = 0
    for pts in line_pts.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _all_collinear(points):
    """Check if all points are collinear."""
    if len(points) < 3:
        return False
    x0, y0 = points[0]
    x1, y1 = points[1]
    for i in range(2, len(points)):
        xi, yi = points[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return False
    return True

def _generate_structured_configs():
    """Generate a variety of structured 12-point configurations."""
    configs = []
    
    # 4x3 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    # 3x4 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    # 2x6 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    # 6x2 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    # 12x1 grid (all collinear - skip)
    
    # Points on two parallel lines (6 each)
    configs.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    # Points on three parallel lines (4 each)
    configs.append([[x, 0] for x in range(4)] + [[x, 1] for x in range(4)] + [[x, 2] for x in range(4)])
    # Points on four parallel lines (3 each)
    configs.append([[x, 0] for x in range(3)] + [[x, 1] for x in range(3)] + [[x, 2] for x in range(3)] + [[x, 3] for x in range(3)])
    
    # Böröczky-like: 6 points on a circle + 6 "at infinity" (far away)
    # Use Pythagorean triples for circle points
    circle_pts = [[3, 4], [4, 3], [-3, 4], [-4, 3], [3, -4], [4, -3]]
    # "Infinity" points: far away in 6 different directions
    D = 1000
    inf_pts = [[D, 0], [-D, 0], [0, D], [0, -D], [D, D], [D, -D]]
    configs.append(circle_pts + inf_pts)
    
    # Another Böröczky variant
    circle_pts2 = [[5, 12], [12, 5], [-5, 12], [-12, 5], [5, -12], [12, -5]]
    configs.append(circle_pts2 + inf_pts)
    
    # Mixed: 4 on one line, 4 on another, 4 on a third (non-parallel)
    configs.append([[x, 0] for x in range(4)] + [[x, 1] for x in range(4)] + [[x, x] for x in range(4)])
    
    # 3x3 grid + 3 extra points
    base_3x3 = [[x, y] for x in range(3) for y in range(3)]
    # Add points to create more collinearities
    configs.append(base_3x3 + [[0, 3], [1, 3], [2, 3]])  # extends to 3x4
    configs.append(base_3x3 + [[3, 0], [3, 1], [3, 2]])  # extends to 4x3
    
    # Affine plane AG(2,3) has 9 points; add 3 more
    ag23 = [[x, y] for x in range(3) for y in range(3)]
    # Add points that create new lines with existing points
    configs.append(ag23 + [[0, 3], [3, 0], [3, 3]])
    configs.append(ag23 + [[1, 3], [3, 1], [2, 3]])
    configs.append(ag23 + [[0, 3], [3, 0], [2, 3]])
    
    # Symmetric configurations
    # Hexagon + center + 6 more
    hex_pts = [[3, 0], [0, 3], [-3, 0], [0, -3], [3, 3], [-3, -3]]
    configs.append(hex_pts + [[-3, 3], [3, -3], [0, 0], [6, 0], [0, 6], [6, 6]])
    
    return configs

def _local_search(points, time_limit=120, seed=42):
    """Local search to minimize ordinary lines."""
    rng = random.Random(seed)
    start_time = time.time()
    best_points = [list(p) for p in points]
    best_score = _count_ordinary_lines(best_points)
    
    n = len(best_points)
    max_coord = 10**9
    
    def _try_move(points, idx, new_pos):
        """Try moving point idx to new_pos, return (new_score, new_points) if valid."""
        new_points = [list(p) for p in points]
        new_points[idx] = new_pos
        # Check distinctness
        seen = set()
        for p in new_points:
            t = (p[0], p[1])
            if t in seen:
                return None, None
            seen.add(t)
        if _all_collinear(new_points):
            return None, None
        score = _count_ordinary_lines(new_points)
        return score, new_points
    
    while time.time() - start_time < time_limit:
        improved = False
        for _ in range(50):  # 50 random moves per iteration
            if time.time() - start_time > time_limit - 2:
                break
            idx = rng.randint(0, n - 1)
            # Generate a random new position
            # Try positions near existing points or on lines through existing points
            strategy = rng.random()
            if strategy < 0.3:
                # Move to a random position
                x = rng.randint(-max_coord, max_coord)
                y = rng.randint(-max_coord, max_coord)
            elif strategy < 0.6:
                # Move to a position that creates collinearity with two existing points
                if n >= 3:
                    i1, i2 = rng.sample(range(n), 2)
                    if i1 == idx or i2 == idx:
                        continue
                    x1, y1 = best_points[i1]
                    x2, y2 = best_points[i2]
                    # Pick a point on the line through i1 and i2
                    t = rng.randint(-20, 20)
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    if x == x1 and y == y1:
                        continue
            else:
                # Small perturbation
                dx = rng.randint(-5, 5)
                dy = rng.randint(-5, 5)
                x = best_points[idx][0] + dx
                y = best_points[idx][1] + dy
            
            if abs(x) > max_coord or abs(y) > max_coord:
                continue
            
            new_score, new_points = _try_move(best_points, idx, [x, y])
            if new_score is None:
                continue
            if new_score < best_score:
                best_points = new_points
                best_score = new_score
                improved = True
                if best_score <= 6:
                    return best_points
            elif new_score == best_score and rng.random() < 0.1:
                # Accept equal-score moves occasionally (simulated annealing-like)
                best_points = new_points
        
        # If no improvement in a full pass, try larger jumps
        if not improved:
            # Random restart with perturbation
            for _ in range(10):
                idx = rng.randint(0, n - 1)
                x = rng.randint(-max_coord, max_coord)
                y = rng.randint(-max_coord, max_coord)
                if abs(x) > max_coord or abs(y) > max_coord:
                    continue
                new_score, new_points = _try_move(best_points, idx, [x, y])
                if new_score is not None and new_score < best_score:
                    best_points = new_points
                    best_score = new_score
                    if best_score <= 6:
                        return best_points
    
    return best_points

def construct():
    """Return exactly 12 distinct integer points with few ordinary lines."""
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # Generate structured configs and evaluate
    configs = _generate_structured_configs()
    best_points = None
    best_score = float('inf')
    
    for config in configs:
        if time.time() - start_time > deadline - 30:
            break
        if len(config) != 12:
            continue
        if len(set(map(tuple, config))) != 12:
            continue
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in config):
            continue
        if _all_collinear(config):
            continue
        score = _count_ordinary_lines(config)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in config]
            if best_score <= 6:
                return best_points
    
    if best_points is None:
        # Fallback
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary_lines(best_points)
    
    # Local search from the best structured config
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 10:
        best_points = _local_search(best_points, time_limit=remaining_time, seed=42)
        best_score = _count_ordinary_lines(best_points)
    
    # Try local search from a few other good configs
    for config in configs[:5]:
        if time.time() - start_time > deadline - 10:
            break
        if len(config) != 12 or len(set(map(tuple, config))) != 12:
            continue
        if _all_collinear(config):
            continue
        score = _count_ordinary_lines(config)
        if score < best_score:
            best_points = [list(p) for p in config]
            best_score = score
            remaining_time = deadline - (time.time() - start_time)
            if remaining_time > 5:
                best_points = _local_search(best_points, time_limit=remaining_time, seed=123)
                best_score = _count_ordinary_lines(best_points)
        if best_score <= 6:
            break
    
    # Final check
    if len(best_points) != 12 or len(set(map(tuple, best_points))) != 12:
        best_points = [[x, y] for x in range(4) for y in range(3)]
    if _all_collinear(best_points):
        best_points = [[x, y] for x in range(4) for y in range(3)]
    if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in best_points):
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
# EVOLVE-BLOCK-END
