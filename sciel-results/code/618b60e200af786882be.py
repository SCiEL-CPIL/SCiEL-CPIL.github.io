"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
import numpy as np

def _count_four_lines(points):
    """Count lines containing exactly 4 of the given points.
    Returns (num_exact_four, has_five) where has_five is True if any line has >=5 points."""
    n = len(points)
    # Use a set of normalized line representations to count points per line
    # A line is determined by any two distinct points.
    line_points = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction: make dx, dy coprime and with a canonical sign
            if dx == 0 and dy == 0:
                continue
            g = _gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            # Canonical sign: first non-zero of (dx_n, dy_n) should be positive
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            # Line key: (dx_n, dy_n, c) where c is the offset such that dy_n*x - dx_n*y = c
            # For point (x1,y1): c = dy_n*x1 - dx_n*y1
            c = dy_n * x1 - dx_n * y1
            key = (dx_n, dy_n, c)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    num_exact_four = 0
    has_five = False
    for key, pts in line_points.items():
        cnt = len(pts)
        if cnt >= 5:
            has_five = True
        elif cnt == 4:
            num_exact_four += 1
    
    return num_exact_four, has_five

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a if a != 0 else b

def _validate(points):
    """Check if points are valid: 16 distinct, |x|,|y|<=1000, no five collinear."""
    if len(points) != 16:
        return False
    # Check distinctness
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return False
    # Check no five collinear
    _, has_five = _count_four_lines(points)
    return not has_five

def _evaluate(points):
    """Return the score: number of exact-4 lines, or -1 if invalid."""
    if not _validate(points):
        return -1
    score, _ = _count_four_lines(points)
    return score

def _generate_random_point(rng):
    return [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]

def _local_search(points, rng, max_iters=500):
    """Local search: try replacing one point at a time to improve score."""
    current = [list(p) for p in points]
    current_score = _evaluate(current)
    if current_score < 0:
        return current, current_score
    
    for it in range(max_iters):
        improved = False
        # Try modifying each point
        for idx in range(16):
            # Try a few random replacements
            for _ in range(3):
                new_point = _generate_random_point(rng)
                candidate = list(current)
                candidate[idx] = new_point
                new_score = _evaluate(candidate)
                if new_score > current_score:
                    current = candidate
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return current, current_score

def _hill_climbing(points, rng, max_iters=200):
    """Hill climbing with random restarts within a single call."""
    best = [list(p) for p in points]
    best_score = _evaluate(best)
    if best_score < 0:
        best_score = 0
    
    for it in range(max_iters):
        # Pick a random point to modify
        idx = rng.randint(0, 15)
        new_point = _generate_random_point(rng)
        candidate = list(best)
        candidate[idx] = new_point
        new_score = _evaluate(candidate)
        if new_score > best_score:
            best = candidate
            best_score = new_score
    
    return best, best_score

def _grid_based_generate(rng):
    """Generate points on a perturbed grid."""
    base_x = rng.randint(-500, 500)
    base_y = rng.randint(-500, 500)
    scale = rng.randint(1, 50)
    points = []
    for i in range(4):
        for j in range(4):
            x = base_x + i * scale + rng.randint(-scale//2, scale//2)
            y = base_y + j * scale + rng.randint(-scale//2, scale//2)
            x = max(-1000, min(1000, x))
            y = max(-1000, min(1000, y))
            points.append([x, y])
    return points

def _line_completion_search(points, rng, max_iters=300):
    """Search that tries to complete 3-point lines to 4-point lines."""
    current = [list(p) for p in points]
    current_score = _evaluate(current)
    if current_score < 0:
        return current, current_score
    
    for it in range(max_iters):
        # Find all lines with exactly 3 points
        n = len(current)
        line_points = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = current[i]
                x2, y2 = current[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                g = _gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n = -dx_n
                    dy_n = -dy_n
                c = dy_n * x1 - dx_n * y1
                key = (dx_n, dy_n, c)
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
        
        # Find lines with exactly 3 points
        three_lines = [(key, pts) for key, pts in line_points.items() if len(pts) == 3]
        
        if not three_lines:
            break
        
        # Pick a random 3-line
        key, pts = rng.choice(three_lines)
        dx_n, dy_n, c = key
        existing_indices = set(pts)
        
        # Try to add a point on this line
        # The line is: dy_n * x - dx_n * y = c
        # We need integer solutions. Parametrize:
        # If dx_n == 0: x = c/dy_n (must be integer), y varies
        # If dy_n == 0: y = -c/dx_n (must be integer), x varies
        # General: find one solution (x0, y0), then general solution is (x0 + t*dx_n, y0 + t*dy_n)
        
        # Find a particular solution
        # dy_n * x - dx_n * y = c
        # Use extended gcd
        g = _gcd(dx_n, dy_n)
        if c % g != 0:
            continue  # No integer solution
        
        # Solve dy_n * x - dx_n * y = c
        # This is a linear Diophantine equation
        # a*x + b*y = c where a=dy_n, b=-dx_n
        a = dy_n
        b = -dx_n
        # Find one solution using extended Euclidean
        # For simplicity, try small values
        found_solution = False
        for t in range(-200, 201):
            # Try x = t, solve for y
            if a != 0:
                # a*x - dx_n*y = c => dx_n*y = a*x - c
                rhs = a * t - c
                if rhs % dx_n == 0 and dx_n != 0:
                    y_val = rhs // dx_n
                    if abs(t) <= 1000 and abs(y_val) <= 1000:
                        new_pt = [t, y_val]
                        if tuple(new_pt) not in set(tuple(p) for p in current):
                            # Check if this point is already on the line (should be by construction)
                            # Add it and check score
                            candidate = list(current)
                            # Replace the point that's "least useful" or add a new one
                            # We need to keep 16 points, so replace one
                            # Replace a random point not on this line
                            non_line_indices = [i for i in range(16) if i not in existing_indices]
                            if non_line_indices:
                                replace_idx = rng.choice(non_line_indices)
                                candidate[replace_idx] = new_pt
                                new_score = _evaluate(candidate)
                                if new_score > current_score:
                                    current = candidate
                                    current_score = new_score
                                    found_solution = True
                                    break
            else:
                # a = 0, so -dx_n * y = c => y = -c/dx_n
                if dx_n != 0 and c % dx_n == 0:
                    y_val = -c // dx_n
                    if abs(y_val) <= 1000:
                        new_pt = [t, y_val]
                        if abs(t) <= 1000 and tuple(new_pt) not in set(tuple(p) for p in current):
                            non_line_indices = [i for i in range(16) if i not in existing_indices]
                            if non_line_indices:
                                replace_idx = rng.choice(non_line_indices)
                                candidate[replace_idx] = new_pt
                                new_score = _evaluate(candidate)
                                if new_score > current_score:
                                    current = candidate
                                    current_score = new_score
                                    found_solution = True
                                    break
            if found_solution:
                break
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    rng = random.Random(42)
    
    best_points = None
    best_score = -1
    
    # Initial: 4x4 grid
    grid_points = [[x, y] for x in range(4) for y in range(4)]
    score = _evaluate(grid_points)
    if score > best_score:
        best_score = score
        best_points = [list(p) for p in grid_points]
    
    # Try several strategies with time budget
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Generate initial points using different methods
        if restart % 3 == 0:
            points = _grid_based_generate(rng)
        elif restart % 3 == 1:
            points = [_generate_random_point(rng) for _ in range(16)]
        else:
            # Perturbed grid
            base = rng.randint(-500, 500)
            scale = rng.randint(10, 100)
            points = []
            for i in range(4):
                for j in range(4):
                    x = base + i * scale + rng.randint(-5, 5)
                    y = base + j * scale + rng.randint(-5, 5)
                    x = max(-1000, min(1000, x))
                    y = max(-1000, min(1000, y))
                    points.append([x, y])
        
        # Local search
        points, score = _local_search(points, rng, max_iters=100)
        if score > best_score:
            best_score = score
            best_points = [list(p) for p in points]
        
        # Line completion
        if time.time() > deadline - 10:
            break
        points2, score2 = _line_completion_search(points, rng, max_iters=50)
        if score2 > best_score:
            best_score = score2
            best_points = [list(p) for p in points2]
    
    # Final hill climbing on best
    if time.time() < deadline - 2 and best_points is not None:
        best_points, final_score = _hill_climbing(best_points, rng, max_iters=100)
        if final_score > best_score:
            best_score = final_score
    
    if best_points is None:
        best_points = grid_points
    
    return best_points
# EVOLVE-BLOCK-END
