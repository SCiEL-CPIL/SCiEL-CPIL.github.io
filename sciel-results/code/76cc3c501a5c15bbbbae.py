import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return None
        dx //= g
        dy //= g
        # Normalize direction
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        # Equation: dy*x - dx*y = C
        C = dy * x1 - dx * y1
        return (dx, dy, C)

    def evaluate(points):
        # Fast evaluation: count points on each unique line
        if len(set(points)) != 16:
            return -1
        
        for x, y in points:
            if abs(x) > 1000 or abs(y) > 1000:
                return -1
        
        # Collect all unique lines from pairs
        unique_lines = set()
        for i in range(16):
            for j in range(i + 1, 16):
                key = line_key(points[i], points[j])
                if key:
                    unique_lines.add(key)
        
        # Count points on each line
        score = 0
        for key in unique_lines:
            dx, dy, C = key
            count = 0
            for p in points:
                x, y = p
                if dx * x - dy * y == C:
                    count += 1
            if count == 4:
                score += 1
            elif count >= 5:
                return -1  # Invalid
        
        return score

    def random_point():
        return (random.randint(-1000, 1000), random.randint(-1000, 1000))

    def in_bounds(p):
        return abs(p[0]) <= 1000 and abs(p[1]) <= 1000

    def local_search(points, max_iters=10000):
        current_score = evaluate(points)
        if current_score < 0:
            return points, 0
        
        best_points = list(points)
        best_score = current_score
        current_points = list(points)
        
        # Track lines for structured moves
        def get_line_structure(pts):
            """Return dict mapping line_key -> list of point indices on that line"""
            line_map = defaultdict(list)
            for i in range(16):
                for j in range(i + 1, 16):
                    key = line_key(pts[i], pts[j])
                    if key:
                        line_map[key].append(i)
            # Deduplicate indices in each line
            for key in line_map:
                line_map[key] = list(set(line_map[key]))
            return line_map
        
        line_struct = get_line_structure(current_points)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random point to move
            idx = random.randint(0, 15)
            old_p = current_points[idx]
            
            # Strategy: Multiple move types
            move_type = random.random()
            
            if move_type < 0.3:
                # Small perturbation
                delta = random.randint(-3, 3)
                new_p = (old_p[0] + delta, old_p[1] + random.randint(-3, 3))
            elif move_type < 0.5:
                # Medium perturbation
                new_p = (old_p[0] + random.randint(-15, 15), old_p[1] + random.randint(-15, 15))
            elif move_type < 0.7:
                # Line-aligned move: move along a line that contains this point
                # Find lines containing idx
                lines_with_idx = [key for key, pts in line_struct.items() if idx in pts and len(pts) >= 2]
                if lines_with_idx:
                    key = random.choice(lines_with_idx)
                    dx, dy, C = key
                    # Move along this line by a small step
                    step = random.randint(-10, 10)
                    if step != 0:
                        # Find another point on this line to get direction
                        other_pts = [p for i, p in enumerate(current_points) if i != idx and dx * p[0] - dy * p[1] == C]
                        if other_pts:
                            ref_p = random.choice(other_pts)
                            # Direction vector from ref_p to old_p
                            dir_x = old_p[0] - ref_p[0]
                            dir_y = old_p[1] - ref_p[1]
                            norm = math.sqrt(dir_x**2 + dir_y**2)
                            if norm > 0:
                                unit_x = dir_x / norm
                                unit_y = dir_y / norm
                                new_p = (int(old_p[0] + unit_x * step), int(old_p[1] + unit_y * step))
                            else:
                                new_p = random_point()
                        else:
                            new_p = random_point()
                    else:
                        new_p = random_point()
                else:
                    new_p = random_point()
            else:
                # Random jump
                new_p = random_point()
            
            # Ensure distinctness and bounds
            if new_p in current_points:
                continue
            if not in_bounds(new_p):
                continue
                
            current_points[idx] = new_p
            new_score = evaluate(current_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = list(current_points)
                current_score = new_score
                # Update line structure for best points
                line_struct = get_line_structure(current_points)
            elif new_score == best_score:
                # Accept equal score to explore (simulated annealing style)
                if random.random() < 0.6:
                    current_score = new_score
                    line_struct = get_line_structure(current_points)
                else:
                    current_points[idx] = old_p
            else:
                # Worse score, revert
                current_points[idx] = old_p
        
        return best_points, best_score

    # Initial configurations
    best_overall_score = 0
    best_overall_points = None
    
    # 1. 4x4 Grid - good starting point
    grid_points = [(x, y) for x in range(4) for y in range(4)]
    p, s = local_search(grid_points, max_iters=8000)
    if s > best_overall_score:
        best_overall_score = s
        best_overall_points = p
    
    # 2. Structured initial configurations
    # Try points on a parabola-like structure
    for trial in range(5):
        if time.time() > deadline:
            break
        # Generate points with some structure
        pts = []
        for i in range(16):
            x = i * random.randint(20, 50)
            y = (i * i) * random.randint(5, 15) + random.randint(-20, 20)
            if abs(x) > 1000:
                x = 1000 if x > 0 else -1000
            if abs(y) > 1000:
                y = 1000 if y > 0 else -1000
            pts.append((x, y))
        # Ensure distinct
        while len(set(pts)) < 16:
            pts[random.randint(0, 15)] = random_point()
        p, s = local_search(pts, max_iters=3000)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = p
    
    # 3. Random restarts
    for _ in range(8):
        if time.time() > deadline:
            break
        random_points = [random_point() for _ in range(16)]
        # Ensure distinct
        while len(set(random_points)) < 16:
            random_points[random.randint(0, 15)] = random_point()
        p, s = local_search(random_points, max_iters=2500)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = p
    
    # 4. Final polish: more iterations on the best found
    if best_overall_points and time.time() < deadline - 5:
        p, s = local_search(best_overall_points, max_iters=5000)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = p
    
    if best_overall_points is None:
        return [[x, y] for x in range(4) for y in range(4)]
    
    return [[x, y] for x, y in best_overall_points]
