import time
import random
import math

def _score(points):
    """
    Calculate the number of lines containing exactly 4 points.
    Returns 0 if any line has 5 or more points (invalid).
    Returns -1 if points are not distinct.
    """
    n = len(points)
    # Check distinctness
    if len(set(map(tuple, points))) != n:
        return -1
        
    # Map from line key to set of points on that line
    line_points = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line equation: ax + by + c = 0
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            
            # Normalize the line equation
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g != 0:
                a //= g
                b //= g
                c //= g
            else:
                continue # Should not happen for distinct points
                
            # Ensure consistent sign
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
                
            key = (a, b, c)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
            
    four_lines = 0
    for pts in line_points.values():
        k = len(pts)
        if k == 4:
            four_lines += 1
        elif k >= 5:
            return 0 # Invalid configuration: 5 or more collinear
            
    return four_lines

def _incremental_score(points, moved_idx, old_x, old_y, new_x, new_y):
    """
    Incrementally update the score when one point moves from (old_x, old_y) to (new_x, new_y).
    Returns the new score, or None if invalid (5+ collinear or duplicate).
    """
    n = len(points)
    
    # Check distinctness
    for k in range(n):
        if k != moved_idx and points[k][0] == new_x and points[k][1] == new_y:
            return None
    
    # Build line map from current points (with the moved point at new position)
    line_points = {}
    
    # First, add all pairs NOT involving moved_idx
    for i in range(n):
        if i == moved_idx:
            continue
        x1, y1 = points[i]
        for j in range(i + 1, n):
            if j == moved_idx:
                continue
            x2, y2 = points[j]
            
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g != 0:
                a //= g
                b //= g
                c //= g
            else:
                continue
                
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
                
            key = (a, b, c)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
            
    # Now add lines involving moved_idx at its NEW position
    mx, my = new_x, new_y
    for j in range(n):
        if j == moved_idx:
            continue
        x2, y2 = points[j]
        
        a = my - y2
        b = x2 - mx
        c = mx * y2 - x2 * my
        
        g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
        if g != 0:
            a //= g
            b //= g
            c //= g
        else:
            continue
            
        if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
            a = -a
            b = -b
            c = -c
            
        key = (a, b, c)
        if key not in line_points:
            line_points[key] = set()
        line_points[key].add(moved_idx)
        line_points[key].add(j)
        
    four_lines = 0
    for pts in line_points.values():
        k = len(pts)
        if k == 4:
            four_lines += 1
        elif k >= 5:
            return 0 # Invalid: 5 or more collinear
            
    return four_lines

def _projective_init():
    """
    Generate an initial configuration inspired by the affine plane of order 4.
    The affine plane AG(2,4) has 16 points and 20 lines of 4 points.
    We map this structure to integer coordinates.
    """
    # AG(2,4) points: (x,y) where x,y in {0,1,2,3}
    # But we need to map to integers with good line structure.
    # Use a different approach: start with a structured grid and perturb.
    
    # Try a 4x4 grid with some structured perturbation
    scale = 4
    base_points = [[i * scale, j * scale] for i in range(4) for j in range(4)]
    
    # Add small structured perturbations to break symmetries while preserving some lines
    random.seed(42)
    points = []
    for i in range(4):
        for j in range(4):
            x = i * scale + random.randint(-1, 1)
            y = j * scale + random.randint(-1, 1)
            points.append([x, y])
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Try multiple initializations and pick the best
    best_points = None
    best_score = -1
    
    # Initialization 1: 4x4 grid scaled
    scale = 3
    points1 = [[i * scale, j * scale] for i in range(4) for j in range(4)]
    score1 = _score(points1)
    
    # Initialization 2: Projective-inspired
    points2 = _projective_init()
    score2 = _score(points2)
    
    # Initialization 3: Another structured grid
    scale2 = 5
    points3 = [[i * scale2, j * scale2] for i in range(4) for j in range(4)]
    score3 = _score(points3)
    
    # Pick the best initialization
    inits = [(points1, score1), (points2, score2), (points3, score3)]
    inits.sort(key=lambda x: x[1], reverse=True)
    points = [p[:] for p in inits[0][0]]
    current_score = inits[0][1]
    best_points = [p[:] for p in points]
    best_score = current_score
    
    n = 16
    max_coord = 1000
    
    # Simulated Annealing parameters
    T = 8.0
    T_min = 0.05
    cooling_rate = 0.9998
    
    stall_counter = 0
    max_stall = 15000
    
    # Move strategies
    def generate_move(idx, old_x, old_y):
        """Generate a new position for point idx."""
        r = random.random()
        if r < 0.4:
            # Small step
            step = random.choice([1, -1, 2, -2, 3, -3, 4, -4])
            dx = step
            dy = random.randint(-abs(step), abs(step)) if step != 0 else random.choice([1, -1])
        elif r < 0.7:
            # Medium step
            step = random.randint(-15, 15)
            dx = step
            dy = random.randint(-abs(step), abs(step)) if step != 0 else random.choice([1, -1])
        elif r < 0.9:
            # Large jump
            step = random.randint(-40, 40)
            dx = step
            dy = random.randint(-abs(step), abs(step)) if step != 0 else random.choice([1, -1])
        else:
            # Very large jump
            dx = random.randint(-100, 100)
            dy = random.randint(-100, 100)
        
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Check bounds
        if not (-max_coord <= new_x <= max_coord and -max_coord <= new_y <= max_coord):
            return None
            
        return [new_x, new_y]
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        old_x, old_y = points[idx]
        
        # Generate a new position
        new_point = generate_move(idx, old_x, old_y)
        
        if new_point is None:
            continue
            
        # Apply move
        points[idx] = new_point
        new_score = _incremental_score(points, idx, old_x, old_y, new_point[0], new_point[1])
        
        if new_score is None:
            # Invalid: revert
            points[idx] = [old_x, old_y]
            stall_counter += 1
            continue
            
        if new_score > current_score:
            # Accept better move
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                stall_counter = 0
            else:
                stall_counter += 1
        else:
            # Accept worse move with probability exp(-delta/T)
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                # If score is 0 (invalid: 5-collinear) and current is > 0, reject strongly
                if new_score == 0 and current_score > 0:
                    accept = random.random() < math.exp(-5.0 / T)
                else:
                    accept = random.random() < math.exp(-delta / T)
                    
            if accept:
                stall_counter += 1
            else:
                # Revert
                points[idx] = [old_x, old_y]
                stall_counter += 1
                
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Restart logic
        if stall_counter > max_stall:
            T = 6.0
            stall_counter = 0
            # 40% chance to reset to best, 30% randomize one point, 30% keep current
            r = random.random()
            if r < 0.4:
                points = [p[:] for p in best_points]
                current_score = best_score
            elif r < 0.7:
                # Randomize one point significantly
                idx2 = random.randint(0, n - 1)
                points[idx2] = [random.randint(-50, 50), random.randint(-50, 50)]
                # Ensure distinct
                for _ in range(100):
                    if len(set(map(tuple, points))) == n:
                        break
                    points[idx2] = [random.randint(-50, 50), random.randint(-50, 50)]
                current_score = _score(points)
                if current_score < 0:
                    points = [p[:] for p in best_points]
                    current_score = best_score
            # else: keep current state, just reheat
    
    return best_points

def _distinct(points):
    return len(set(map(tuple, points))) == len(points)
