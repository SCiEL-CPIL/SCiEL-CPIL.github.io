import time
import random
import math

def _score_points(points):
    """
    Calculate the number of lines containing exactly 4 points
    and check if any line contains 5 or more points.
    
    Returns:
        tuple: (num_exact_4_lines, has_5_collinear)
    """
    n = 16
    lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction vector (dx, dy)
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            
            # Canonical direction
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            # Line equation: A*x + B*y + C = 0
            # Normal vector (A, B) is perpendicular to direction (dx_n, dy_n)
            A = dy_n
            B = -dx_n
            C = -(A * x1 + B * y1)
            
            # Normalize line coefficients (A, B, C)
            g2 = math.gcd(abs(A), abs(B))
            if g2 > 1:
                A //= g2
                B //= g2
                C //= g2
            
            # Canonical sign
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    exact_4 = 0
    has_5 = False
    for pts_on_line in lines.values():
        cnt = len(pts_on_line)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
            # Early exit if we just need to know validity, but we need exact_4 count too.
            # However, if has_5 is true, the configuration is invalid for the objective.
    
    return exact_4, has_5

def construct():
    deadline = time.time() + 145
    
    # Initial configuration: 4x4 grid
    # This configuration has 10 lines of exactly 4 points (4 rows, 4 cols, 2 diags)
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    # Scale to center and spread out to allow room for movement
    # The grid is 0..3. Let's place it at -6..3 to have space.
    # Or just keep it small and allow moves.
    # Let's use a larger grid to start? No, 4x4 is the standard dense packing.
    # Let's start with the 4x4 grid shifted.
    best_points = [p[:] for p in initial_points]
    best_score, _ = _score_points(best_points)
    
    rng = random.Random(42)
    
    # Simulated Annealing
    current_points = [p[:] for p in best_points]
    current_score, _ = _score_points(current_points)
    
    T = 1.0
    min_T = 0.01
    cooling = 0.999
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 5:
        if no_improve_count > max_no_improve:
            # Reheat or restart
            T = 1.0
            no_improve_count = 0
            # Randomize current points slightly to escape local minimum
            for _ in range(rng.randint(5, 15)):
                idx = rng.randint(0, 15)
                old_x, old_y = current_points[idx]
                new_x = old_x + rng.randint(-10, 10)
                new_y = old_y + rng.randint(-10, 10)
                if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                    # Check distinctness
                    new_pt = [new_x, new_y]
                    if new_pt not in current_points:
                        current_points[idx] = new_pt
            
            current_score, has_5 = _score_points(current_points)
            if has_5:
                # If restart led to invalid state, revert to best
                current_points = [p[:] for p in best_points]
                current_score, _ = _score_points(current_points)
                T = 1.0
                continue
        
        idx = rng.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Choose move size
        # Larger moves when T is high or stuck
        move_size = 5 if T > 0.5 else 3
        if no_improve_count > 500:
            move_size = 15
            
        new_x = old_x + rng.randint(-move_size, move_size)
        new_y = old_y + rng.randint(-move_size, move_size)
        
        # Boundary check
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        new_pt = [new_x, new_y]
        
        # Distinctness check
        # Optimization: Only check if new_pt is in the list
        # Since n=16, linear scan is fast
        if new_pt in current_points:
            continue
            
        # Apply move
        current_points[idx] = new_pt
        
        new_score, has_5 = _score_points(current_points)
        
        if has_5:
            # Invalid move, revert
            current_points[idx] = [old_x, old_y]
            no_improve_count += 1
            T *= cooling
            if T < min_T: T = min_T
            continue
            
        # Acceptance criterion
        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        elif new_score == current_score:
            # Accept equal score with some probability to explore plateau
            if rng.random() < 0.5:
                no_improve_count += 1
            else:
                current_points[idx] = [old_x, old_y]
                no_improve_count += 1
        else:
            # Worse score
            # Simulated annealing acceptance
            delta = new_score - current_score
            if T > min_T:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    current_score = new_score
                    no_improve_count += 1
                else:
                    current_points[idx] = [old_x, old_y]
                    no_improve_count += 1
            else:
                current_points[idx] = [old_x, old_y]
                no_improve_count += 1
        
        T *= cooling
        if T < min_T: T = min_T

    return best_points
