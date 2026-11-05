import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def evaluate(points):
        """
        Returns (score, is_valid).
        score: number of lines with exactly 4 points.
        is_valid: False if any line has >= 5 points.
        """
        lines = {}
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue # Duplicate points, shouldn't happen
                dx /= g
                dy /= g
                
                # Canonical direction
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                
                # Line coefficients Ax + By + C = 0
                A = dy
                B = -dx
                C = -(A * x1 + B * y1)
                
                # Normalize A, B, C
                g2 = gcd(abs(int(A)), abs(int(B)))
                if g2 == 0:
                    g2 = 1
                A /= g2
                B /= g2
                C /= g2
                
                # Ensure A > 0 or (A == 0 and B > 0)
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                    
                # Use integer keys for hashing
                key = (int(A), int(B), int(C))
                
                if key not in lines:
                    lines[key] = set([i, j])
                else:
                    lines[key].add(i)
                    lines[key].add(j)
        
        score = 0
        for key, pts in lines.items():
            if len(pts) >= 5:
                return 0, False
            if len(pts) == 4:
                score += 1
        return score, True

    def get_grid(scale_x=1, scale_y=1, offset_x=0, offset_y=0, x_vals=None, y_vals=None):
        if x_vals is None:
            x_vals = [0, 1, 2, 3]
        if y_vals is None:
            y_vals = [0, 1, 2, 3]
        
        points = []
        for x in x_vals:
            for y in y_vals:
                points.append([int(x * scale_x + offset_x), int(y * scale_y + offset_y)])
        return points

    # Initial candidates
    candidates = []
    
    # Standard Grid
    candidates.append(get_grid())
    
    # Stretched Grids (Geometric)
    candidates.append(get_grid(x_vals=[0, 1, 2, 4], y_vals=[0, 1, 2, 4]))
    candidates.append(get_grid(x_vals=[0, 1, 3, 6], y_vals=[0, 1, 3, 6]))
    candidates.append(get_grid(x_vals=[0, 1, 2, 5], y_vals=[0, 1, 2, 5]))
    
    # Random Candidates
    for _ in range(20):
        pts = []
        for _ in range(16):
            pts.append([random.randint(-100, 100), random.randint(-100, 100)])
        candidates.append(pts)

    best_score = 0
    best_points = candidates[0]
    current_points = candidates[0]
    current_score, _ = evaluate(current_points)
    best_score = current_score

    # Local Search
    while time.time() < deadline:
        # Random Restart Probability
        if random.random() < 0.05:
            new_points = []
            for _ in range(16):
                new_points.append([random.randint(-1000, 1000), random.randint(-1000, 1000)])
            score, valid = evaluate(new_points)
            if valid and score > current_score:
                current_points = new_points
                current_score = score
            elif valid and score == current_score and random.random() < 0.5:
                current_points = new_points
                current_score = score
        else:
            # Local Move
            new_points = [p[:] for p in current_points]
            idx = random.randint(0, 15)
            
            # Random perturbation
            move_type = random.random()
            if move_type < 0.5:
                # Small step
                new_points[idx][0] += random.randint(-2, 2)
                new_points[idx][1] += random.randint(-2, 2)
            else:
                # Random jump within bounds
                new_points[idx][0] = random.randint(-1000, 1000)
                new_points[idx][1] = random.randint(-1000, 1000)
            
            # Check distinctness
            # Simple check: ensure new point is not in other points
            is_distinct = True
            for i, p in enumerate(new_points):
                if i == idx: continue
                if p[0] == new_points[idx][0] and p[1] == new_points[idx][1]:
                    is_distinct = False
                    break
            if not is_distinct:
                continue

            score, valid = evaluate(new_points)
            
            if valid:
                if score > current_score:
                    current_points = new_points
                    current_score = score
                elif score == current_score:
                    if random.random() < 0.5:
                        current_points = new_points
                else:
                    # Accept worse moves to escape local optima
                    if random.random() < 0.1:
                        current_points = new_points
                        current_score = score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points

    return best_points
