import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # Grid bounds for search
    GRID_MIN = -10
    GRID_MAX = 10
    
    # Initial configuration: 4x4 grid centered at 0
    # This gives score 10
    def initial_config():
        pts = []
        for x in range(-2, 2):
            for y in range(-2, 2):
                pts.append([x, y])
        return pts

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        # A = y1 - y2
        # B = x2 - x1
        # C = x1*y2 - x2*y1
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        # Normalize by GCD to get unique key for parallel lines
        if A == 0 and B == 0:
            return None # Should not happen for distinct points
            
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
        if g == 0:
            g = 1
            
        A //= g
        B //= g
        C //= g
        
        # Ensure consistent sign (e.g., first non-zero is positive)
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)

    def evaluate(points):
        """
        Returns (score, is_valid)
        score: number of lines with exactly 4 points
        is_valid: False if any line has >= 5 points or points are not distinct
        """
        n = len(points)
        # Check distinctness
        if len(set(tuple(p) for p in points)) != n:
            return 0, False
            
        line_points = defaultdict(set)
        
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key is None:
                    return 0, False
                line_points[key].add(i)
                line_points[key].add(j)
        
        score = 0
        for key, pts in line_points.items():
            count = len(pts)
            if count >= 5:
                return 0, False
            if count == 4:
                score += 1
                
        return score, True

    def evaluate_fast(points):
        """
        Faster evaluation using a different approach.
        Returns (score, is_valid)
        """
        n = len(points)
        # Check distinctness
        if len(set(tuple(p) for p in points)) != n:
            return 0, False
            
        # For each pair, compute line key and count points on that line
        line_counts = defaultdict(int)
        line_point_sets = defaultdict(set)
        
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key is None:
                    return 0, False
                line_point_sets[key].add(i)
                line_point_sets[key].add(j)
        
        score = 0
        for key, pts in line_point_sets.items():
            count = len(pts)
            if count >= 5:
                return 0, False
            if count == 4:
                score += 1
                
        return score, True

    def try_improve_4point_line(points, idx):
        """
        Try to find a better position for point idx that creates or maintains 4-point lines.
        """
        best_new_pos = None
        best_delta = 0
        current_score, _ = evaluate(points)
        
        # Try positions near the current point
        cx, cy = points[idx]
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if abs(nx) > GRID_MAX or abs(ny) > GRID_MAX:
                    continue
                new_points = [p[:] for p in points]
                new_points[idx] = [nx, ny]
                new_score, is_valid = evaluate(new_points)
                if is_valid:
                    delta = new_score - current_score
                    if delta > best_delta:
                        best_delta = delta
                        best_new_pos = [nx, ny]
        
        return best_new_pos, best_delta

    # Simulated Annealing
    best_points = initial_config()
    best_score, _ = evaluate(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score, _ = evaluate(current_points)
    
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    step_count = 0
    while time.time() < deadline:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.6:
            # Shift a point
            idx = random.randint(0, 15)
            dx = random.randint(-4, 4)
            dy = random.randint(-4, 4)
            if dx == 0 and dy == 0:
                continue
            new_points = [p[:] for p in current_points]
            new_points[idx][0] += dx
            new_points[idx][1] += dy
            
            # Check bounds
            if abs(new_points[idx][0]) > 1000 or abs(new_points[idx][1]) > 1000:
                continue
            # Check if it stays within our search grid to maintain density
            if abs(new_points[idx][0]) > GRID_MAX or abs(new_points[idx][1]) > GRID_MAX:
                 # Allow escape less frequently
                 if random.random() > 0.15:
                     continue
                     
        elif move_type < 0.85:
            # Try to improve a specific point to create 4-point lines
            idx = random.randint(0, 15)
            best_new_pos, best_delta = try_improve_4point_line(current_points, idx)
            if best_new_pos is None or best_delta <= 0:
                # Fall back to random shift
                dx = random.randint(-4, 4)
                dy = random.randint(-4, 4)
                if dx == 0 and dy == 0:
                    continue
                new_points = [p[:] for p in current_points]
                new_points[idx][0] += dx
                new_points[idx][1] += dy
                if abs(new_points[idx][0]) > GRID_MAX or abs(new_points[idx][1]) > GRID_MAX:
                    if random.random() > 0.15:
                        continue
            else:
                new_points = [p[:] for p in current_points]
                new_points[idx] = best_new_pos
                
        else:
            # Replace a point with a random point in grid
            idx = random.randint(0, 15)
            new_points = [p[:] for p in current_points]
            new_points[idx] = [random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX)]

        new_score, is_valid = evaluate(new_points)
        
        if not is_valid:
            # Reject invalid moves
            pass
        else:
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_points = new_points
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                    # Reset temperature on improvement
                    T = max(T, 5.0)
        
        T *= cooling
        if T < min_T:
            T = min_T
            
        step_count += 1
        # Periodic restart if stuck
        if step_count % 5000 == 0 and best_score <= 10:
            # Try a fresh random configuration
            current_points = []
            for _ in range(16):
                current_points.append([random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX)])
            current_score, is_valid = evaluate(current_points)
            if not is_valid:
                current_points = [p[:] for p in best_points]
                current_score = best_score
            T = 10.0

    return best_points

# EVOLVE-BLOCK-END
