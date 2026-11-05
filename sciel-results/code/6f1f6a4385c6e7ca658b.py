import time
import random
import math

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    ordinary_count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize the direction vector
            if dx == 0 and dy == 0:
                continue # Should not happen for distinct points
                
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            
            # Ensure consistent sign (e.g., dx > 0 or dx == 0 and dy > 0)
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
                
            # Line key: (dx, dy, x*dy - y*dx) is not quite right for parallel lines.
            # Use (A, B, C) from Ax + By + C = 0
            # A = dy, B = -dx, C = -(dy*x1 - dx*y1)
            A = dy
            B = -dx
            C = -(dy * x1 - dx * y1)
            
            # Normalize (A, B, C)
            g2 = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g2 != 0:
                A //= g2
                B //= g2
                C //= g2
                
            if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Candidate 1: 3x4 Grid (Seed)
    points_seed = [[x, y] for x in range(4) for y in range(3)]
    best_points = points_seed
    best_score = count_ordinary_lines(points_seed)
    
    # Candidate 2: Pappus-like 9 points + 3
    # 3x3 grid
    p3x3 = [[x, y] for x in range(3) for y in range(3)]
    # Add points to break ordinary lines
    # The 3x3 grid has 12 ordinary lines.
    # Let's try adding (3,1), (-1,1), (1,3)
    cand2 = p3x3 + [[3, 1], [-1, 1], [1, 3]]
    score2 = count_ordinary_lines(cand2)
    if score2 < best_score:
        best_score = score2
        best_points = cand2
        
    # Candidate 3: Random Search / Local Search
    # If we have time, run a simple local search
    # We can try to improve best_points
    
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]
    
    current_points = best_points[:]
    current_score = best_score
    
    # Local Search
    while time.time() < deadline:
        improved = False
        # Try moving each point
        for i in range(12):
            if time.time() > deadline:
                break
            orig_x, orig_y = current_points[i]
            # Try some random moves
            for _ in range(10):
                new_x = orig_x + random.randint(-10, 10)
                new_y = orig_y + random.randint(-10, 10)
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                # Check distinctness
                if any(current_points[j] == [new_x, new_y] for j in range(12) if j != i):
                    continue
                old_pt = current_points[i]
                current_points[i] = [new_x, new_y]
                new_score = count_ordinary_lines(current_points)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_points = current_points[:]
                    break
                else:
                    current_points[i] = old_pt
        
        if not improved:
            # Random restart
            current_points = [random_point() for _ in range(12)]
            current_score = count_ordinary_lines(current_points)
            if current_score < best_score:
                best_score = current_score
                best_points = current_points[:]

    return best_points
