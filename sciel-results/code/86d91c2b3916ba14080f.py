import math
import random
import time

def construct():
    start_time = time.time()
    deadline = 115.0
    
    best_score = -1
    best_points = None
    
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    def evaluate(points):
        seen = set()
        for p in points:
            x, y = p
            if x < -1000 or x > 1000 or y < -1000 or y > 1000:
                return False, 0
            if (x, y) in seen:
                return False, 0
            seen.add((x, y))
            
        n = len(points)
        lines = {}
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                if g == 0: continue
                A = dy // g
                B = -(dx // g)
                C = -(A * x1 + B * y1)
                
                g2 = math.gcd(abs(A), abs(B))
                if g2 > 0:
                    A //= g2
                    B //= g2
                    C //= g2
                
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                    
                key = (A, B, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
                
        score = 0
        lines_3 = []
        for key, pts in lines.items():
            c = len(pts)
            if c >= 5:
                return False, 0
            if c == 4:
                score += 1
            if c == 3:
                lines_3.append((key, pts))
                
        return True, score, lines_3

    def get_candidates(points, lines_3):
        """
        Generate candidate points to fill 3-point lines to make them 4-point lines.
        """
        candidates = []
        occupied = set((p[0], p[1]) for p in points)
        
        for key, pts in lines_3:
            A, B, C = key
            # Line: A x + B y + C = 0
            # Find integer points on this line within bounds
            # We just need one point that is not occupied.
            # To be deterministic or random, we can pick a few.
            # Let's find the range of x
            if A != 0:
                # x = (-B y - C) / A
                # We need (-B y - C) % A == 0
                # Iterate y?
                # Better: parameterize.
                # If A=1, x = -B y - C.
                # Let's just try a few random points on the line.
                pass
            
            # Simpler: Pick 2 points from pts, extend the line.
            p1 = points[list(pts)[0]]
            p2 = points[list(pts)[1]]
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            
            # Points on line: (x1 + t*dx, y1 + t*dy)
            # We want a point not in occupied.
            # Try t = 4, 5, -1, -2, etc.
            for t in range(-5, 10):
                if t == 0 or t == 1: # Skip existing points p1, p2
                    continue
                nx = x1 + t * dx
                ny = y1 + t * dy
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    if (nx, ny) not in occupied:
                        candidates.append([nx, ny])
                        
        return candidates

    # Initialize
    valid, score, lines_3 = evaluate(initial_points)
    if valid:
        best_score = score
        best_points = [p[:] for p in initial_points]
        
    current_points = [p[:] for p in initial_points]
    current_valid, current_score, current_lines_3 = evaluate(current_points)
    
    no_improve_count = 0
    
    while time.time() < deadline:
        if not current_valid:
            # Random restart
            current_points = [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)]
            current_valid, current_score, current_lines_3 = evaluate(current_points)
            no_improve_count = 0
            continue
            
        # Generate candidates
        candidates = get_candidates(current_points, current_lines_3)
        
        if not candidates:
            # Fallback to random move
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            current_points[idx] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
            current_valid, current_score, current_lines_3 = evaluate(current_points)
            if current_valid and current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
            else:
                current_points[idx] = [old_x, old_y]
            no_improve_count += 1
            if no_improve_count > 100:
                # Restart
                current_points = [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)]
                current_valid, current_score, current_lines_3 = evaluate(current_points)
                no_improve_count = 0
            continue
            
        # Pick a random candidate
        new_point = random.choice(candidates)
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # If the point we are moving is already at the candidate, skip
        if old_x == new_point[0] and old_y == new_point[1]:
            continue
            
        current_points[idx] = new_point
        valid, score, lines_3 = evaluate(current_points)
        
        if valid and score > current_score:
            current_score = score
            current_lines_3 = lines_3
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            current_points[idx] = [old_x, old_y]
            no_improve_count += 1
            if no_improve_count > 100:
                # Random kick
                for _ in range(3):
                    k = random.randint(0, 15)
                    current_points[k] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
                current_valid, current_score, current_lines_3 = evaluate(current_points)
                no_improve_count = 0
                
    return best_points
