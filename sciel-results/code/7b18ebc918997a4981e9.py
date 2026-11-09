import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Initial 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        if A == 0 and B == 0:
            return None
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
        # Normalize sign
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def score(points):
        line_counts = {}
        for i in range(16):
            for j in range(i+1, 16):
                key = get_line_key(points[i], points[j])
                if key:
                    line_counts[key] = line_counts.get(key, 0) + 1
        
        four_lines = 0
        for count in line_counts.values():
            if count == 6: # 4 points -> 6 pairs
                four_lines += 1
            elif count > 6:
                # 5 or more points
                return -1000
        
        return four_lines

    best_score = score(points)
    best_points = [p[:] for p in points]
    
    # Local search
    while time.time() < deadline:
        improved = False
        # Shuffle points to avoid bias
        indices = list(range(16))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            orig_x, orig_y = points[i]
            # Try moving point i
            # Search a box around the point
            # Increase search radius if stuck?
            radius = 5
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    # Check distinctness
                    is_distinct = True
                    for j in range(16):
                        if j != i and points[j][0] == new_x and points[j][1] == new_y:
                            is_distinct = False
                            break
                    if not is_distinct:
                        continue
                    
                    points[i] = [new_x, new_y]
                    s = score(points)
                    if s > best_score:
                        best_score = s
                        best_points = [p[:] for p in points]
                        improved = True
                    else:
                        points[i] = [orig_x, orig_y]
        
        if not improved:
            # Perturb a random point to escape local maximum
            idx = random.randint(0, 15)
            points[idx] = [random.randint(-20, 20), random.randint(-20, 20)]
            # Ensure distinct
            for _ in range(100):
                distinct = True
                for i in range(16):
                    for j in range(i+1, 16):
                        if points[i] == points[j]:
                            distinct = False
                            points[i] = [random.randint(-20, 20), random.randint(-20, 20)]
                            break
                    if not distinct:
                        break
                if distinct:
                    break
            s = score(points)
            if s > best_score:
                best_score = s
                best_points = [p[:] for p in points]
                
    return best_points
