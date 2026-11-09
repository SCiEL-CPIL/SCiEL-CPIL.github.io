import random
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit

    def normalize_line(A, B, C):
        g = math.gcd(A, math.gcd(B, C))
        if g != 0:
            A //= g
            B //= g
            C //= g
        if A < 0:
            A, B, C = -A, -B, -C
        elif A == 0 and B < 0:
            A, B, C = -A, -B, -C
        return (A, B, C)

    def evaluate(points):
        line_points = defaultdict(set)
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                A = y1 - y2
                B = x2 - x1
                C = -(A * x1 + B * y1)
                key = normalize_line(A, B, C)
                line_points[key].add(i)
                line_points[key].add(j)
        
        score = 0
        for key, pts in line_points.items():
            if len(pts) > 4:
                return -1
            elif len(pts) == 4:
                score += 1
        return score

    # Initial configuration: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = evaluate(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    occupied = set(map(tuple, current_points))
    
    # Local search with small moves first, then larger
    # We'll intermix small and large moves
    while time.time() < deadline:
        # Pick a random point
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Determine move size
        # 50% chance small move, 50% chance large move
        if random.random() < 0.5:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        else:
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
            
        if new_x == old_x and new_y == old_y:
            continue
            
        if (new_x, new_y) in occupied:
            continue
            
        # Try the move
        current_points[idx] = [new_x, new_y]
        occupied.remove((old_x, old_y))
        occupied.add((new_x, new_y))
        
        new_score = evaluate(current_points)
        
        if new_score > current_score:
            best_points = [p[:] for p in current_points]
            best_score = new_score
            current_score = new_score
        elif new_score == current_score and random.random() < 0.2:
            # Accept neutral moves to explore
            current_score = new_score
        else:
            # Revert
            current_points[idx] = [old_x, old_y]
            occupied.remove((new_x, new_y))
            occupied.add((old_x, old_y))
            
        if best_score >= 15:
            break

    return best_points
