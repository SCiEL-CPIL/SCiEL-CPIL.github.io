import random
import itertools
import time

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    ordinary = 0
    # Use a set of lines represented by (a,b,c) for ax+by+c=0 normalized
    # Or just check pairs
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Check if any other point k is collinear
            is_ordinary = True
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                # Cross product
                if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                    is_ordinary = False
                    break
            if is_ordinary:
                ordinary += 1
    return ordinary

def construct():
    # Start with 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    best_score = count_ordinary_lines(points)
    
    # Local search
    for _ in range(1000):
        # Pick a random point and move it
        idx = random.randint(0, 11)
        orig_x, orig_y = points[idx]
        # Try moving to a nearby integer coordinate
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                new_x = orig_x + dx
                new_y = orig_y + dy
                if abs(new_x) > 100 or abs(new_y) > 100:
                    continue
                # Check if point is already in set
                if [new_x, new_y] in points:
                    continue
                points[idx] = [new_x, new_y]
                score = count_ordinary_lines(points)
                if score < best_score:
                    best_score = score
                    # Accept move
                else:
                    # Revert
                    points[idx] = [orig_x, orig_y]
        if best_score == 6:
            break
            
    return points
