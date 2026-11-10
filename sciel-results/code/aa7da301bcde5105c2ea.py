"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

def count_4_lines(points):
    """Count lines with exactly 4 points from the given set of 16 points."""
    n = len(points)
    if n != 16:
        return 0
    
    # Check for any 5 collinear points
    # For each pair, count how many other points are on the same line
    line_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize the direction vector
            if dx == 0:
                key = (0, 1, x1)  # vertical line at x=x1
            elif dy == 0:
                key = (1, 0, y1)  # horizontal line at y=y1
            else:
                # Normalize by GCD
                from math import gcd
                g = gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                # Ensure consistent sign
                if dx_n < 0:
                    dx_n = -dx_n
                    dy_n = -dy_n
                # Line: dy_n*x - dx_n*y = c
                c = dy_n * x1 - dx_n * y1
                key = (dx_n, dy_n, c)
            
            if key not in line_counts:
                line_counts[key] = set()
            line_counts[key].add(i)
            line_counts[key].add(j)
    
    # Check for 5+ collinear
    for key, pts in line_counts.items():
        if len(pts) >= 5:
            return -1  # Invalid
    
    # Count lines with exactly 4 points
    count = 0
    for key, pts in line_counts.items():
        if len(pts) == 4:
            count += 1
    
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with the 4x4 grid as baseline
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = count_4_lines(best_points)
    
    # Try local search: perturb points and see if we improve
    import random
    random.seed(42)
    
    current = [p[:] for p in best_points]
    current_score = best_score
    
    # Try random restarts and local moves
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random restart: generate random points
        if restart > 0:
            current = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(16)]
            # Ensure distinct
            while len(set(tuple(p) for p in current)) < 16:
                idx = random.randint(0, 15)
                current[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
        
        current_score = count_4_lines(current)
        if current_score < 0:
            continue
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current]
        
        # Local search: try moving one point at a time
        for _ in range(500):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, 15)
            old = current[idx][:]
            # Try small perturbations
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old[0] + dx
                    new_y = old[1] + dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    current[idx] = [new_x, new_y]
                    # Check distinct
                    if len(set(tuple(p) for p in current)) < 16:
                        continue
                    s = count_4_lines(current)
                    if s > current_score:
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_points = [p[:] for p in current]
                    else:
                        current[idx] = old
    
    return best_points
# EVOLVE-BLOCK-END
