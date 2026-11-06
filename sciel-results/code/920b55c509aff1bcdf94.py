"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score_points(points):
    """Count lines containing exactly 4 points; return -inf if any 5 collinear."""
    n = len(points)
    if n != 16:
        return -10**9
    # Check distinctness
    if len(set(tuple(p) for p in points)) != 16:
        return -10**9
    # Check bounds
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -10**9
    
    # For each pair, compute the line and count how many of the 16 points lie on it.
    # A line is identified by (A, B, C) where Ax + By + C = 0 with gcd(A,B,C)=1 and sign convention.
    # To avoid duplicates, we only consider pairs (i, j) with i < j, and normalize the line.
    
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2): A*x + B*y + C = 0
            # A = y2 - y1, B = x1 - x2, C = x2*y1 - x1*y2
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue  # duplicate points, shouldn't happen
            # Normalize
            from math import gcd
            g = gcd(abs(A), abs(B))
            if g != 0:
                A //= g
                B //= g
                C //= g
            else:
                # A and B are both 0, handled above
                continue
            # Sign convention: make the first non-zero of (A, B, C) positive
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in line_counts:
                line_counts[key] = 0
            # We'll count later; for now just mark the line exists.
    
    # Now for each unique line, count how many points lie on it.
    count4 = 0
    for (A, B, C) in line_counts:
        cnt = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                cnt += 1
        if cnt >= 5:
            return -10**9
        if cnt == 4:
            count4 += 1
    return count4


def _random_point():
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]


def _local_search(points, time_limit):
    """Simulated annealing / hill climbing to improve the configuration."""
    current = [p[:] for p in points]
    current_score = _score_points(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    start = time.time()
    T = 10.0
    cooling = 0.9995
    
    while time.time() - start < time_limit:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old = current[idx][:]
        # Perturb: move by a small random delta
        dx = random.randint(-50, 50)
        dy = random.randint(-50, 50)
        new_x = old[0] + dx
        new_y = old[1] + dy
        # Clamp to bounds
        new_x = max(-1000, min(1000, new_x))
        new_y = max(-1000, min(1000, new_y))
        if new_x == old[0] and new_y == old[1]:
            continue
        current[idx] = [new_x, new_y]
        new_score = _score_points(current)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = [p[:] for p in current]
        else:
            # Accept with probability exp((new_score - current_score)/T)
            if new_score >= 0 and random.random() < (2.718281828 ** ((new_score - current_score) / T)):
                current_score = new_score
            else:
                current[idx] = old
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return best, best_score


def construct():
    deadline = time.time() + 140  # safety margin for 150s limit
    
    # Start with the 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    best_points = grid
    best_score = _score_points(grid)
    
    # Try multiple random restarts with local search
    while time.time() < deadline:
        # Generate a random starting configuration
        # Mix of strategies: sometimes start near grid, sometimes fully random
        if random.random() < 0.3:
            # Start from grid with small perturbations
            start = [[x + random.randint(-5, 5), y + random.randint(-5, 5)] 
                     for x in range(4) for y in range(4)]
        else:
            # Fully random
            start = [_random_point() for _ in range(16)]
        
        # Ensure distinctness
        seen = set()
        valid = True
        for p in start:
            t = tuple(p)
            if t in seen:
                valid = False
                break
            seen.add(t)
        if not valid:
            continue
        
        # Check bounds
        if any(abs(p[0]) > 1000 or abs(p[1]) > 1000 for p in start):
            continue
        
        # Local search with remaining time
        remaining = deadline - time.time()
        if remaining < 2:
            break
        # Use a portion of remaining time for each restart
        search_time = min(remaining - 1, 5.0)
        if search_time < 0.5:
            break
        
        pts, score = _local_search(start, search_time)
        if score > best_score:
            best_score = score
            best_points = pts
    
    return best_points
# EVOLVE-BLOCK-END
