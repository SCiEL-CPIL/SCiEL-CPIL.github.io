"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd
from itertools import combinations

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Map from normalized line (A, B, C) to count of pairs on that line
    line_pairs = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Line through (x1,y1) and (x2,y2): dy*(x - x1) - dx*(y - y1) = 0
            # => dy*x - dx*y + (dx*y1 - dy*x1) = 0
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            # Normalize: divide by gcd
            if A == 0 and B == 0:
                continue  # shouldn't happen for distinct points
            g = gcd(abs(A), gcd(abs(B), abs(C)))
            if g > 0:
                A //= g
                B //= g
                C //= g
            else:
                g = 1
            # Make A positive, or if A==0 make B positive, or if both 0 make C positive
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            line_pairs[key] = line_pairs.get(key, 0) + 1
    # A line with k points contributes C(k,2) pairs.
    # Ordinary lines are those with exactly 2 points, i.e., pair count == 2.
    count = 0
    for k, v in line_pairs.items():
        if v == 2:
            count += 1
    return count


def construct():
    deadline = time.time() + 130.0
    best_points = None
    best_score = float('inf')
    
    # Initial: 3x4 grid
    base = [[x, y] for x in range(4) for y in range(3)]
    best_score = _count_ordinary_lines(base)
    best_points = [list(p) for p in base]
    
    random.seed(42)
    
    # Simulated annealing with random restarts
    start_t = time.time()
    restart = 0
    while time.time() < deadline - 5.0:
        restart += 1
        if restart > 1:
            # Random restart: random initial configuration
            points = []
            for _ in range(12):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if [x, y] not in points:
                    points.append([x, y])
        else:
            points = [list(p) for p in base]
        
        curr_score = _count_ordinary_lines(points)
        if curr_score < best_score:
            best_score = curr_score
            best_points = [list(p) for p in points]
        
        if best_score == 6:
            break
        
        T = 10.0
        T_min = 0.1
        T_dec = 0.9995
        
        max_steps = 5000
        for step in range(max_steps):
            if time.time() > deadline - 3.0:
                break
            
            # Pick a random point
            idx = random.randint(0, 11)
            old_x, old_y = points[idx]
            
            # Random move: change one or both coordinates
            move_type = random.random()
            if move_type < 0.3:
                # Small perturbation
                delta = random.randint(-5, 5)
                if random.random() < 0.5:
                    new_x = old_x + delta
                    new_y = old_y
                else:
                    new_x = old_x
                    new_y = old_y + delta
            elif move_type < 0.6:
                # Medium perturbation
                delta = random.randint(-50, 50)
                if random.random() < 0.5:
                    new_x = old_x + delta
                    new_y = old_y
                else:
                    new_x = old_x
                    new_y = old_y + delta
            else:
                # Large jump
                new_x = random.randint(-500, 500)
                new_y = random.randint(-500, 500)
            
            # Check bounds and distinctness
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            new_pt = [new_x, new_y]
            if any(new_pt == points[i] for i in range(12) if i != idx):
                continue
            
            # Apply move
            points[idx] = new_pt
            new_score = _count_ordinary_lines(points)
            
            # Acceptance criterion
            if new_score <= curr_score:
                curr_score = new_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_points = [list(p) for p in points]
                    if best_score == 6:
                        break
            else:
                # Metropolis acceptance
                import math
                try:
                    prob = math.exp(-(new_score - curr_score) / T)
                except OverflowError:
                    prob = 0.0
                if random.random() < prob:
                    curr_score = new_score
                else:
                    # Revert
                    points[idx] = [old_x, old_y]
            
            T = max(T_min, T * T_dec)
        
        if best_score == 6:
            break
    
    # Final check: ensure not all collinear
    def _all_collinear(pts):
        if len(pts) < 3:
            return True
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        dx = x2 - x1
        dy = y2 - y1
        for i in range(2, len(pts)):
            x, y = pts[i]
            if dx * (y - y1) - dy * (x - x1) != 0:
                return False
        return True
    
    if best_points is None or _all_collinear(best_points):
        # Fallback: use a known non-collinear configuration
        best_points = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1],
                       [3, 0], [3, 1], [0, 2], [1, 2], [2, 2], [3, 2]]
    
    # Scale to ensure integer coords within bounds (already integers)
    # Return as list of [x, y]
    return [[int(p[0]), int(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
