"""
Program to find 16 integer points with many four-point lines and no five collinear.
Uses a dual-space CSP approach combined with local search refinement.
"""
import random
import itertools
from collections import defaultdict
from math import gcd

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def normalize_line_from_points(p1, p2):
    """
    Given two points, return the canonical line representation (A, B, C)
    such that A*x + B*y = C for all points on the line.
    (A, B) is the normalized normal vector, C is normalized accordingly.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    # Normal vector is (dy, -dx)
    A = dy
    B = -dx
    
    g = _gcd(abs(A), abs(B))
    if g > 0:
        A //= g
        B //= g
    
    # Normalize sign: make first non-zero component positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
    
    C = A * x1 + B * y1
    
    return (A, B, C)

def count_four_lines(points):
    """
    Count lines with exactly 4 points.
    Return 0 if any line has 5+ points (violation).
    """
    n = len(points)
    if n < 4:
        return 0
    
    # Map each line to the set of point indices on it
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            key = normalize_line_from_points(points[i], points[j])
            line_map[key].add(i)
            line_map[key].add(j)
    
    count = 0
    for key, pt_set in line_map.items():
        if len(pt_set) >= 5:
            return 0
        if len(pt_set) == 4:
            count += 1
    
    return count

def construct():
    random.seed(12345)
    
    # Start with a 4x4 grid - this gives many 4-point lines
    # Grid points: (0,0), (1,0), (2,0), (3,0), (0,1), ..., (3,3)
    # Horizontal lines: 4 (y=0,1,2,3) each with 4 points
    # Vertical lines: 4 (x=0,1,2,3) each with 4 points
    # Diagonal lines: various
    # Total 4-point lines in 4x4 grid: let's compute
    
    points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = count_four_lines(points)
    best_points = [p[:] for p in points]
    
    # Multiple restarts with different initial configurations
    all_initializations = []
    
    # 4x4 grid
    all_initializations.append([[x, y] for x in range(4) for y in range(4)])
    
    # Slightly perturbed grids
    for attempt in range(20):
        pts = [[random.randint(-50, 50), random.randint(-50, 50)] for _ in range(16)]
        # Remove duplicates
        unique_pts = []
        seen = set()
        for p in pts:
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                unique_pts.append(p)
        while len(unique_pts) < 16:
            new_p = [random.randint(-50, 50), random.randint(-50, 50)]
            key = (new_p[0], new_p[1])
            if key not in seen:
                seen.add(key)
                unique_pts.append(new_p)
        all_initializations.append(unique_pts[:16])
    
    best_score = 0
    best_points = None
    
    for init_points in all_initializations:
        current_points = [p[:] for p in init_points]
        current_score = count_four_lines(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
        
        # Simulated annealing
        T = 30.0
        T_min = 0.5
        cooling = 0.998
        max_iters = 30000
        no_improve = 0
        plateau_limit = 5000
        
        for it in range(max_iters):
            if T < T_min:
                break
            if no_improve > plateau_limit:
                break
            
            # Pick a random point to move
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Try a random new position - mix of small and large moves
            if random.random() < 0.7:
                # Small move
                move_range = random.choice([1, 2, 3, 5])
            else:
                # Larger move
                move_range = random.choice([10, 20, 50, 100])
            
            new_x = old_x + random.randint(-move_range, move_range)
            new_y = old_y + random.randint(-move_range, move_range)
            
            # Clamp to bounds
            new_x = max(-1000, min(1000, new_x))
            new_y = max(-1000, min(1000, new_y))
            
            # Check if point is already taken
            if any(current_points[i] == [new_x, new_y] for i in range(16) if i != idx):
                continue
            
            current_points[idx] = [new_x, new_y]
            new_score = count_four_lines(current_points)
            
            delta = new_score - current_score
            if delta > 0 or (delta == 0 and random.random() < 0.3) or random.random() < pow(2.718281828, max(delta, -10) / T):
                if delta > 0:
                    current_score = new_score
                    no_improve = 0
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current_points]
                else:
                    no_improve += 1
            else:
                current_points[idx] = [old_x, old_y]
                no_improve += 1
            
            T *= cooling
        
        # After SA, do local refinement: try all possible single-point moves
        improved = True
        while improved:
            improved = False
            for idx in range(16):
                best_local_score = count_four_lines(current_points)
                best_local_pos = current_points[idx][:]
                
                # Try nearby positions
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = current_points[idx][0] + dx
                        new_y = current_points[idx][1] + dy
                        if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
                            continue
                        if any(current_points[i] == [new_x, new_y] for i in range(16) if i != idx):
                            continue
                        
                        current_points[idx] = [new_x, new_y]
                        local_score = count_four_lines(current_points)
                        
                        if local_score > best_local_score:
                            best_local_score = local_score
                            best_local_pos = [new_x, new_y]
                            improved = True
                
                if improved:
                    current_points[idx] = best_local_pos[:]
                    if best_local_score > best_score:
                        best_score = best_local_score
                        best_points = [p[:] for p in current_points]
                else:
                    current_points[idx] = best_local_pos[:]
    
    return best_points
