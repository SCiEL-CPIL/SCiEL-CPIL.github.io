"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _line_key(p1, p2):
    """Return a normalized key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    return sum(1 for pts in line_points.values() if len(pts) == 2)

def _count_ordinary_fast(points):
    """Faster count using line equation counting."""
    n = len(points)
    line_count = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_count[key] = line_count.get(key, 0) + 1
    # A line with k points appears C(k,2) times in line_count
    # We need to find how many lines have exactly 2 points
    # line_count[key] = C(k,2) where k is number of points on that line
    # C(2,2)=1, C(3,2)=3, C(4,2)=6, C(5,2)=10, etc.
    # So if line_count[key] == 1, it's an ordinary line (2 points)
    # If line_count[key] == 3, it's a 3-point line
    # etc.
    # But we need to be careful: line_count gives the number of pairs, not points
    # For a line with k points, it contributes C(k,2) to the count
    # We want lines where k=2, i.e., C(k,2)=1
    return sum(1 for v in line_count.values() if v == 1)

def _generate_hexagon_config(scale=2):
    """Generate a hexagon-based configuration with integer coordinates.
    
    Uses a regular hexagon approximation with integer coordinates.
    For a regular hexagon, vertices are at angles 0,60,120,180,240,300 degrees.
    With radius r: (r,0), (r/2, r*sqrt(3)/2), etc.
    To get integers, use r=2k: vertices at (2k,0), (k, k*sqrt(3)), ...
    sqrt(3) is irrational, so we approximate.
    
    Alternative: use a "hexagonal lattice" point set.
    """
    # Hexagonal lattice points: use axial coordinates
    # Points in a hexagonal arrangement
    # For 12 points: use a 3x4 or 4x3 hexagonal patch
    
    # Configuration: points on 3 parallel lines with 4 points each
    # Line 1: y=0, x in {-3,-1,1,3}
    # Line 2: y=2, x in {-2,0,2,4}  (shifted)
    # Line 3: y=4, x in {-3,-1,1,3}
    pts = [(-3, 0), (-1, 0), (1, 0), (3, 0),
           (-2, 2), (0, 2), (2, 2), (4, 2),
           (-3, 4), (-1, 4), (1, 4), (3, 4)]
    return [[x, y] for x, y in pts]

def _generate_grid_config(rows=3, cols=4):
    """Generate a grid configuration."""
    return [[x, y] for x in range(cols) for y in range(rows)]

def _generate_line_based_config():
    """Generate a configuration with points on a few lines.
    
    Try: 4 points on line y=0, 4 on line y=1, 4 on line y=2,
    but shifted to create more 3-point lines.
    """
    # Try: 3 lines with 4 points each, with some shared intersection points
    # Line 1: y=0, x in {0,1,2,3}
    # Line 2: y=1, x in {0,1,2,3}  
    # Line 3: y=x, points where y=x and y in {0,1,2,3}: (0,0),(1,1),(2,2),(3,3)
    # But (0,0) is on both line 1 and line 3, (1,1) is on line 2 and line 3
    # This creates overlaps. Let's count distinct points.
    pts = set()
    for x in range(4):
        pts.add((x, 0))  # 4 points on y=0
    for x in range(4):
        pts.add((x, 1))  # 4 points on y=1
    # Add 4 more on y=x, but avoid duplicates
    for x in range(4):
        if (x, x) not in pts:
            pts.add((x, x))
        if len(pts) >= 12:
            break
    # If we don't have 12, add more
    while len(pts) < 12:
        x = random.randint(0, 5)
        y = random.randint(0, 5)
        pts.add((x, y))
    return [[x, y] for x, y in list(pts)[:12]]

def _generate_boroczky_like():
    """Generate a Böröczky-like configuration.
    
    For n=12, the Böröczky construction uses points on a conic.
    A simpler approach: use points that are vertices and edge midpoints
    of a regular polygon, scaled to integers.
    
    Let's try: 6 points forming a hexagon + 6 points at the "dual" positions.
    Hexagon vertices (approximate with integers):
    Using a 30-60-90 triangle approximation:
    (3,0), (1,2), (-1,2), (-3,0), (-1,-2), (1,-2)
    Dual points (intersections of diagonals):
    The main diagonals of a hexagon intersect at the center (0,0) and other points.
    For our hexagon, diagonals connect opposite vertices:
    (3,0)-(-3,0): y=0
    (1,2)-(-1,-2): y=x
    (-1,2)-(1,-2): y=-x
    These all pass through (0,0).
    
    Let's try a different approach: use a 4x3 grid but with one point moved
    to create more 3-point lines.
    """
    # Start with 4x3 grid
    pts = [[x, y] for x in range(4) for y in range(3)]
    return pts

def _valid_move(points, idx, nx, ny):
    """Check if moving points[idx] to (nx, ny) is valid."""
    if abs(nx) > 10**9 or abs(ny) > 10**9:
        return False
    for j in range(len(points)):
        if j != idx and points[j][0] == nx and points[j][1] == ny:
            return False
    return True

def _swap_valid(points, i, j, new_pos_i, new_pos_j):
    """Check if swapping positions of points i and j is valid."""
    if abs(new_pos_i[0]) > 10**9 or abs(new_pos_i[1]) > 10**9:
        return False
    if abs(new_pos_j[0]) > 10**9 or abs(new_pos_j[1]) > 10**9:
        return False
    # Check no other point is at new_pos_i or new_pos_j
    for k in range(len(points)):
        if k != i and k != j:
            if points[k][0] == new_pos_i[0] and points[k][1] == new_pos_i[1]:
                return False
            if points[k][0] == new_pos_j[0] and points[k][1] == new_pos_j[1]:
                return False
    return True

def _initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # Grid configurations
    configs.append(_generate_grid_config(3, 4))
    configs.append(_generate_grid_config(4, 3))
    configs.append(_generate_grid_config(2, 6))
    configs.append(_generate_grid_config(6, 2))
    
    # Hexagon-like
    configs.append(_generate_hexagon_config())
    
    # Line-based
    configs.append(_generate_line_based_config())
    
    # Böröczky-like
    configs.append(_generate_boroczky_like())
    
    # Two vertical lines with 6 points each
    configs.append([[0, y] for y in range(6)] + [[1, y] for y in range(6)])
    
    # Three horizontal lines with 4 points each
    configs.append([[x, y] for y in range(3) for x in range(4)])
    
    # Staircase
    configs.append([[i, i * i % 7] for i in range(12)])
    
    # Random small grid
    rng = random.Random(42)
    used = set()
    pts = []
    while len(pts) < 12:
        x = rng.randint(-4, 4)
        y = rng.randint(-4, 4)
        if (x, y) not in used:
            used.add((x, y))
            pts.append([x, y])
    configs.append(pts)
    
    # Another random
    rng2 = random.Random(123)
    used2 = set()
    pts2 = []
    while len(pts2) < 12:
        x = rng2.randint(-5, 5)
        y = rng2.randint(-5, 5)
        if (x, y) not in used2:
            used2.add((x, y))
            pts2.append([x, y])
    configs.append(pts2)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (50s buffer for 150s limit)
    
    best_points = None
    best_score = float('inf')
    
    # Try multiple structured initial configurations
    initial_configs = _initial_configs()
    
    for cfg_idx, init_cfg in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        current_points = [[p[0], p[1]] for p in init_cfg]
        current_score = _count_ordinary_fast(current_points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [list(p) for p in current_points]
        
        # Local search with simulated annealing and swap-based moves
        T = 1.0  # Temperature
        T_min = 0.01
        cooling = 0.995
        
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Single point move
                idx = random.randint(0, 11)
                cx, cy = current_points[idx]
                
                improved = False
                # Try several random moves with varying step sizes
                for _ in range(15):
                    step = random.choice([1, 2, 3, 5, 8])
                    nx = cx + random.randint(-step, step)
                    ny = cy + random.randint(-step, step)
                    
                    if not _valid_move(current_points, idx, nx, ny):
                        continue
                    
                    old_score = current_score
                    current_points[idx][0] = nx
                    current_points[idx][1] = ny
                    new_score = _count_ordinary_fast(current_points)
                    
                    delta = new_score - old_score
                    
                    if delta < 0 or random.random() < min(1.0, 1.0 / (1.0 + T)) * (1.0 if delta <= 0 else math.exp(-delta / T)):
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [list(p) for p in current_points]
                        break
                    else:
                        # Revert
                        current_points[idx][0] = cx
                        current_points[idx][1] = cy
                
                if not improved:
                    T *= cooling
                    
            else:
                # Swap two points
                i, j = random.sample(range(12), 2)
                old_i = [current_points[i][0], current_points[i][1]]
                old_j = [current_points[j][0], current_points[j][1]]
                
                # Try swapping with small perturbations
                improved = False
                for _ in range(10):
                    step = random.choice([1, 2, 3])
                    new_i = [old_i[0] + random.randint(-step, step), old_i[1] + random.randint(-step, step)]
                    new_j = [old_j[0] + random.randint(-step, step), old_j[1] + random.randint(-step, step)]
                    
                    if not _swap_valid(current_points, i, j, new_i, new_j):
                        continue
                    
                    old_score = current_score
                    current_points[i] = new_i
                    current_points[j] = new_j
                    new_score = _count_ordinary_fast(current_points)
                    
                    delta = new_score - old_score
                    
                    if delta < 0 or random.random() < min(1.0, 1.0 / (1.0 + T)) * (1.0 if delta <= 0 else math.exp(-delta / T)):
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [list(p) for p in current_points]
                        break
                    else:
                        # Revert
                        current_points[i] = old_i
                        current_points[j] = old_j
                
                if not improved:
                    T *= cooling
            
            # Occasional larger jump
            if random.random() < 0.02:
                idx = random.randint(0, 11)
                cx, cy = current_points[idx]
                current_points[idx][0] = random.randint(-15, 15)
                current_points[idx][1] = random.randint(-15, 15)
                # Check validity
                valid = True
                for j in range(12):
                    if j != idx and current_points[j] == current_points[idx]:
                        valid = False
                        break
                if not valid:
                    current_points[idx][0] = cx
                    current_points[idx][1] = cy
                current_score = _count_ordinary_fast(current_points)
                if current_score < best_score:
                    best_score = current_score
                    best_points = [list(p) for p in current_points]
    
    # Final validation
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    # Ensure all are integers and within bounds
    result = []
    for p in best_points:
        x, y = int(p[0]), int(p[1])
        if abs(x) > 10**9 or abs(y) > 10**9:
            x, y = 0, 0
        result.append([x, y])
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for p in result:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique_result.append(p)
    
    # Pad if needed (shouldn't happen)
    while len(unique_result) < 12:
        unique_result.append([random.randint(-10, 10), random.randint(-10, 10)])
    
    return unique_result[:12]
# EVOLVE-BLOCK-END
