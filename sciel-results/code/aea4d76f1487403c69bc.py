"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

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
    # Normalize sign
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

def _valid_move(points, idx, nx, ny):
    """Check if moving points[idx] to (nx, ny) is valid."""
    if abs(nx) > 10**9 or abs(ny) > 10**9:
        return False
    for j in range(len(points)):
        if j != idx and points[j][0] == nx and points[j][1] == ny:
            return False
    return True

def _structured_initial(seed=0):
    """Generate a structured initial configuration."""
    rng = random.Random(seed)
    configs = []
    
    # Config 1: 3x4 grid
    configs.append([(x, y) for x in range(4) for y in range(3)])
    
    # Config 2: 4x3 grid
    configs.append([(x, y) for x in range(3) for y in range(4)])
    
    # Config 3: Points on two vertical lines
    configs.append([(0, y) for y in range(6)] + [(1, y) for y in range(6)])
    
    # Config 4: Points on a "staircase" pattern
    pts = []
    for i in range(12):
        pts.append((i, i * i % 7))
    configs.append(pts)
    
    # Config 5: Random small grid
    pts = []
    used = set()
    while len(pts) < 12:
        x = rng.randint(-5, 5)
        y = rng.randint(-5, 5)
        if (x, y) not in used:
            used.add((x, y))
            pts.append((x, y))
    configs.append(pts)
    
    # Config 6: Points on three lines forming a triangle
    # Line 1: y = 0, x in {0,1,2,3}
    # Line 2: x = 0, y in {0,1,2,3} (shares (0,0))
    # Line 3: x = y, points (0,0),(1,1),(2,2),(3,3) (shares (0,0),(1,1),(2,2),(3,3))
    # This is messy; let's use a different approach
    # Use 4 points on each of 3 lines, with shared intersections
    pts = [(0, 0), (1, 0), (2, 0), (3, 0),  # y=0
           (0, 1), (0, 2), (0, 3),           # x=0 (shares (0,0))
           (1, 1), (2, 2), (3, 3),           # y=x (shares (0,0),(1,1),(2,2),(3,3))
           (4, 0), (5, 0)]                   # more on y=0
    # That's 12 points but with many overlaps; let me count distinct
    distinct = list(set(pts))
    if len(distinct) == 12:
        configs.append(distinct)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Try multiple structured initial configurations
    initial_configs = _structured_initial()
    
    for cfg_idx, init_cfg in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        current_points = [[p[0], p[1]] for p in init_cfg]
        current_score = _count_ordinary(current_points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [list(p) for p in current_points]
        
        # Local search: hill climbing with random restarts
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Pick a random point to move
            idx = random.randint(0, 11)
            cx, cy = current_points[idx]
            
            improved = False
            # Try several random moves
            for _ in range(30):
                # Move within a small neighborhood
                nx = cx + random.randint(-3, 3)
                ny = cy + random.randint(-3, 3)
                
                if not _valid_move(current_points, idx, nx, ny):
                    continue
                
                current_points[idx][0] = nx
                current_points[idx][1] = ny
                new_score = _count_ordinary(current_points)
                
                if new_score < current_score:
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
            
            if not improved and random.random() < 0.1:
                # Occasional larger jump
                idx = random.randint(0, 11)
                current_points[idx][0] = random.randint(-10, 10)
                current_points[idx][1] = random.randint(-10, 10)
                # Check validity
                valid = True
                for j in range(12):
                    if j != idx and current_points[j] == current_points[idx]:
                        valid = False
                        break
                if not valid:
                    # Revert
                    current_points[idx][0] = cx
                    current_points[idx][1] = cy
                    current_score = _count_ordinary(current_points)
    
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
