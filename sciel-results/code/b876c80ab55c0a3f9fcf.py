"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def get_line_key(p1, p2):
    """Returns a canonical key for the line passing through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return None
        
    g = _gcd(abs(dx), abs(dy))
    if g > 0:
        dx_n = dx // g
        dy_n = dy // g
    else:
        dx_n, dy_n = dx, dy
        
    # Normalize sign
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n = -dx_n
        dy_n = -dy_n
        
    # Normal vector (nx, ny) is perpendicular to (dx_n, dy_n)
    # (dy_n, -dx_n)
    nx = dy_n
    ny = -dx_n
    
    # Normalize normal vector
    g2 = _gcd(abs(nx), abs(ny))
    if g2 > 0:
        nx = nx // g2
        ny = ny // g2
        
    if nx < 0 or (nx == 0 and ny < 0):
        nx = -nx
        ny = -ny
        
    c = nx * x1 + ny * y1
    return (nx, ny, c)

def count_four_lines(points):
    """Count lines with exactly 4 points, return 0 if any line has 5+."""
    n = len(points)
    if n < 4:
        return 0
    
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
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
    random.seed(42)
    
    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    # Precompute line map and point lines for efficient updates
    def build_structure(pts):
        line_map = defaultdict(set)
        point_lines = [set() for _ in range(len(pts))]
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                key = get_line_key(pts[i], pts[j])
                if key is None:
                    continue
                line_map[key].add(i)
                line_map[key].add(j)
                point_lines[i].add(key)
                point_lines[j].add(key)
        return line_map, point_lines

    current_points = [p[:] for p in points]
    line_map, point_lines = build_structure(current_points)
    
    def get_score(line_map):
        score = 0
        for key, pt_set in line_map.items():
            if len(pt_set) >= 5:
                return -100 # Penalty for invalid
            if len(pt_set) == 4:
                score += 1
        return score

    current_score = get_score(line_map)
    best_score = current_score
    best_points = [p[:] for p in current_points]
    
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    max_iters = 200000
    
    for it in range(max_iters):
        if T < T_min:
            break
        
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Random move
        move_range = random.choice([1, 2, 3, 5, 8, 12, 20])
        new_x = old_x + random.randint(-move_range, move_range)
        new_y = old_y + random.randint(-move_range, move_range)
        
        new_x = max(-1000, min(1000, new_x))
        new_y = max(-1000, min(1000, new_y))
        
        # Check duplicate
        new_point = [new_x, new_y]
        if any(current_points[i] == new_point for i in range(16) if i != idx):
            continue
            
        # Save state for rollback
        old_point_lines = [s.copy() for s in point_lines]
        old_line_map = {k: v.copy() for k, v in line_map.items()}
        
        # Remove idx from all its lines
        for key in point_lines[idx]:
            line_map[key].remove(idx)
            if not line_map[key]:
                del line_map[key]
            point_lines[idx].remove(key)
        
        # Add idx to new lines
        current_points[idx] = new_point
        for j in range(16):
            if j == idx:
                continue
            key = get_line_key(new_point, current_points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(idx)
            line_map[key].add(j)
            point_lines[idx].add(key)
            point_lines[j].add(key)
            
        new_score = get_score(line_map)
        
        delta = new_score - current_score
        if delta > 0 or (new_score > -50 and random.random() < pow(2.718281828, delta / T)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Rollback
            current_points[idx] = [old_x, old_y]
            line_map = old_line_map
            point_lines = old_point_lines
            # Re-add idx to original lines (already in old_line_map)
            # We just restored the structures, so we are back to state before move.
            
        T *= cooling
        
        # Occasionally restart from best if stuck?
        if it % 5000 == 0 and current_score < best_score - 2:
             # Soft restart: move points slightly
             for i in range(16):
                 if random.random() < 0.5:
                     dx = random.randint(-2, 2)
                     dy = random.randint(-2, 2)
                     current_points[i] = [max(-1000, min(1000, current_points[i][0] + dx)), 
                                          max(-1000, min(1000, current_points[i][1] + dy))]
             # Rebuild structure
             line_map, point_lines = build_structure(current_points)
             current_score = get_score(line_map)
             T = 10.0

    return best_points

# EVOLVE-BLOCK-END
